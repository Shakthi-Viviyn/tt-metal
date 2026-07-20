# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import divup
from models.demos.vision.classification.vit.common.common import load_torch_model
from models.demos.vision.classification.vit.common.tests.vit_test_infra import create_test_infra
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

try:
    import cpp_vit_preprocess
except ImportError as error:
    raise ImportError(
        "Failed to import cpp_vit_preprocess. If the extension is not built, run "
        "`python3 setup.py build_ext --inplace` from this directory. If the error mentions "
        "libturbojpeg.so, make sure the TurboJPEG library directory is visible at runtime, "
        "for example: `LD_LIBRARY_PATH=/opt/libjpeg-turbo/lib64:$LD_LIBRARY_PATH python3 "
        "benchmark_e2e_profile_cpp_preprocess.py`."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_DIR = SCRIPT_DIR / "images"
DEFAULT_BATCH_SIZE = 10
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
VALIDATION_PCC_THRESHOLD = 0.85
PACKED_INPUT_ATOL = 0.01

CSV_COLUMNS = [
    "Batch",
    "Workers",
    "Total Run Time (sec)",
    "Pre-processing Time (sec)",
    "Inference Time (sec)",
    "CPU Inference Time (sec)",
    "TT Inference Throughput (images/sec)",
    "Total Throughput With Pre-processing (images/sec)",
    "CPU Throughput (images/sec)",
    "Total Images Processed",
    "TT Inference Iterations",
    "CPU Iterations",
    "Status",
]


def default_output_path():
    timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    return SCRIPT_DIR.parent / "reports" / f"vit_blackhole_trace_1cq_cpp_preprocessing_e2e_{timestamp}.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure Blackhole ViT trace+1CQ steady-state inference with C++ TurboJPEG preprocessing. "
            "JPEG decode, resize, ViT normalization, BF16 packing, and host-side worker threading run in C++; "
            "model execution and CPU reference validation remain in Python/TT-NN."
        )
    )
    parser.add_argument("--output", type=Path, default=default_output_path(), help="CSV path to write.")
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--l1-small-size", type=int, default=32768)
    parser.add_argument("--trace-region-size", type=int, default=2771968)
    parser.add_argument("--measurement-iterations", type=int, default=100)
    parser.add_argument("--tt-warmup-iterations", type=int, default=5)
    parser.add_argument("--preprocess-warmup-iterations", type=int, default=5)
    parser.add_argument(
        "--preprocess-worker-counts",
        default="1,4,8,10",
        help="Comma-separated C++ worker counts to sweep. Values above 10 are capped at the batch size.",
    )
    parser.add_argument("--cpu-warmup-iterations", type=int, default=1)
    parser.add_argument("--cpu-iterations", type=int, default=5)
    return parser.parse_args()


def parse_int_list(values):
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def normalize_worker_counts(worker_counts, batch_size):
    normalized_worker_counts = []
    for worker_count in worker_counts:
        if worker_count <= 0:
            raise ValueError(f"Preprocessing worker counts must be positive, got {worker_count}")
        effective_worker_count = min(worker_count, batch_size)
        if effective_worker_count not in normalized_worker_counts:
            normalized_worker_counts.append(effective_worker_count)
    if not normalized_worker_counts:
        raise ValueError("At least one preprocessing worker count must be provided")
    return normalized_worker_counts


def format_seconds(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_throughput(batch_size, seconds):
    if seconds is None or seconds == 0:
        return ""
    return f"{batch_size / seconds:.4f}"


def collect_image_paths(image_dir):
    image_paths = (
        sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if image_dir.exists()
        else []
    )
    only_legacy_tiny_imagenet = image_paths and all(path.name.startswith("tiny_imagenet_") for path in image_paths)
    if not image_paths or only_legacy_tiny_imagenet:
        logger.info(f"No current native-resolution benchmark JPEGs found in {image_dir}; downloading images")
        from models.demos.vision.classification.vit.blackhole.demo.cpp_preprocessing.download_images import (
            download_images,
        )

        download_images()
        image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported ImageNet JPEG files found in {image_dir} after download")
    return image_paths


def load_input_image_bytes():
    return [path.read_bytes() for path in collect_image_paths(IMAGE_DIR)]


def measure_cpu_reference(torch_vit, inputs, warmup_iterations, measurement_iterations):
    timings = []
    logits = None
    with torch.no_grad():
        for _ in range(warmup_iterations):
            torch_vit(inputs).logits

        for _ in range(measurement_iterations):
            start = time.perf_counter()
            logits = torch_vit(inputs).logits
            timings.append(time.perf_counter() - start)

    return sum(timings) / len(timings), logits


def unpack_packed_inputs_to_nchw(packed_inputs):
    batch_size, image_height, patch_groups, packed_channels = packed_inputs.shape
    patch_size = 16
    padded_channels = 4
    if packed_channels != patch_size * padded_channels:
        raise ValueError(f"Expected packed channel dimension 64, got {packed_channels}")

    nhwc = (
        packed_inputs.to(torch.float32)
        .reshape(batch_size, image_height, patch_groups, patch_size, padded_channels)[..., :3]
        .reshape(batch_size, image_height, patch_groups * patch_size, 3)
    )
    return nhwc.permute(0, 3, 1, 2).contiguous()


def validate_packed_inputs_match_cpu_reference(cpu_inputs, packed_inputs):
    unpacked_inputs = unpack_packed_inputs_to_nchw(packed_inputs)
    abs_diff = torch.abs(cpu_inputs.to(torch.float32) - unpacked_inputs)
    max_abs_diff = torch.max(abs_diff).item()
    mean_abs_diff = torch.mean(abs_diff).item()
    logger.info(
        "Input consistency validation: "
        f"cpu_inputs_shape={tuple(cpu_inputs.shape)}, packed_inputs_shape={tuple(packed_inputs.shape)}, "
        f"max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}"
    )
    return max_abs_diff <= PACKED_INPUT_ATOL


def read_device_profiler(device):
    ttnn.ReadDeviceProfiler(device)


def to_torch_logits(output_tensor, test_infra):
    mesh_composer = test_infra.output_mesh_composer
    if mesh_composer is None:
        output_tensor = ttnn.to_torch(output_tensor)
    else:
        output_tensor = ttnn.to_torch(output_tensor, mesh_composer=mesh_composer)

    if output_tensor.ndim == 3:
        return output_tensor[:, 0, :1000]
    return torch.reshape(output_tensor, (output_tensor.shape[0], -1))[:, :1000]


def setup_dram_sharded_packed_input(device, packed_inputs, test_infra):
    batch_size, img_h, img_w, img_c = packed_inputs.shape
    batch_size_per_device = batch_size // test_infra.num_devices
    shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(7, 1),
            ),
        }
    )
    n_cores = 16
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [batch_size_per_device * img_h * img_w // n_cores, img_c],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.types.BufferType.L1,
        shard_spec,
    )

    tt_inputs_host = ttnn.from_torch(
        packed_inputs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=test_infra.inputs_mesh_mapper,
    )

    dram_grid_size = device.dram_grid_size()
    dram_shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))}
        ),
        [
            divup(tt_inputs_host.volume() // tt_inputs_host.shape[-1], dram_grid_size.x),
            tt_inputs_host.shape[-1],
        ],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config_dram = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        dram_shard_spec,
    )

    return tt_inputs_host, sharded_mem_config_dram, input_mem_config


def run_trace_1cq_pipeline(device, packed_inputs, batch_size, measurement_iterations, warmup_iterations):
    test_infra = create_test_infra(device, batch_size, use_random_input_tensor=True)
    tt_inputs_host, sharded_mem_config_dram, input_mem_config = setup_dram_sharded_packed_input(
        device,
        packed_inputs,
        test_infra,
    )

    def model_wrapper(l1_input_tensor):
        test_infra.input_tensor = l1_input_tensor
        return test_infra.run()

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=1, all_transfers_on_separate_command_queue=False),
        model=model_wrapper,
        device=device,
        dram_input_memory_config=sharded_mem_config_dram,
        l1_input_memory_config=input_mem_config,
    )

    try:
        pipeline.compile(tt_inputs_host)
        read_device_profiler(device)

        if warmup_iterations:
            pipeline.preallocate_output_tensors_on_host(warmup_iterations)
            pipeline.enqueue([tt_inputs_host] * warmup_iterations).pop_all()

        pipeline.preallocate_output_tensors_on_host(measurement_iterations)
        inference_start = time.perf_counter()
        outputs = pipeline.enqueue([tt_inputs_host] * measurement_iterations).pop_all()
        inference_total = time.perf_counter() - inference_start
        read_device_profiler(device)

        return inference_total / measurement_iterations, to_torch_logits(outputs[-1], test_infra)
    finally:
        pipeline.cleanup()


def calculate_pcc(expected, actual):
    expected = expected.detach().flatten().to(torch.float32).cpu()
    actual = actual.detach().flatten().to(torch.float32).cpu()
    expected = expected - torch.mean(expected)
    actual = actual - torch.mean(actual)

    denominator = torch.sqrt(torch.sum(expected * expected) * torch.sum(actual * actual))
    if denominator == 0:
        return 1.0 if torch.allclose(expected, actual) else 0.0
    return torch.sum(expected * actual).item() / denominator.item()


def compare_outputs(cpu_logits, tt_logits, top_k=5):
    cpu_top1 = cpu_logits.argmax(dim=-1).cpu()
    tt_top1 = tt_logits.argmax(dim=-1).cpu()
    top1_matches = int(torch.eq(cpu_top1, tt_top1).sum().item())
    total = int(cpu_top1.numel())

    cpu_topk = torch.topk(cpu_logits, k=top_k, dim=-1).indices.cpu()
    tt_topk = torch.topk(tt_logits, k=top_k, dim=-1).indices.cpu()
    topk_overlaps = 0
    for sample_index in range(total):
        if set(cpu_topk[sample_index].tolist()) & set(tt_topk[sample_index].tolist()):
            topk_overlaps += 1

    per_image_pcc = [calculate_pcc(cpu_logits[index], tt_logits[index]) for index in range(total)]
    return top1_matches, topk_overlaps, total, calculate_pcc(cpu_logits, tt_logits), per_image_pcc


def benchmark_batch(device, batch_size, image_bytes_list, torch_vit, worker_count, args):
    preprocessing_time, cpu_inputs, packed_inputs = cpp_vit_preprocess.measure_preprocessing(
        image_bytes_list,
        batch_size,
        args.preprocess_warmup_iterations,
        args.measurement_iterations,
        worker_count,
    )

    row = {
        "Batch": str(batch_size),
        "Workers": str(worker_count),
        "Total Run Time (sec)": "",
        "Pre-processing Time (sec)": format_seconds(preprocessing_time),
        "Inference Time (sec)": "",
        "CPU Inference Time (sec)": "",
        "TT Inference Throughput (images/sec)": "",
        "Total Throughput With Pre-processing (images/sec)": "",
        "CPU Throughput (images/sec)": "",
        "Total Images Processed": str(batch_size * args.measurement_iterations),
        "TT Inference Iterations": str(args.measurement_iterations),
        "CPU Iterations": str(args.cpu_iterations),
        "Status": "PREPROCESSING_AND_CPU_OK",
    }

    try:
        if not validate_packed_inputs_match_cpu_reference(cpu_inputs, packed_inputs):
            row["Status"] = "INPUT_PACKING_MISMATCH"
            return row, None

        cpu_time, cpu_logits = measure_cpu_reference(
            torch_vit,
            cpu_inputs,
            args.cpu_warmup_iterations,
            args.cpu_iterations,
        )

        ttnn.synchronize_device(device)
        inference_time, tt_logits = run_trace_1cq_pipeline(
            device,
            packed_inputs,
            batch_size,
            args.measurement_iterations,
            args.tt_warmup_iterations,
        )

        top1_matches, top5_overlaps, total, pcc, per_image_pcc = compare_outputs(cpu_logits, tt_logits)
        mean_per_image_pcc = sum(per_image_pcc) / len(per_image_pcc)
        minimum_per_image_pcc = min(per_image_pcc)
        logger.info(
            "Classification validation: "
            f"top1_matches={top1_matches}/{total}, top5_overlaps={top5_overlaps}/{total}, "
            f"aggregate_pcc={pcc:.4f}, mean_per_image_pcc={mean_per_image_pcc:.4f}, "
            f"minimum_per_image_pcc={minimum_per_image_pcc:.4f}"
        )
        logger.info("Per-image PCC: " + ", ".join(f"{value:.4f}" for value in per_image_pcc))
        total_run_time = preprocessing_time + inference_time
        status = "OK" if top1_matches == total or pcc >= VALIDATION_PCC_THRESHOLD else "VALIDATION_MISMATCH"
    except Exception:
        logger.exception(f"Failed to run ViT TT pipeline for batch_size={batch_size}")
        row["Status"] = "TT_PIPELINE_ERROR"
        return row, None

    row.update(
        {
            "Inference Time (sec)": format_seconds(inference_time),
            "CPU Inference Time (sec)": format_seconds(cpu_time),
            "TT Inference Throughput (images/sec)": format_throughput(batch_size, inference_time),
            "Total Run Time (sec)": format_seconds(total_run_time),
            "Total Throughput With Pre-processing (images/sec)": format_throughput(batch_size, total_run_time),
            "CPU Throughput (images/sec)": format_throughput(batch_size, cpu_time),
            "Status": status,
        }
    )
    return row, tt_logits


def make_error_row(batch_size, worker_count=""):
    return {
        "Batch": str(batch_size),
        "Workers": str(worker_count),
        "Total Run Time (sec)": "",
        "Pre-processing Time (sec)": "",
        "Inference Time (sec)": "",
        "CPU Inference Time (sec)": "",
        "TT Inference Throughput (images/sec)": "",
        "Total Throughput With Pre-processing (images/sec)": "",
        "CPU Throughput (images/sec)": "",
        "Total Images Processed": "",
        "TT Inference Iterations": "",
        "CPU Iterations": "",
        "Status": "ERROR",
    }


def calculate_fastest_total_throughput_worker(rows):
    best_worker = None
    best_throughput = None
    for row in rows:
        if row.get("Status") != "OK":
            continue
        worker_count = int(row["Workers"])
        throughput = float(row["Total Throughput With Pre-processing (images/sec)"])
        if best_throughput is None or throughput > best_throughput:
            best_worker = worker_count
            best_throughput = throughput

    return best_worker, best_throughput


def write_csv(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        best_worker, best_throughput = calculate_fastest_total_throughput_worker(rows)
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([])
        if best_worker is None:
            csv_writer.writerow(["Best Workers By Total Throughput With Pre-processing", "unknown"])
        else:
            csv_writer.writerow(["Best Workers By Total Throughput With Pre-processing", best_worker])
            csv_writer.writerow(
                ["Best Aggregate Total Throughput With Pre-processing (images/sec)", f"{best_throughput:.4f}"]
            )


def main():
    args = parse_args()
    worker_counts = normalize_worker_counts(parse_int_list(args.preprocess_worker_counts), DEFAULT_BATCH_SIZE)
    image_bytes_list = load_input_image_bytes()

    torch_vit = load_torch_model(embedding=True)
    torch_vit.eval()

    original_default_device = ttnn.GetDefaultDevice()
    device = ttnn.CreateDevice(
        device_id=args.device_id,
        l1_small_size=args.l1_small_size,
        trace_region_size=args.trace_region_size,
        num_command_queues=1,
    )
    rows = []
    last_tt_logits = None

    try:
        ttnn.SetDefaultDevice(device)
        for worker_count in worker_counts:
            logger.info(f"Benchmarking with cpp_preprocess_workers={worker_count}")
            logger.info(f"Benchmarking ViT trace+1CQ batch_size={DEFAULT_BATCH_SIZE}")
            try:
                row, tt_logits = benchmark_batch(
                    device,
                    DEFAULT_BATCH_SIZE,
                    image_bytes_list,
                    torch_vit,
                    worker_count,
                    args,
                )
                rows.append(row)
                if tt_logits is not None:
                    last_tt_logits = tt_logits
            except Exception:
                logger.exception(f"Failed to benchmark batch_size={DEFAULT_BATCH_SIZE}, workers={worker_count}")
                rows.append(make_error_row(DEFAULT_BATCH_SIZE, worker_count))
    finally:
        ttnn.SetDefaultDevice(original_default_device)
        ttnn.close_device(device)

    if last_tt_logits is not None:
        print("Last TT logits tensor:")
        print(last_tt_logits)

    write_csv(args.output, rows)
    logger.info(f"Wrote {len(rows)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
