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
from models.demos.vision.classification.mobilenetv2.common import (
    MOBILENETV2_BATCH_SIZE,
    MOBILENETV2_L1_SMALL_SIZE,
    MOBILENETV2_TRACE_REGION_SIZE,
    load_torch_model,
)
from models.demos.vision.classification.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.vision.classification.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.vision.classification.mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_memory_configs,
    create_mobilenetv2_model_parameters,
    unpack_mobilenetv2_pipeline_input,
)
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from tests.ttnn.utils_for_testing import assert_with_pcc

try:
    import cpp_mobilenetv2_preprocess
except ImportError as error:
    raise ImportError(
        "Failed to import cpp_mobilenetv2_preprocess. If the extension is not built, run "
        "`python3 setup.py build_ext --inplace` from this directory. If the error mentions "
        "libturbojpeg.so, make sure the TurboJPEG library directory is visible at runtime, "
        "for example: `LD_LIBRARY_PATH=/opt/libjpeg-turbo/lib64:$LD_LIBRARY_PATH python3 "
        "benchmark_e2e_profile_cpp_preprocess.py`."
    ) from error


SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_ROOT = SCRIPT_DIR.parents[7]
DEFAULT_IMAGE_DIR = TT_METAL_ROOT / "models/demos/vision/classification/resnet50/ttnn_resnet/demo/images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
PACKED_INPUT_ATOL = 0.02
DEFAULT_PCC_THRESHOLD = 0.90

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
    return SCRIPT_DIR.parent / "reports" / f"mobilenetv2_blackhole_trace_1cq_cpp_preprocessing_e2e_{timestamp}.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure Blackhole MobileNetV2 batch-10 trace+1CQ inference with C++ TurboJPEG preprocessing. "
            "The model path stays in Python/TT-NN; JPEG decode, resize, crop, normalization, BF16 padding, "
            "and Blackhole input packing run in C++."
        )
    )
    parser.add_argument("--output", type=Path, default=default_output_path(), help="CSV path to write.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional single JPEG used for preprocessing measurements. If omitted, --image-dir is used.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory of JPEG images to cycle through for preprocessing measurements.",
    )
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--l1-small-size", type=int, default=MOBILENETV2_L1_SMALL_SIZE)
    parser.add_argument("--trace-region-size", type=int, default=MOBILENETV2_TRACE_REGION_SIZE)
    parser.add_argument("--measurement-iterations", type=int, default=15)
    parser.add_argument("--preprocess-warmup-iterations", type=int, default=5)
    parser.add_argument(
        "--preprocess-worker-counts",
        default="1,2,4,8,10",
        help="Comma-separated C++ worker counts to sweep. Values above 10 are capped at the batch size.",
    )
    parser.add_argument("--cpu-warmup-iterations", type=int, default=1)
    parser.add_argument("--cpu-iterations", type=int, default=5)
    parser.add_argument(
        "--pcc-threshold",
        type=float,
        default=DEFAULT_PCC_THRESHOLD,
        help=(
            "Minimum PyTorch-versus-TT output PCC. The default is an execution guard for real-image benchmark "
            "inputs; use the model PCC test for qualification."
        ),
    )
    return parser.parse_args()


def parse_int_list(values):
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def format_seconds(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_throughput(batch_size, seconds):
    if seconds is None or seconds == 0:
        return ""
    return f"{batch_size / seconds:.4f}"


def collect_image_paths(image_dir):
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported JPEG files found in {image_dir}")
    return image_paths


def load_input_image_bytes(args):
    if args.image is not None:
        return [args.image.read_bytes()]
    return [path.read_bytes() for path in collect_image_paths(args.image_dir)]


def measure_cpu_reference(torch_mobilenetv2, inputs, warmup_iterations, measurement_iterations):
    timings = []
    logits = None
    with torch.no_grad():
        for _ in range(warmup_iterations):
            torch_mobilenetv2(inputs)

        for _ in range(measurement_iterations):
            start = time.perf_counter()
            logits = torch_mobilenetv2(inputs)
            timings.append(time.perf_counter() - start)

    return sum(timings) / len(timings), logits


def unpack_packed_inputs_to_nchw(packed_inputs):
    expected_shape = (1, 1, MOBILENETV2_BATCH_SIZE * 224 * 224 // 2, 32)
    if tuple(packed_inputs.shape) != expected_shape:
        raise ValueError(f"Expected packed input shape {expected_shape}, got {tuple(packed_inputs.shape)}")

    nhwc = (
        packed_inputs.to(torch.float32)
        .reshape(MOBILENETV2_BATCH_SIZE, 224 * 224 // 2, 2, 16)
        .reshape(MOBILENETV2_BATCH_SIZE, 224, 224, 16)[..., :3]
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


def to_torch_logits(output_tensor):
    output_tensor = ttnn.to_torch(output_tensor)
    return torch.reshape(output_tensor, (MOBILENETV2_BATCH_SIZE, -1))[:, :1000]


def create_tt_host_input(packed_inputs):
    return ttnn.from_torch(
        packed_inputs,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def run_trace_1cq_pipeline(
    device,
    packed_inputs,
    ttnn_model,
    cpu_logits,
    measurement_iterations,
    pcc_threshold,
):
    tt_inputs_host = create_tt_host_input(packed_inputs)
    input_dram_mem_config, input_l1_mem_config = create_mobilenetv2_input_memory_configs(tt_inputs_host, device)

    def model_wrapper(packed_l1_input_tensor):
        return ttnn_model(unpack_mobilenetv2_pipeline_input(packed_l1_input_tensor))

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=1, all_transfers_on_separate_command_queue=False),
        model=model_wrapper,
        device=device,
        dram_input_memory_config=input_dram_mem_config,
        l1_input_memory_config=input_l1_mem_config,
    )

    try:
        pipeline.compile(tt_inputs_host)
        read_device_profiler(device)

        pipeline.preallocate_output_tensors_on_host(measurement_iterations)
        host_inputs = [tt_inputs_host] * measurement_iterations

        inference_start = time.perf_counter()
        outputs = pipeline.enqueue(host_inputs).pop_all()
        inference_total = time.perf_counter() - inference_start
        read_device_profiler(device)

        last_logits = None
        for index, output in enumerate(outputs):
            last_logits = to_torch_logits(output)
            _, pcc_message = assert_with_pcc(cpu_logits, last_logits, pcc_threshold)
            logger.info(f"Output {index} validation: {pcc_message}")

        return inference_total / measurement_iterations, last_logits
    finally:
        pipeline.cleanup()


def benchmark_batch(device, image_bytes_list, torch_mobilenetv2, ttnn_model, worker_count, args):
    preprocessing_time, cpu_inputs, packed_inputs = cpp_mobilenetv2_preprocess.measure_preprocessing(
        image_bytes_list,
        MOBILENETV2_BATCH_SIZE,
        args.preprocess_warmup_iterations,
        args.measurement_iterations,
        worker_count,
    )

    row = {
        "Batch": str(MOBILENETV2_BATCH_SIZE),
        "Workers": str(worker_count),
        "Total Run Time (sec)": "",
        "Pre-processing Time (sec)": format_seconds(preprocessing_time),
        "Inference Time (sec)": "",
        "CPU Inference Time (sec)": "",
        "TT Inference Throughput (images/sec)": "",
        "Total Throughput With Pre-processing (images/sec)": "",
        "CPU Throughput (images/sec)": "",
        "Total Images Processed": str(MOBILENETV2_BATCH_SIZE * args.measurement_iterations),
        "TT Inference Iterations": str(args.measurement_iterations),
        "CPU Iterations": str(args.cpu_iterations),
        "Status": "PREPROCESSING_AND_CPU_OK",
    }

    try:
        if not validate_packed_inputs_match_cpu_reference(cpu_inputs, packed_inputs):
            row["Status"] = "INPUT_PACKING_MISMATCH"
            return row, None

        cpu_time, cpu_logits = measure_cpu_reference(
            torch_mobilenetv2,
            cpu_inputs,
            args.cpu_warmup_iterations,
            args.cpu_iterations,
        )

        ttnn.synchronize_device(device)
        inference_time, tt_logits = run_trace_1cq_pipeline(
            device,
            packed_inputs,
            ttnn_model,
            cpu_logits,
            args.measurement_iterations,
            args.pcc_threshold,
        )
        total_run_time = preprocessing_time + inference_time
        preprocessing_share = 100.0 * preprocessing_time / total_run_time
        logger.info(
            f"Pipeline split for workers={worker_count}: preprocessing={preprocessing_share:.2f}%, "
            f"inference={100.0 - preprocessing_share:.2f}%"
        )
    except Exception:
        logger.exception(
            f"Failed to run MobileNetV2 TT pipeline for batch_size={MOBILENETV2_BATCH_SIZE}, workers={worker_count}"
        )
        row["Status"] = "TT_PIPELINE_ERROR"
        return row, None

    row.update(
        {
            "Inference Time (sec)": format_seconds(inference_time),
            "CPU Inference Time (sec)": format_seconds(cpu_time),
            "TT Inference Throughput (images/sec)": format_throughput(MOBILENETV2_BATCH_SIZE, inference_time),
            "Total Run Time (sec)": format_seconds(total_run_time),
            "Total Throughput With Pre-processing (images/sec)": format_throughput(
                MOBILENETV2_BATCH_SIZE, total_run_time
            ),
            "CPU Throughput (images/sec)": format_throughput(MOBILENETV2_BATCH_SIZE, cpu_time),
            "Status": "OK",
        }
    )
    return row, tt_logits


def make_error_row(worker_count=""):
    return {
        "Batch": str(MOBILENETV2_BATCH_SIZE),
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
    worker_counts = parse_int_list(args.preprocess_worker_counts)
    image_bytes_list = load_input_image_bytes(args)

    torch_mobilenetv2 = load_torch_model(Mobilenetv2())
    torch_mobilenetv2.eval()

    original_default_device = ttnn.GetDefaultDevice()
    device = ttnn.CreateDevice(
        device_id=args.device_id,
        l1_small_size=args.l1_small_size,
        trace_region_size=args.trace_region_size,
        num_command_queues=1,
    )
    rows = []
    last_model_output = None

    try:
        ttnn.SetDefaultDevice(device)
        model_parameters = create_mobilenetv2_model_parameters(torch_mobilenetv2, device=device)
        ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(
            model_parameters,
            device,
            batchsize=MOBILENETV2_BATCH_SIZE,
        )

        for worker_count in worker_counts:
            logger.info(
                f"Benchmarking MobileNetV2 trace+1CQ batch_size={MOBILENETV2_BATCH_SIZE}, "
                f"cpp_preprocess_workers={worker_count}"
            )
            try:
                row, tt_logits = benchmark_batch(
                    device,
                    image_bytes_list,
                    torch_mobilenetv2,
                    ttnn_model,
                    worker_count,
                    args,
                )
                rows.append(row)
                if tt_logits is not None:
                    last_model_output = tt_logits
            except Exception:
                logger.exception(f"Failed to benchmark batch_size={MOBILENETV2_BATCH_SIZE}, workers={worker_count}")
                rows.append(make_error_row(worker_count))
    finally:
        ttnn.SetDefaultDevice(original_default_device)
        ttnn.close_device(device)

    if last_model_output is not None:
        print("Last model output tensor:")
        print(last_model_output)

    write_csv(args.output, rows)
    logger.info(f"Wrote {len(rows)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
