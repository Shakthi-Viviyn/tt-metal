import argparse
import csv
import time
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common.resnet50_test_infra import (
    create_test_infra,
    load_resnet50_model,
)
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config

try:
    import cpp_resnet_preprocess
except ImportError as error:
    raise ImportError(
        "Failed to import cpp_resnet_preprocess. If the extension is not built, run "
        "`python3 setup.py build_ext --inplace` from this directory. If the error mentions "
        "libturbojpeg.so, make sure the TurboJPEG library directory is visible at runtime, "
        "for example: `LD_LIBRARY_PATH=/opt/libjpeg-turbo/lib64:$LD_LIBRARY_PATH python3 "
        "benchmark_e2e_profile_cpp_preprocess.py`."
    ) from error


DEFAULT_BATCH_SIZES = (16, 32)
SCRIPT_DIR = Path(__file__).resolve().parent
TT_METAL_ROOT = SCRIPT_DIR.parents[7]
DEFAULT_IMAGE_DIR = TT_METAL_ROOT / "models/demos/vision/classification/resnet50/ttnn_resnet/demo/images"
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
MODEL_CONFIG = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}
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
    return SCRIPT_DIR / "reports" / f"resnet50_blackhole_trace_2cqs_cpp_preprocessing_e2e_{timestamp}.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure Blackhole ResNet50 trace+2CQ steady-state inference with C++ TurboJPEG preprocessing. "
            "The model path stays in Python/TT-NN; JPEG decode, resize, crop, normalize, and BF16 packing run in C++."
        )
    )
    parser.add_argument(
        "--batch-sizes",
        default=",".join(str(batch_size) for batch_size in DEFAULT_BATCH_SIZES),
        help="Comma-separated device batch sizes to measure. Default: 16,32.",
    )
    parser.add_argument("--output", type=Path, default=default_output_path(), help="CSV path to write.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional single image used for preprocessing measurements. If omitted, --image-dir is used.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help="Directory of images to cycle through for preprocessing measurements.",
    )
    parser.add_argument("--device-id", type=int, default=1)
    parser.add_argument("--l1-small-size", type=int, default=32768)
    parser.add_argument("--trace-region-size", type=int, default=2777088)
    parser.add_argument("--measurement-iterations", type=int, default=15)
    parser.add_argument("--preprocess-warmup-iterations", type=int, default=5)
    parser.add_argument(
        "--preprocess-worker-counts",
        default="1,4,8,16,32",
        help="Comma-separated C++ worker counts to sweep. Default: 1,4,8,16,32.",
    )
    parser.add_argument("--cpu-warmup-iterations", type=int, default=1)
    parser.add_argument("--cpu-iterations", type=int, default=5)
    parser.add_argument(
        "--torchvision-weights-path",
        type=Path,
        default=None,
        help=(
            "Optional path to IMAGENET1K_V1.pt. If omitted, TorchVision's ResNet50_Weights.IMAGENET1K_V1 "
            "loading path is used."
        ),
    )
    return parser.parse_args()


def parse_int_list(values):
    return [int(value.strip()) for value in values.split(",") if value.strip()]


def parse_batch_sizes(batch_sizes):
    return parse_int_list(batch_sizes)


def format_seconds(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_throughput(batch_size, seconds):
    if seconds is None or seconds == 0:
        return ""
    return f"{batch_size / seconds:.4f}"


def make_model_location_generator(torchvision_weights_path):
    if torchvision_weights_path is None:
        return None

    def model_location_generator(model_version, model_subdir=""):
        return str(torchvision_weights_path)

    return model_location_generator


def collect_image_paths(image_dir):
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in {image_dir}")
    return image_paths


def load_input_image_bytes(args):
    if args.image is not None:
        return [args.image.read_bytes()]
    return [path.read_bytes() for path in collect_image_paths(args.image_dir)]


def measure_cpu_reference(torch_resnet50, inputs, warmup_iterations, measurement_iterations):
    timings = []
    logits = None
    with torch.no_grad():
        for _ in range(warmup_iterations):
            torch_resnet50(inputs)

        for _ in range(measurement_iterations):
            start = time.perf_counter()
            logits = torch_resnet50(inputs)
            timings.append(time.perf_counter() - start)

    return sum(timings) / len(timings), logits


def read_device_profiler(device):
    ttnn.ReadDeviceProfiler(device)


def run_trace_2cq_pipeline(device, inputs, test_infra, measurement_iterations):
    tt_inputs_host, sharded_mem_config_dram, input_mem_config = test_infra.setup_dram_sharded_input(device, inputs)

    def model_wrapper(l1_input_tensor):
        test_infra.input_tensor = l1_input_tensor
        return test_infra.run()

    pipeline = create_pipeline_from_config(
        config=PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False),
        model=model_wrapper,
        device=device,
        dram_input_memory_config=sharded_mem_config_dram,
        l1_input_memory_config=input_mem_config,
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

        for index, output in enumerate(outputs):
            passed, pcc_message = test_infra.validate(output)
            logger.info(f"Output {index} validation: {pcc_message}")
            assert passed, f"Output {index} validation failed: {pcc_message}"

        return inference_total / measurement_iterations
    finally:
        pipeline.cleanup()


def benchmark_batch(
    device,
    batch_size,
    image_bytes_list,
    torch_resnet50,
    model_location_generator,
    worker_count,
    args,
):
    preprocessing_time, inputs = cpp_resnet_preprocess.measure_preprocessing(
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
        cpu_time, cpu_logits = measure_cpu_reference(
            torch_resnet50,
            inputs,
            args.cpu_warmup_iterations,
            args.cpu_iterations,
        )

        test_infra = create_test_infra(
            device,
            batch_size,
            MODEL_CONFIG["ACTIVATIONS_DTYPE"],
            MODEL_CONFIG["WEIGHTS_DTYPE"],
            MODEL_CONFIG["MATH_FIDELITY"],
            dealloc_input=True,
            final_output_mem_config=ttnn.L1_MEMORY_CONFIG,
            model_location_generator=model_location_generator,
        )
        test_infra.torch_input_tensor = inputs
        test_infra.torch_output_tensor = cpu_logits

        ttnn.synchronize_device(device)
        inference_time = run_trace_2cq_pipeline(device, inputs, test_infra, args.measurement_iterations)

        total_run_time = preprocessing_time + inference_time
    except Exception:
        logger.exception(f"Failed to run TT pipeline for batch_size={batch_size}")
        row["Status"] = "TT_PIPELINE_ERROR"
        return row

    row.update(
        {
            "Inference Time (sec)": format_seconds(inference_time),
            "CPU Inference Time (sec)": format_seconds(cpu_time),
            "TT Inference Throughput (images/sec)": format_throughput(batch_size, inference_time),
            "Total Run Time (sec)": format_seconds(total_run_time),
            "Total Throughput With Pre-processing (images/sec)": format_throughput(batch_size, total_run_time),
            "CPU Throughput (images/sec)": format_throughput(batch_size, cpu_time),
            "Status": "OK",
        }
    )
    return row


def make_error_row(batch_size, error, worker_count=""):
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
            csv_writer.writerow(["Best Aggregate Total Throughput With Pre-processing (images/sec)", f"{best_throughput:.4f}"])


def main():
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    worker_counts = parse_int_list(args.preprocess_worker_counts)
    image_bytes_list = load_input_image_bytes(args)
    model_location_generator = make_model_location_generator(args.torchvision_weights_path)

    torch_resnet50 = load_resnet50_model(model_location_generator)
    torch_resnet50.eval()
    torch_resnet50.to(torch.bfloat16)

    original_default_device = ttnn.GetDefaultDevice()
    device = ttnn.CreateDevice(
        device_id=args.device_id,
        l1_small_size=args.l1_small_size,
        trace_region_size=args.trace_region_size,
        num_command_queues=2,
    )
    rows = []

    try:
        ttnn.SetDefaultDevice(device)
        for worker_count in worker_counts:
            logger.info(f"Benchmarking with cpp_preprocess_workers={worker_count}")
            for batch_size in batch_sizes:
                logger.info(f"Benchmarking ResNet50 trace+2CQ batch_size={batch_size}")
                try:
                    rows.append(
                        benchmark_batch(
                            device,
                            batch_size,
                            image_bytes_list,
                            torch_resnet50,
                            model_location_generator,
                            worker_count,
                            args,
                        )
                    )
                except Exception as error:
                    logger.exception(f"Failed to benchmark batch_size={batch_size}, workers={worker_count}")
                    rows.append(make_error_row(batch_size, error, worker_count))
    finally:
        ttnn.SetDefaultDevice(original_default_device)
        ttnn.close_device(device)

    write_csv(args.output, rows)
    logger.info(f"Wrote {len(rows)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
