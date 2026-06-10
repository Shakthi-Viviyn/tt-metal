# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import io
import statistics
import time
from pathlib import Path

import torch
from loguru import logger
from PIL import Image
from transformers import AutoImageProcessor

import ttnn
from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common.resnet50_test_infra import (
    create_test_infra,
    load_resnet50_model,
)
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


MODEL_NAME = "microsoft/resnet-50"
DEFAULT_BATCH_SIZES = (16, 32)
DEFAULT_IMAGE_DIR = Path("models/demos/vision/classification/resnet50/ttnn_resnet/demo/images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_CONFIG = {
    "MATH_FIDELITY": ttnn.MathFidelity.LoFi,
    "WEIGHTS_DTYPE": ttnn.bfloat8_b,
    "ACTIVATIONS_DTYPE": ttnn.bfloat8_b,
}
CSV_COLUMNS = [
    "Batch",
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure Blackhole ResNet50 trace+2CQ steady-state inference plus host image preprocessing, "
            "and write one CSV row per batch size."
        )
    )
    parser.add_argument(
        "--batch-sizes",
        default=",".join(str(batch_size) for batch_size in DEFAULT_BATCH_SIZES),
        help="Comma-separated device batch sizes to measure. Default: 16,32.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output_path(),
        help="CSV path to write.",
    )
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
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached HuggingFace image processor files.",
    )
    return parser.parse_args()


def parse_batch_sizes(batch_sizes):
    return [int(batch_size.strip()) for batch_size in batch_sizes.split(",") if batch_size.strip()]


def format_seconds(value, digits=4):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def format_throughput(batch_size, seconds):
    if seconds is None or seconds == 0:
        return ""
    return f"{batch_size / seconds:.4f}"


def default_output_path():
    timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000_000:09d}"
    return Path("generated") / f"resnet50_blackhole_trace_2cqs_hf_preprocessing_e2e_{timestamp}.csv"


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


def make_image_byte_batch(image_bytes_list, batch_size, iteration_index):
    start = iteration_index * batch_size
    return [image_bytes_list[(start + offset) % len(image_bytes_list)] for offset in range(batch_size)]


def decode_rgb_image(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        image.load()
        return image.copy()


def preprocess_image_batch(image_processor, image_byte_batch):
    images = [decode_rgb_image(image_bytes) for image_bytes in image_byte_batch]
    try:
        return image_processor(images, return_tensors="pt")["pixel_values"].bfloat16()
    finally:
        for image in images:
            image.close()


def measure_preprocessing(image_processor, image_bytes_list, batch_size, warmup_iterations, measurement_iterations):
    for iteration_index in range(warmup_iterations):
        preprocess_image_batch(
            image_processor,
            make_image_byte_batch(image_bytes_list, batch_size, iteration_index),
        )

    timings = []
    last_inputs = None
    for iteration_index in range(warmup_iterations, warmup_iterations + measurement_iterations):
        image_byte_batch = make_image_byte_batch(image_bytes_list, batch_size, iteration_index)
        start = time.perf_counter()
        last_inputs = preprocess_image_batch(image_processor, image_byte_batch)
        timings.append(time.perf_counter() - start)

    return statistics.mean(timings), last_inputs


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

    return statistics.mean(timings), logits


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
        compile_start = time.perf_counter()
        pipeline.compile(tt_inputs_host)
        compile_time = time.perf_counter() - compile_start
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

        return compile_time, inference_total / measurement_iterations
    finally:
        pipeline.cleanup()


def benchmark_batch(
    device,
    batch_size,
    image_processor,
    image_bytes_list,
    torch_resnet50,
    model_location_generator,
    args,
):
    preprocessing_time, inputs = measure_preprocessing(
        image_processor,
        image_bytes_list,
        batch_size,
        args.preprocess_warmup_iterations,
        args.measurement_iterations,
    )

    cpu_time, cpu_logits = measure_cpu_reference(
        torch_resnet50,
        inputs,
        args.cpu_warmup_iterations,
        args.cpu_iterations,
    )

    row = {
        "Batch": str(batch_size),
        "Total Run Time (sec)": "",
        "Pre-processing Time (sec)": format_seconds(preprocessing_time),
        "Inference Time (sec)": "",
        "CPU Inference Time (sec)": format_seconds(cpu_time),
        "TT Inference Throughput (images/sec)": "",
        "Total Throughput With Pre-processing (images/sec)": "",
        "CPU Throughput (images/sec)": format_throughput(batch_size, cpu_time),
        "Total Images Processed": str(batch_size * args.measurement_iterations),
        "TT Inference Iterations": str(args.measurement_iterations),
        "CPU Iterations": str(args.cpu_iterations),
        "Status": "PREPROCESSING_AND_CPU_OK",
    }

    try:
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
        _, inference_time = run_trace_2cq_pipeline(device, inputs, test_infra, args.measurement_iterations)
        total_run_time = preprocessing_time + inference_time
    except Exception as error:
        logger.exception(f"Failed to run TT pipeline for batch_size={batch_size}")
        row["Status"] = "TT_PIPELINE_ERROR"
        return row

    row.update(
        {
            "Inference Time (sec)": format_seconds(inference_time),
            "TT Inference Throughput (images/sec)": format_throughput(batch_size, inference_time),
            "Total Run Time (sec)": format_seconds(total_run_time),
            "Total Throughput With Pre-processing (images/sec)": format_throughput(batch_size, total_run_time),
            "Status": "OK",
        }
    )
    return row


def make_error_row(batch_size, error):
    return {
        "Batch": str(batch_size),
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


def write_csv(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    image_bytes_list = load_input_image_bytes(args)
    image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME, local_files_only=args.local_files_only)
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
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking ResNet50 trace+2CQ batch_size={batch_size}")
            try:
                rows.append(
                    benchmark_batch(
                        device,
                        batch_size,
                        image_processor,
                        image_bytes_list,
                        torch_resnet50,
                        model_location_generator,
                        args,
                    )
                )
            except Exception as error:
                logger.exception(f"Failed to benchmark batch_size={batch_size}")
                rows.append(make_error_row(batch_size, error))
    finally:
        ttnn.SetDefaultDevice(original_default_device)
        ttnn.close_device(device)

    write_csv(args.output, rows)
    logger.info(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
