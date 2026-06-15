# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.vision.classification.resnet50.ttnn_resnet.tests.common.resnet50_test_infra import (
    create_test_infra,
    load_resnet50_model,
)
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


DEFAULT_BATCH_SIZES = (16, 32)
DEFAULT_IMAGE_DIR = Path("models/demos/vision/classification/resnet50/ttnn_resnet/demo/images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
RESIZE_SHORTER_SIDE = 256
CROP_SIZE = 224
IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).reshape(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).reshape(1, 3, 1, 1)
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
            "Measure Blackhole ResNet50 trace+1CQ steady-state inference plus host image preprocessing, "
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
        "--turbojpeg-lib-path",
        type=str,
        default=None,
        help="Optional explicit path to libturbojpeg.so/.dylib if PyTurboJPEG cannot locate it automatically.",
    )
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
    return Path(__file__).resolve().parent / "reports" / f"resnet50_blackhole_trace_1cq_turbojpeg_preprocessing_e2e_{timestamp}.csv"


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


def create_jpeg_decoder(turbojpeg_lib_path=None):
    try:
        from turbojpeg import TJPF_RGB, TurboJPEG
    except ImportError as error:
        raise ImportError(
            "TurboJPEG Python bindings are required for this benchmark. Install PyTurboJPEG and make sure "
            "libjpeg-turbo is available on the host."
        ) from error

    jpeg = TurboJPEG(turbojpeg_lib_path)

    def decode_rgb(image_bytes):
        return jpeg.decode(image_bytes, pixel_format=TJPF_RGB)

    return decode_rgb


def resize_shorter_side_batch(batch_nchw):
    _, _, height, width = batch_nchw.shape
    if height <= width:
        resized_height = RESIZE_SHORTER_SIDE
        resized_width = round(width * RESIZE_SHORTER_SIDE / height)
    else:
        resized_height = round(height * RESIZE_SHORTER_SIDE / width)
        resized_width = RESIZE_SHORTER_SIDE

    return F.interpolate(
        batch_nchw,
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    )


def center_crop_batch(batch_nchw):
    _, _, height, width = batch_nchw.shape
    top = (height - CROP_SIZE) // 2
    left = (width - CROP_SIZE) // 2
    return batch_nchw[:, :, top : top + CROP_SIZE, left : left + CROP_SIZE]


def decode_image_batch(decode_rgb, image_byte_batch):
    return [decode_rgb(image_bytes) for image_bytes in image_byte_batch]


def resize_crop_decoded_images(decoded_images):
    groups = defaultdict(list)
    for index, image_hwc in enumerate(decoded_images):
        groups[image_hwc.shape[:2]].append((index, image_hwc))

    resized_cropped_images = [None] * len(decoded_images)
    for items in groups.values():
        batch_hwc = np.stack([image_hwc for _, image_hwc in items], axis=0)
        batch_nchw = torch.from_numpy(batch_hwc).permute(0, 3, 1, 2).to(torch.float32)
        batch_nchw = center_crop_batch(resize_shorter_side_batch(batch_nchw))
        for item_index, (original_index, _) in enumerate(items):
            resized_cropped_images[original_index] = batch_nchw[item_index]

    return torch.stack(resized_cropped_images, dim=0)


def preprocess_image_batch(decode_rgb, image_byte_batch):
    batch = resize_crop_decoded_images(decode_image_batch(decode_rgb, image_byte_batch))
    batch = batch.mul_(1.0 / 255.0)
    batch = batch.sub_(IMAGE_MEAN).div_(IMAGE_STD)
    return batch.bfloat16()


def measure_preprocessing(decode_rgb, image_bytes_list, batch_size, warmup_iterations, measurement_iterations):
    for iteration_index in range(warmup_iterations):
        preprocess_image_batch(
            decode_rgb,
            make_image_byte_batch(image_bytes_list, batch_size, iteration_index),
        )

    timings = []
    last_inputs = None
    for iteration_index in range(warmup_iterations, warmup_iterations + measurement_iterations):
        image_byte_batch = make_image_byte_batch(image_bytes_list, batch_size, iteration_index)
        start = time.perf_counter()
        last_inputs = preprocess_image_batch(decode_rgb, image_byte_batch)
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


def run_trace_1cq_pipeline(device, inputs, test_infra, measurement_iterations):
    tt_inputs_host, sharded_mem_config_dram, input_mem_config = test_infra.setup_dram_sharded_input(device, inputs)

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
    decode_rgb,
    image_bytes_list,
    torch_resnet50,
    model_location_generator,
    args,
):
    preprocessing_time, inputs = measure_preprocessing(
        decode_rgb,
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
        _, inference_time = run_trace_1cq_pipeline(device, inputs, test_infra, args.measurement_iterations)
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
    decode_rgb = create_jpeg_decoder(args.turbojpeg_lib_path)
    model_location_generator = make_model_location_generator(args.torchvision_weights_path)

    torch_resnet50 = load_resnet50_model(model_location_generator)
    torch_resnet50.eval()
    torch_resnet50.to(torch.bfloat16)

    original_default_device = ttnn.GetDefaultDevice()
    device = ttnn.CreateDevice(
        device_id=args.device_id,
        l1_small_size=args.l1_small_size,
        trace_region_size=args.trace_region_size,
        num_command_queues=1,
    )
    rows = []

    try:
        ttnn.SetDefaultDevice(device)
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking ResNet50 trace+1CQ batch_size={batch_size}")
            try:
                rows.append(
                    benchmark_batch(
                        device,
                        batch_size,
                        decode_rgb,
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
