# MobileNetV2 C++ Preprocessing Benchmark

This benchmark mirrors the ResNet50 Blackhole C++ preprocessing benchmark while using MobileNetV2's fixed batch size of 10.

The C++ extension performs:

- TurboJPEG decode through `libturbojpeg`
- resize the shorter side to 224
- center crop to 224x224
- ImageNet normalization
- float32 NCHW packing for PyTorch reference inference
- BF16 channel padding and `[1, 1, 250880, 32]` packing for the Blackhole TT-NN pipeline
- true host-side worker threading with `std::thread`

## Build

Activate the repository Python environment and build from this directory:

```bash
source python_env/bin/activate
cd models/demos/vision/classification/mobilenetv2/blackhole/demo/cpp_preprocessing
python3 setup.py build_ext --inplace
```

If `libturbojpeg` is installed in a non-standard location:

```bash
TURBOJPEG_INCLUDE_DIR=/path/to/include \
TURBOJPEG_LIB_DIR=/path/to/lib \
python3 setup.py build_ext --inplace
```

## Run

From the `tt-metal` repository root:

```bash
source python_env/bin/activate
python3 models/demos/vision/classification/mobilenetv2/blackhole/demo/cpp_preprocessing/benchmark_e2e_profile_cpp_preprocess.py \
  --device-id 1 \
  --preprocess-worker-counts 1,2,4,8,10
```

By default, the benchmark cycles through the ImageNet JPEGs already used by the ResNet50 demo. Use `--image` or `--image-dir` to select other JPEG inputs.

The CSV schema follows the ResNet50 benchmark. Preprocessing is timed inside C++, while PyTorch reference inference, TT-NN trace compilation, TT enqueue, PCC validation, and CSV writing remain in Python. The log also reports the preprocessing-versus-inference percentage split for each worker count.

Unless `--output` is specified, reports are written to:

```text
models/demos/vision/classification/mobilenetv2/blackhole/demo/reports/
mobilenetv2_blackhole_trace_1cq_cpp_preprocessing_e2e_<timestamp>.csv
```

The column names, column order, per-worker rows, blank separator row, and best-worker summary footer match the ResNet50 C++ preprocessing report. MobileNetV2 reports contain one batch-size row per worker because the benchmark currently supports batch size 10.

The benchmark uses a default PCC threshold of 0.90 as a real-image execution guard. Override it with `--pcc-threshold`; use the model PCC test for model qualification.
