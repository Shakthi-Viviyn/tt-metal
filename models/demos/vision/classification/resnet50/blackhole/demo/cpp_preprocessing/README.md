# ResNet50 C++ Preprocessing Benchmark

The C++ extension performs:

- TurboJPEG decode through `libturbojpeg`
- resize shorter side to 256
- center crop to 224x224
- ImageNet normalization
- BF16 NCHW tensor packing
- true host-side worker threading with `std::thread`

## Build

From this directory:

```bash
python3 setup.py build_ext --inplace
```

If `libturbojpeg` is installed in a non-standard location, point the build at it:

```bash
TURBOJPEG_INCLUDE_DIR=/path/to/include TURBOJPEG_LIB_DIR=/path/to/lib python3 setup.py build_ext --inplace
```

## Run

From the `tt-metal` repository root:

```bash
python3 models/demos/vision/classification/resnet50/blackhole/demo/cpp_preprocessing/benchmark_e2e_profile_cpp_preprocess.py \
  --batch-sizes 16,32 \
  --preprocess-worker-counts 1,4,8,16,32
```

The CSV schema matches the existing Python TurboJPEG threaded benchmark. The preprocessing time is measured inside C++, while CPU reference inference, TT-NN trace compilation, TT enqueue, validation, and CSV writing remain in Python.
