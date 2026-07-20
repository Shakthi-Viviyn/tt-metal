# ViT C++ Preprocessing Benchmark

The C++ extension performs host-side preprocessing for the Blackhole ViT benchmark:

- TurboJPEG decode through `libturbojpeg`
- resize to `224x224`
- ViT image normalization matching `google/vit-base-patch16-224`
- CPU-reference NCHW float32 tensor output
- TT-ready BF16 packing to `[batch, 224, 14, 64]`
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

The benchmark automatically downloads its JPEG images when the sibling `images/` directory is missing or contains no
JPEGs. To populate it separately before running inference, use:

```bash
python3 models/demos/vision/classification/vit/blackhole/demo/cpp_preprocessing/download_images.py
```

This downloads native-resolution JPEGs from `microsoft/cats_vs_dogs` (`train` split), skips images whose shorter side
is below 224 pixels, and writes the original bytes into the sibling `images/` directory. It does not resize, crop,
normalize, re-encode, or otherwise preprocess the images. Legacy 64x64 Tiny ImageNet downloads are removed after the
replacement set has downloaded successfully.

From the `tt-metal` repository root:

```bash
python3 models/demos/vision/classification/vit/blackhole/demo/cpp_preprocessing/benchmark_e2e_profile_cpp_preprocess.py \
  --preprocess-worker-counts 1,4,8,10
```

The benchmark reads JPEGs from `models/demos/vision/classification/vit/blackhole/demo/cpp_preprocessing/images`.
The first version intentionally supports batch size `10`, matching the currently tested Blackhole ViT sharding path.
Inference is measured with a single command queue trace path; validation runs separately after the timed iterations.
Worker counts above the batch size are capped at 10 and duplicate effective worker counts are measured only once.
The timed inference path includes host input transfer, traced model execution, and host output transfer.
Validation logs top-1/top-5 classification agreement for visibility and gates the benchmark status on exact top-1 agreement or logits PCC.
The log includes aggregate, mean per-image, and minimum per-image PCC so image-dependent numerical behavior is visible.
The CSV uses the same column names, ordering, row structure, and best-worker summary as the MobileNetV2 C++
preprocessing benchmark.
