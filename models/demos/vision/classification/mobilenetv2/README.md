# Mobilenetv2

## Platforms:
    Blackhole

### Introduction
The MobileNetV2 model is a convolutional neural network (CNN) architecture designed for efficient mobile and embedded vision applications. It was introduced in the paper ["MobileNetV2: Inverted Residuals and Linear Bottlenecks"](https://arxiv.org/abs/1801.04381). </br>
The MobileNetV2 model has been pre-trained on the ImageNet dataset and can be used for various tasks such as image classification, object detection, and semantic segmentation. It has achieved state-of-the-art performance on several benchmarks 1 for mobile and embedded vision applications.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- login to huggingface with: `huggingface-cli login` or by setting the token with the command `export HF_TOKEN=<token>`
   - To obtain a huggingface token visit: https://huggingface.co/docs/hub/security-tokens

## How to Run
- Use the following command to run the model:
```
pytest models/demos/vision/classification/mobilenetv2/tests/pcc/test_mobilenetv2.py::test_mobilenetv2
```

### Performant Model with Trace+2CQ
#### Single Device (BS=10):
- Validated on Blackhole at batch size 10. A local 32-iteration run measured 2.73 ms per batch (3664.71 FPS); results vary by system and build.

```
pytest models/demos/vision/classification/mobilenetv2/tests/perf/test_perf_e2e_mobilenetv2.py::test_mobilenetv2_e2e
```

#### Multi Device (DP=2):
- Blackhole multi-device correctness and performance are not yet characterized.

```
pytest models/demos/vision/classification/mobilenetv2/tests/perf/test_perf_e2e_mobilenetv2.py::test_mobilenetv2_e2e_dp
```

### Performant Demo with Trace+2CQ
- Make sure your HuggingFace token is set ([See Prerequisites](#prerequisites) for instructions)
#### Single Device (BS=10):
```
pytest models/demos/vision/classification/mobilenetv2/demo/demo.py::test_mobilenetv2_imagenet_demo
```

#### Multi Device (DP=2):
```
pytest models/demos/vision/classification/mobilenetv2/demo/demo.py::test_mobilenetv2_imagenet_demo_dp
```

### End-to-end C++ preprocessing benchmark

The Blackhole benchmark measures TurboJPEG decode, resize, crop, normalization, TT input packing, and trace+1CQ inference at the supported batch size of 10. See the [C++ preprocessing benchmark README](blackhole/demo/cpp_preprocessing/README.md) for build and run instructions.

## Testing

### Performant Data evaluation with Trace+2CQ
#### Single Device (BS=10):
```
pytest models/demos/vision/classification/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval
```

#### Multi Device (DP=2):
```
pytest models/demos/vision/classification/classification_eval/classification_eval.py::test_mobilenetv2_image_classification_eval_dp
```
Note: The model is evaluated with 512 samples.

## Details
- The post-processing is performed using PyTorch.
- The entry point to mobilenetv2 model is MobileNetV2 in `models/demos/vision/classification/mobilenetv2/tt/ttnn_mobilenetv2.py`.
- Supported Input Resolution - (224,224) (Height,Width)
- Batch Size :10
- Dataset used for evaluation - **imagenet-1k**
- The current implementation is Blackhole-only. Run the PCC test before the traced performance or demo paths when bringing it up on a new Blackhole SKU.
