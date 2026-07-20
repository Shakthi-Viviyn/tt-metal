# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.vision.classification.mobilenetv2.common import (
    MOBILENETV2_INPUT_CHANNELS,
    MOBILENETV2_PIPELINE_INPUT_CHANNELS,
)
from models.demos.vision.classification.mobilenetv2.reference.mobilenetv2 import (  # Import Conv2dNormActivation
    Conv2dNormActivation,
    InvertedResidual,
    Mobilenetv2,
)
from models.tt_cnn.tt.pipeline import get_memory_config_for_persistent_dram_tensor


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return bias


def create_mobilenetv2_input_tensors(
    batch=1, input_channels=3, input_height=224, input_width=224, pad_channels=None, mesh_mapper=None
):
    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    if pad_channels:
        ttnn_input_tensor = torch.nn.functional.pad(
            ttnn_input_tensor, (0, pad_channels - ttnn_input_tensor.shape[-1]), value=0
        )
    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper
    )
    ttnn_input_tensor = ttnn.reshape(
        ttnn_input_tensor,
        (
            1,
            1,
            ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
            ttnn_input_tensor.shape[3],
        ),
    )
    return torch_input_tensor, ttnn_input_tensor


def pack_mobilenetv2_pipeline_input(input_tensor):
    """Pack pairs of 16-channel rows into DRAM-aligned 32-channel rows."""

    if len(input_tensor.shape) != 4:
        raise ValueError(f"MobileNetV2 pipeline packing expects a 4D tensor (was {input_tensor.shape})")
    if input_tensor.shape[-1] != MOBILENETV2_INPUT_CHANNELS:
        raise ValueError(
            f"MobileNetV2 pipeline packing expects {MOBILENETV2_INPUT_CHANNELS} channels "
            f"(was {input_tensor.shape[-1]})"
        )
    if input_tensor.shape[-2] % 2 != 0:
        raise ValueError(
            f"MobileNetV2 pipeline packing expects an even flattened height (was {input_tensor.shape[-2]})"
        )

    return ttnn.reshape(
        input_tensor,
        (
            input_tensor.shape[0],
            input_tensor.shape[1],
            input_tensor.shape[2] // 2,
            MOBILENETV2_PIPELINE_INPUT_CHANNELS,
        ),
    )


def unpack_mobilenetv2_pipeline_input(input_tensor):
    """Restore a packed persistent-pipeline input before the first convolution."""

    if len(input_tensor.shape) != 4:
        raise ValueError(f"MobileNetV2 pipeline unpacking expects a 4D tensor (was {input_tensor.shape})")
    if input_tensor.shape[-1] != MOBILENETV2_PIPELINE_INPUT_CHANNELS:
        raise ValueError(
            f"MobileNetV2 pipeline unpacking expects {MOBILENETV2_PIPELINE_INPUT_CHANNELS} channels "
            f"(was {input_tensor.shape[-1]})"
        )

    return ttnn.reshape(
        input_tensor,
        (
            input_tensor.shape[0],
            input_tensor.shape[1],
            input_tensor.shape[2] * 2,
            MOBILENETV2_INPUT_CHANNELS,
        ),
    )


def create_mobilenetv2_input_memory_configs(host_input_tensor, device):
    """Create Blackhole-safe persistent DRAM and L1 input configurations.

    The persistent pipeline packs pairs of 16-channel rows into 32-channel rows
    so each stick is naturally aligned to Blackhole's 64-byte DRAM alignment
    without increasing the tensor volume. A 16-channel DRAM stick requires a
    reshard scratch circular buffer that exceeds the hardware page-count limit
    at batch 10. Spread the L1 tensor across the largest rectangular,
    evenly-dividing worker grid so the live input shards do not overlap the
    first convolution's static circular buffers and reshape remains a metadata
    operation.
    """

    if host_input_tensor.shape[-1] != MOBILENETV2_PIPELINE_INPUT_CHANNELS:
        raise ValueError(
            f"The Blackhole MobileNetV2 persistent pipeline requires "
            f"{MOBILENETV2_PIPELINE_INPUT_CHANNELS}-channel input staging "
            f"(was {host_input_tensor.shape[-1]})"
        )

    input_dram_mem_config = get_memory_config_for_persistent_dram_tensor(
        host_input_tensor.shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, device.dram_grid_size()
    )

    device_grid = device.compute_with_storage_grid_size()
    flattened_height = host_input_tensor.shape[-2]
    grid_candidates = [
        (grid_x * grid_y, grid_x, grid_y)
        for grid_x in range(1, device_grid.x + 1)
        for grid_y in range(1, device_grid.y + 1)
        if flattened_height % (grid_x * grid_y) == 0
    ]
    num_input_cores, input_grid_x, input_grid_y = max(grid_candidates)
    input_l1_core_grid = ttnn.CoreGrid(x=input_grid_x, y=input_grid_y)

    input_l1_mem_config = ttnn.create_sharded_memory_config(
        shape=(flattened_height // num_input_cores, host_input_tensor.shape[-1]),
        core_grid=input_l1_core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return input_dram_mem_config, input_l1_mem_config


def fold_batch_norm2d_into_conv2d(conv, bn, mesh_mapper=None):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into Conv2d")
    weight = conv.weight.data
    running_mean = bn.running_mean
    running_var = bn.running_var.data
    eps = bn.eps
    scale = bn.weight.data
    shift = bn.bias.data
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))
    bias = torch.reshape(bias, (1, 1, 1, -1))
    weight = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    bias = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    return weight, bias


def create_mobilenetv2_model_parameters(model, device):
    model_parameters = {}
    conv_bn_counter = 0
    counter = 0
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    for name, module in model.named_modules():
        if isinstance(module, InvertedResidual):
            for idx, submodule in enumerate(module.conv):
                if isinstance(submodule, nn.Conv2d):
                    bn = (
                        module.conv[idx + 1]
                        if idx + 1 < len(module.conv) and isinstance(module.conv[idx + 1], nn.BatchNorm2d)
                        else None
                    )
                    if bn:
                        weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv2d(
                            submodule, bn, mesh_mapper=weights_mesh_mapper
                        )
                        model_parameters[f"conv_{counter}_weight"] = weight_ttnn
                        model_parameters[f"conv_{counter}_bias"] = bias_ttnn
                        counter += 1

        elif isinstance(module, Conv2dNormActivation):
            if len(module) == 3 and isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.BatchNorm2d):
                conv = module[0]
                bn = module[1]
                weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv2d(conv, bn, mesh_mapper=weights_mesh_mapper)
                model_parameters[f"fused_conv_{conv_bn_counter}_weight"] = weight_ttnn
                model_parameters[f"fused_conv_{conv_bn_counter}_bias"] = bias_ttnn
                conv_bn_counter += 1

        elif isinstance(module, nn.Linear):
            model_parameters["classifier_1_weight"] = preprocess_linear_weight(
                module.weight.data, dtype=ttnn.float32, mesh_mapper=weights_mesh_mapper
            )
            model_parameters["classifier_1_bias"] = preprocess_linear_bias(
                module.bias.data, dtype=ttnn.float32, mesh_mapper=weights_mesh_mapper
            )
            model_parameters["classifier_1_weight"] = ttnn.to_device(model_parameters["classifier_1_weight"], device)
            model_parameters["classifier_1_bias"] = ttnn.to_device(model_parameters["classifier_1_bias"], device)

    return model_parameters
