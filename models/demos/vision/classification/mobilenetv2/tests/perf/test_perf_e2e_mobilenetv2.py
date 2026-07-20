# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.vision.classification.mobilenetv2.common import (
    MOBILENETV2_BATCH_SIZE,
    MOBILENETV2_INPUT_CHANNELS,
    MOBILENETV2_L1_SMALL_SIZE,
    MOBILENETV2_TRACE_REGION_SIZE,
    load_torch_model,
)
from models.demos.vision.classification.mobilenetv2.reference.mobilenetv2 import Mobilenetv2
from models.demos.vision.classification.mobilenetv2.tt import ttnn_mobilenetv2
from models.demos.vision.classification.mobilenetv2.tt.model_preprocessing import (
    create_mobilenetv2_input_memory_configs,
    create_mobilenetv2_input_tensors,
    create_mobilenetv2_model_parameters,
    get_mesh_mappers,
    pack_mobilenetv2_pipeline_input,
    unpack_mobilenetv2_pipeline_input,
)
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_mobilenetv2_e2e(
    device,
    batch_size_per_device,
    model_location_generator=None,
):
    num_devices = device.get_num_devices()
    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)
    batch_size = batch_size_per_device * num_devices
    torch_input_tensor, host_input_tensor = create_mobilenetv2_input_tensors(
        batch=batch_size,
        input_height=224,
        input_width=224,
        pad_channels=MOBILENETV2_INPUT_CHANNELS,
        mesh_mapper=inputs_mesh_mapper,
    )
    host_input_tensor = pack_mobilenetv2_pipeline_input(host_input_tensor)
    torch_model = load_torch_model(Mobilenetv2(), model_location_generator=model_location_generator)
    torch_output_tensor = torch_model(torch_input_tensor)

    model_parameters = create_mobilenetv2_model_parameters(torch_model, device=device)
    ttnn_model = ttnn_mobilenetv2.TtMobileNetV2(model_parameters, device, batchsize=batch_size_per_device)

    def model_wrapper(packed_input_tensor):
        return ttnn_model(unpack_mobilenetv2_pipeline_input(packed_input_tensor))

    input_dram_mem_config, input_l1_mem_config = create_mobilenetv2_input_memory_configs(host_input_tensor, device)
    logger.info(
        f"Auto-selected persistent DRAM tensor memory config: shape={host_input_tensor.shape}, shard_shape={input_dram_mem_config.shard_spec.shape}, grid={input_dram_mem_config.shard_spec.grid}"
    )

    config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipe = create_pipeline_from_config(
        config,
        model_wrapper,
        device,
        dram_input_memory_config=input_dram_mem_config,
        l1_input_memory_config=input_l1_mem_config,
    )

    iterations = 32
    host_inputs = [host_input_tensor] * iterations

    pipe.compile(host_input_tensor)
    pipe.preallocate_output_tensors_on_host(len(host_inputs))

    start = time.time()
    outputs = pipe.enqueue(host_inputs).pop_all()
    end = time.time()

    pipe.cleanup()

    inference_time = (end - start) / iterations
    logger.info(f"Average model time={1000.0 * inference_time : .2f} ms")
    logger.info(f"Average model performance={iterations * batch_size / (end-start) : .2f} fps")
    assert_with_pcc(torch_output_tensor, ttnn.to_torch(outputs[-1], mesh_composer=output_mesh_composer), 0.94)


@run_for_blackhole()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": MOBILENETV2_L1_SMALL_SIZE,
            "trace_region_size": MOBILENETV2_TRACE_REGION_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    ((MOBILENETV2_BATCH_SIZE),),
)
def test_mobilenetv2_e2e(batch_size, device, model_location_generator):
    run_mobilenetv2_e2e(device, batch_size, model_location_generator)


@run_for_blackhole()
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": MOBILENETV2_L1_SMALL_SIZE,
            "trace_region_size": MOBILENETV2_TRACE_REGION_SIZE,
            "num_command_queues": 2,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_batch_size",
    ((MOBILENETV2_BATCH_SIZE),),
)
def test_mobilenetv2_e2e_dp(device_batch_size, mesh_device, model_location_generator):
    run_mobilenetv2_e2e(mesh_device, device_batch_size, model_location_generator)
