#include "ttnn-precompiled.hpp"
::std::vector<::ttnn::Tensor>
main_const_eval_0(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_1(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_2(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_3(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_4(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_5(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_6(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_7(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_8(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_9(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_10(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_11(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_12(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_13(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_14(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_15(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_16(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_17(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_18(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_19(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_20(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_21(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_22(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_23(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_24(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_25(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_26(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_27(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_28(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_29(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_30(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_31(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_32(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_33(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_34(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_35(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_36(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_37(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_38(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_39(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_40(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_41(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_42(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_43(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_44(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_45(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_46(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_47(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_48(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_49(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_50(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_51(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_52(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_53(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_54(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_55(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_56(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_57(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_58(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_59(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_60(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_61(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_62(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_63(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_64(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_65(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_66(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_67(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_68(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_69(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_70(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_71(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_72(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_73(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_74(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_75(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_76(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_77(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_78(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_79(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_80(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_81(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_82(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_83(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_84(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_85(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_86(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_87(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_88(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_89(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_90(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_91(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_92(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_93(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_94(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_95(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_96(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_97(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_98(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_99(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_100(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_101(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_102(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_103(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_104(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_105(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_106(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_107(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_108(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_109(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_110(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_111(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_112(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_113(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_114(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_115(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_116(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_117(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_118(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_119(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_120(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_121(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_122(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_123(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_124(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_125(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_126(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_127(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_128(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_129(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_130(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_131(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_132(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_133(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_134(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_135(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_136(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_137(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_138(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_139(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_140(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_141(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_142(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_143(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_144(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_145(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_146(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_147(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_148(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_149(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_150(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_151(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_152(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_153(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_154(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_155(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_156(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_157(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_158(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_159(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_160(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_161(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_162(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_163(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_164(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_165(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_166(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_167(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_168(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_169(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_170(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_171(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_172(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_173(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_174(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_175(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_176(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_177(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_178(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_179(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_180(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_181(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_182(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_183(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_184(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_185(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_186(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_187(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_188(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_189(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_190(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_191(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_192(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_193(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_194(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_195(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_196(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 2048, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_197(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_198(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_199(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_200(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_201(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_202(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_203(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 512, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_204(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_205(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_206(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_207(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_208(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 1024, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_209(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 64, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_210(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 128, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
::std::vector<::ttnn::Tensor>
main_const_eval_211(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ttnn::distributed::MeshDevice *v3 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v4 = ttnn::to_device(
      v2, v3,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::to_layout(
      v4, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v4, false);
  ::ttnn::Tensor v6 = ttnn::reshape(
      v5, ::std::vector<int32_t>{1, 256, 1, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v5, false);
  ::std::vector<::ttnn::Tensor> v7 = util_create_vec(v6);
  return v7;
}
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_0;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_1;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_2;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_3;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_4;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_5;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_6;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_7;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_8;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_9;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_10;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_11;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_12;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_13;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_14;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_15;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_16;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_17;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_18;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_19;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_20;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_21;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_22;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_23;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_24;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_25;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_26;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_27;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_28;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_29;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_30;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_31;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_32;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_33;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_34;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_35;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_36;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_37;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_38;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_39;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_40;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_41;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_42;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_43;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_44;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_45;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_46;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_47;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_48;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_49;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_50;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_51;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_52;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_53;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_54;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_55;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_56;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_57;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_58;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_59;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_60;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_61;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_62;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_63;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_64;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_65;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_66;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_67;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_68;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_69;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_70;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_71;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_72;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_73;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_74;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_75;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_76;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_77;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_78;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_79;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_80;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_81;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_82;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_83;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_84;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_85;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_86;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_87;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_88;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_89;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_90;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_91;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_92;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_93;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_94;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_95;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_96;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_97;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_98;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_99;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_100;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_101;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_102;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_103;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_104;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_105;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_106;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_107;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_108;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_109;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_110;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_111;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_112;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_113;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_114;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_115;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_116;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_117;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_118;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_119;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_120;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_121;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_122;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_123;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_124;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_125;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_126;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_127;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_128;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_129;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_130;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_131;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_132;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_133;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_134;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_135;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_136;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_137;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_138;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_139;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_140;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_141;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_142;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_143;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_144;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_145;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_146;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_147;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_148;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_149;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_150;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_151;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_152;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_153;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_154;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_155;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_156;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_157;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_158;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_159;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_160;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_161;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_162;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_163;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_164;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_165;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_166;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_167;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_168;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_169;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_170;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_171;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_172;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_173;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_174;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_175;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_176;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_177;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_178;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_179;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_180;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_181;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_182;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_183;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_184;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_185;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_186;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_187;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_188;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_189;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_190;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_191;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_192;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_193;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_194;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_195;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_196;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_197;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_198;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_199;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_200;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_201;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_202;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_203;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_204;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_205;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_206;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_207;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_208;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_209;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_210;
static ::std::vector<::ttnn::Tensor> g_cached_result_main_const_eval_211;
::std::vector<::ttnn::Tensor> _main(::std::vector<::ttnn::Tensor> v1) {
  ::ttnn::Tensor v2 = v1[0];
  ::ttnn::Tensor v3 = v1[1];
  ::ttnn::Tensor v4 = v1[2];
  ::ttnn::Tensor v5 = v1[3];
  ::ttnn::Tensor v6 = v1[4];
  ::ttnn::Tensor v7 = v1[5];
  ::ttnn::Tensor v8 = v1[6];
  ::ttnn::Tensor v9 = v1[7];
  ::ttnn::Tensor v10 = v1[8];
  ::ttnn::Tensor v11 = v1[9];
  ::ttnn::Tensor v12 = v1[10];
  ::ttnn::Tensor v13 = v1[11];
  ::ttnn::Tensor v14 = v1[12];
  ::ttnn::Tensor v15 = v1[13];
  ::ttnn::Tensor v16 = v1[14];
  ::ttnn::Tensor v17 = v1[15];
  ::ttnn::Tensor v18 = v1[16];
  ::ttnn::Tensor v19 = v1[17];
  ::ttnn::Tensor v20 = v1[18];
  ::ttnn::Tensor v21 = v1[19];
  ::ttnn::Tensor v22 = v1[20];
  ::ttnn::Tensor v23 = v1[21];
  ::ttnn::Tensor v24 = v1[22];
  ::ttnn::Tensor v25 = v1[23];
  ::ttnn::Tensor v26 = v1[24];
  ::ttnn::Tensor v27 = v1[25];
  ::ttnn::Tensor v28 = v1[26];
  ::ttnn::Tensor v29 = v1[27];
  ::ttnn::Tensor v30 = v1[28];
  ::ttnn::Tensor v31 = v1[29];
  ::ttnn::Tensor v32 = v1[30];
  ::ttnn::Tensor v33 = v1[31];
  ::ttnn::Tensor v34 = v1[32];
  ::ttnn::Tensor v35 = v1[33];
  ::ttnn::Tensor v36 = v1[34];
  ::ttnn::Tensor v37 = v1[35];
  ::ttnn::Tensor v38 = v1[36];
  ::ttnn::Tensor v39 = v1[37];
  ::ttnn::Tensor v40 = v1[38];
  ::ttnn::Tensor v41 = v1[39];
  ::ttnn::Tensor v42 = v1[40];
  ::ttnn::Tensor v43 = v1[41];
  ::ttnn::Tensor v44 = v1[42];
  ::ttnn::Tensor v45 = v1[43];
  ::ttnn::Tensor v46 = v1[44];
  ::ttnn::Tensor v47 = v1[45];
  ::ttnn::Tensor v48 = v1[46];
  ::ttnn::Tensor v49 = v1[47];
  ::ttnn::Tensor v50 = v1[48];
  ::ttnn::Tensor v51 = v1[49];
  ::ttnn::Tensor v52 = v1[50];
  ::ttnn::Tensor v53 = v1[51];
  ::ttnn::Tensor v54 = v1[52];
  ::ttnn::Tensor v55 = v1[53];
  ::ttnn::Tensor v56 = v1[54];
  ::ttnn::Tensor v57 = v1[55];
  ::ttnn::Tensor v58 = v1[56];
  ::ttnn::Tensor v59 = v1[57];
  ::ttnn::Tensor v60 = v1[58];
  ::ttnn::Tensor v61 = v1[59];
  ::ttnn::Tensor v62 = v1[60];
  ::ttnn::Tensor v63 = v1[61];
  ::ttnn::Tensor v64 = v1[62];
  ::ttnn::Tensor v65 = v1[63];
  ::ttnn::Tensor v66 = v1[64];
  ::ttnn::Tensor v67 = v1[65];
  ::ttnn::Tensor v68 = v1[66];
  ::ttnn::Tensor v69 = v1[67];
  ::ttnn::Tensor v70 = v1[68];
  ::ttnn::Tensor v71 = v1[69];
  ::ttnn::Tensor v72 = v1[70];
  ::ttnn::Tensor v73 = v1[71];
  ::ttnn::Tensor v74 = v1[72];
  ::ttnn::Tensor v75 = v1[73];
  ::ttnn::Tensor v76 = v1[74];
  ::ttnn::Tensor v77 = v1[75];
  ::ttnn::Tensor v78 = v1[76];
  ::ttnn::Tensor v79 = v1[77];
  ::ttnn::Tensor v80 = v1[78];
  ::ttnn::Tensor v81 = v1[79];
  ::ttnn::Tensor v82 = v1[80];
  ::ttnn::Tensor v83 = v1[81];
  ::ttnn::Tensor v84 = v1[82];
  ::ttnn::Tensor v85 = v1[83];
  ::ttnn::Tensor v86 = v1[84];
  ::ttnn::Tensor v87 = v1[85];
  ::ttnn::Tensor v88 = v1[86];
  ::ttnn::Tensor v89 = v1[87];
  ::ttnn::Tensor v90 = v1[88];
  ::ttnn::Tensor v91 = v1[89];
  ::ttnn::Tensor v92 = v1[90];
  ::ttnn::Tensor v93 = v1[91];
  ::ttnn::Tensor v94 = v1[92];
  ::ttnn::Tensor v95 = v1[93];
  ::ttnn::Tensor v96 = v1[94];
  ::ttnn::Tensor v97 = v1[95];
  ::ttnn::Tensor v98 = v1[96];
  ::ttnn::Tensor v99 = v1[97];
  ::ttnn::Tensor v100 = v1[98];
  ::ttnn::Tensor v101 = v1[99];
  ::ttnn::Tensor v102 = v1[100];
  ::ttnn::Tensor v103 = v1[101];
  ::ttnn::Tensor v104 = v1[102];
  ::ttnn::Tensor v105 = v1[103];
  ::ttnn::Tensor v106 = v1[104];
  ::ttnn::Tensor v107 = v1[105];
  ::ttnn::Tensor v108 = v1[106];
  ::ttnn::Tensor v109 = v1[107];
  ::ttnn::Tensor v110 = v1[108];
  ::ttnn::Tensor v111 = v1[109];
  ::ttnn::Tensor v112 = v1[110];
  ::ttnn::Tensor v113 = v1[111];
  ::ttnn::Tensor v114 = v1[112];
  ::ttnn::Tensor v115 = v1[113];
  ::ttnn::Tensor v116 = v1[114];
  ::ttnn::Tensor v117 = v1[115];
  ::ttnn::Tensor v118 = v1[116];
  ::ttnn::Tensor v119 = v1[117];
  ::ttnn::Tensor v120 = v1[118];
  ::ttnn::Tensor v121 = v1[119];
  ::ttnn::Tensor v122 = v1[120];
  ::ttnn::Tensor v123 = v1[121];
  ::ttnn::Tensor v124 = v1[122];
  ::ttnn::Tensor v125 = v1[123];
  ::ttnn::Tensor v126 = v1[124];
  ::ttnn::Tensor v127 = v1[125];
  ::ttnn::Tensor v128 = v1[126];
  ::ttnn::Tensor v129 = v1[127];
  ::ttnn::Tensor v130 = v1[128];
  ::ttnn::Tensor v131 = v1[129];
  ::ttnn::Tensor v132 = v1[130];
  ::ttnn::Tensor v133 = v1[131];
  ::ttnn::Tensor v134 = v1[132];
  ::ttnn::Tensor v135 = v1[133];
  ::ttnn::Tensor v136 = v1[134];
  ::ttnn::Tensor v137 = v1[135];
  ::ttnn::Tensor v138 = v1[136];
  ::ttnn::Tensor v139 = v1[137];
  ::ttnn::Tensor v140 = v1[138];
  ::ttnn::Tensor v141 = v1[139];
  ::ttnn::Tensor v142 = v1[140];
  ::ttnn::Tensor v143 = v1[141];
  ::ttnn::Tensor v144 = v1[142];
  ::ttnn::Tensor v145 = v1[143];
  ::ttnn::Tensor v146 = v1[144];
  ::ttnn::Tensor v147 = v1[145];
  ::ttnn::Tensor v148 = v1[146];
  ::ttnn::Tensor v149 = v1[147];
  ::ttnn::Tensor v150 = v1[148];
  ::ttnn::Tensor v151 = v1[149];
  ::ttnn::Tensor v152 = v1[150];
  ::ttnn::Tensor v153 = v1[151];
  ::ttnn::Tensor v154 = v1[152];
  ::ttnn::Tensor v155 = v1[153];
  ::ttnn::Tensor v156 = v1[154];
  ::ttnn::Tensor v157 = v1[155];
  ::ttnn::Tensor v158 = v1[156];
  ::ttnn::Tensor v159 = v1[157];
  ::ttnn::Tensor v160 = v1[158];
  ::ttnn::Tensor v161 = v1[159];
  ::ttnn::Tensor v162 = v1[160];
  ::ttnn::Tensor v163 = v1[161];
  ::ttnn::Tensor v164 = v1[162];
  ::ttnn::Tensor v165 = v1[163];
  ::ttnn::Tensor v166 = v1[164];
  ::ttnn::Tensor v167 = v1[165];
  ::ttnn::Tensor v168 = v1[166];
  ::ttnn::Tensor v169 = v1[167];
  ::ttnn::Tensor v170 = v1[168];
  ::ttnn::Tensor v171 = v1[169];
  ::ttnn::Tensor v172 = v1[170];
  ::ttnn::Tensor v173 = v1[171];
  ::ttnn::Tensor v174 = v1[172];
  ::ttnn::Tensor v175 = v1[173];
  ::ttnn::Tensor v176 = v1[174];
  ::ttnn::Tensor v177 = v1[175];
  ::ttnn::Tensor v178 = v1[176];
  ::ttnn::Tensor v179 = v1[177];
  ::ttnn::Tensor v180 = v1[178];
  ::ttnn::Tensor v181 = v1[179];
  ::ttnn::Tensor v182 = v1[180];
  ::ttnn::Tensor v183 = v1[181];
  ::ttnn::Tensor v184 = v1[182];
  ::ttnn::Tensor v185 = v1[183];
  ::ttnn::Tensor v186 = v1[184];
  ::ttnn::Tensor v187 = v1[185];
  ::ttnn::Tensor v188 = v1[186];
  ::ttnn::Tensor v189 = v1[187];
  ::ttnn::Tensor v190 = v1[188];
  ::ttnn::Tensor v191 = v1[189];
  ::ttnn::Tensor v192 = v1[190];
  ::ttnn::Tensor v193 = v1[191];
  ::ttnn::Tensor v194 = v1[192];
  ::ttnn::Tensor v195 = v1[193];
  ::ttnn::Tensor v196 = v1[194];
  ::ttnn::Tensor v197 = v1[195];
  ::ttnn::Tensor v198 = v1[196];
  ::ttnn::Tensor v199 = v1[197];
  ::ttnn::Tensor v200 = v1[198];
  ::ttnn::Tensor v201 = v1[199];
  ::ttnn::Tensor v202 = v1[200];
  ::ttnn::Tensor v203 = v1[201];
  ::ttnn::Tensor v204 = v1[202];
  ::ttnn::Tensor v205 = v1[203];
  ::ttnn::Tensor v206 = v1[204];
  ::ttnn::Tensor v207 = v1[205];
  ::ttnn::Tensor v208 = v1[206];
  ::ttnn::Tensor v209 = v1[207];
  ::ttnn::Tensor v210 = v1[208];
  ::ttnn::Tensor v211 = v1[209];
  ::ttnn::Tensor v212 = v1[210];
  ::ttnn::Tensor v213 = v1[211];
  ::ttnn::Tensor v214 = v1[212];
  ::ttnn::Tensor v215 = v1[213];
  ::ttnn::Tensor v216 = v1[214];
  ::ttnn::Tensor v217 = v1[215];
  ::ttnn::Tensor v218 = v1[216];
  ::ttnn::Tensor v219 = v1[217];
  ::ttnn::Tensor v220 = v1[218];
  ::ttnn::Tensor v221 = v1[219];
  ::ttnn::Tensor v222 = v1[220];
  ::ttnn::Tensor v223 = v1[221];
  ::ttnn::Tensor v224 = v1[222];
  ::ttnn::Tensor v225 = v1[223];
  ::ttnn::Tensor v226 = v1[224];
  ::ttnn::Tensor v227 = v1[225];
  ::ttnn::Tensor v228 = v1[226];
  ::ttnn::Tensor v229 = v1[227];
  ::ttnn::Tensor v230 = v1[228];
  ::ttnn::Tensor v231 = v1[229];
  ::ttnn::Tensor v232 = v1[230];
  ::ttnn::Tensor v233 = v1[231];
  ::ttnn::Tensor v234 = v1[232];
  ::ttnn::Tensor v235 = v1[233];
  ::ttnn::Tensor v236 = v1[234];
  ::ttnn::Tensor v237 = v1[235];
  ::ttnn::Tensor v238 = v1[236];
  ::ttnn::Tensor v239 = v1[237];
  ::ttnn::Tensor v240 = v1[238];
  ::ttnn::Tensor v241 = v1[239];
  ::ttnn::Tensor v242 = v1[240];
  ::ttnn::Tensor v243 = v1[241];
  ::ttnn::Tensor v244 = v1[242];
  ::ttnn::Tensor v245 = v1[243];
  ::ttnn::Tensor v246 = v1[244];
  ::ttnn::Tensor v247 = v1[245];
  ::ttnn::Tensor v248 = v1[246];
  ::ttnn::Tensor v249 = v1[247];
  ::ttnn::Tensor v250 = v1[248];
  ::ttnn::Tensor v251 = v1[249];
  ::ttnn::Tensor v252 = v1[250];
  ::ttnn::Tensor v253 = v1[251];
  ::ttnn::Tensor v254 = v1[252];
  ::ttnn::Tensor v255 = v1[253];
  ::ttnn::Tensor v256 = v1[254];
  ::ttnn::Tensor v257 = v1[255];
  ::ttnn::Tensor v258 = v1[256];
  ::ttnn::Tensor v259 = v1[257];
  ::ttnn::Tensor v260 = v1[258];
  ::ttnn::Tensor v261 = v1[259];
  ::ttnn::Tensor v262 = v1[260];
  ::ttnn::Tensor v263 = v1[261];
  ::ttnn::Tensor v264 = v1[262];
  ::ttnn::Tensor v265 = v1[263];
  ::ttnn::Tensor v266 = v1[264];
  ::ttnn::Tensor v267 = v1[265];
  ::ttnn::Tensor v268 = v1[266];
  ::ttnn::Tensor v269 = v1[267];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v270 = &main_const_eval_0;
  ::std::vector<::ttnn::Tensor> v271 = util_create_vec(v25);
  ::std::vector<::ttnn::Tensor> *v272 = &g_cached_result_main_const_eval_0;
  ttnn::constEvalFuncWrapper(v270, v271, v272);
  ::std::vector<::ttnn::Tensor> v273 = g_cached_result_main_const_eval_0;
  ::ttnn::Tensor v274 = v273[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v275 = &main_const_eval_1;
  ::std::vector<::ttnn::Tensor> v276 = util_create_vec(v173);
  ::std::vector<::ttnn::Tensor> *v277 = &g_cached_result_main_const_eval_1;
  ttnn::constEvalFuncWrapper(v275, v276, v277);
  ::std::vector<::ttnn::Tensor> v278 = g_cached_result_main_const_eval_1;
  ::ttnn::Tensor v279 = v278[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v280 = &main_const_eval_2;
  ::std::vector<::ttnn::Tensor> v281 = util_create_vec(v58);
  ::std::vector<::ttnn::Tensor> *v282 = &g_cached_result_main_const_eval_2;
  ttnn::constEvalFuncWrapper(v280, v281, v282);
  ::std::vector<::ttnn::Tensor> v283 = g_cached_result_main_const_eval_2;
  ::ttnn::Tensor v284 = v283[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v285 = &main_const_eval_3;
  ::std::vector<::ttnn::Tensor> v286 = util_create_vec(v85);
  ::std::vector<::ttnn::Tensor> *v287 = &g_cached_result_main_const_eval_3;
  ttnn::constEvalFuncWrapper(v285, v286, v287);
  ::std::vector<::ttnn::Tensor> v288 = g_cached_result_main_const_eval_3;
  ::ttnn::Tensor v289 = v288[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v290 = &main_const_eval_4;
  ::std::vector<::ttnn::Tensor> v291 = util_create_vec(v231);
  ::std::vector<::ttnn::Tensor> *v292 = &g_cached_result_main_const_eval_4;
  ttnn::constEvalFuncWrapper(v290, v291, v292);
  ::std::vector<::ttnn::Tensor> v293 = g_cached_result_main_const_eval_4;
  ::ttnn::Tensor v294 = v293[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v295 = &main_const_eval_5;
  ::std::vector<::ttnn::Tensor> v296 = util_create_vec(v93);
  ::std::vector<::ttnn::Tensor> *v297 = &g_cached_result_main_const_eval_5;
  ttnn::constEvalFuncWrapper(v295, v296, v297);
  ::std::vector<::ttnn::Tensor> v298 = g_cached_result_main_const_eval_5;
  ::ttnn::Tensor v299 = v298[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v300 = &main_const_eval_6;
  ::std::vector<::ttnn::Tensor> v301 = util_create_vec(v200);
  ::std::vector<::ttnn::Tensor> *v302 = &g_cached_result_main_const_eval_6;
  ttnn::constEvalFuncWrapper(v300, v301, v302);
  ::std::vector<::ttnn::Tensor> v303 = g_cached_result_main_const_eval_6;
  ::ttnn::Tensor v304 = v303[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v305 = &main_const_eval_7;
  ::std::vector<::ttnn::Tensor> v306 = util_create_vec(v121);
  ::std::vector<::ttnn::Tensor> *v307 = &g_cached_result_main_const_eval_7;
  ttnn::constEvalFuncWrapper(v305, v306, v307);
  ::std::vector<::ttnn::Tensor> v308 = g_cached_result_main_const_eval_7;
  ::ttnn::Tensor v309 = v308[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v310 = &main_const_eval_8;
  ::std::vector<::ttnn::Tensor> v311 = util_create_vec(v66);
  ::std::vector<::ttnn::Tensor> *v312 = &g_cached_result_main_const_eval_8;
  ttnn::constEvalFuncWrapper(v310, v311, v312);
  ::std::vector<::ttnn::Tensor> v313 = g_cached_result_main_const_eval_8;
  ::ttnn::Tensor v314 = v313[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v315 = &main_const_eval_9;
  ::std::vector<::ttnn::Tensor> v316 = util_create_vec(v101);
  ::std::vector<::ttnn::Tensor> *v317 = &g_cached_result_main_const_eval_9;
  ttnn::constEvalFuncWrapper(v315, v316, v317);
  ::std::vector<::ttnn::Tensor> v318 = g_cached_result_main_const_eval_9;
  ::ttnn::Tensor v319 = v318[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v320 = &main_const_eval_10;
  ::std::vector<::ttnn::Tensor> v321 = util_create_vec(v130);
  ::std::vector<::ttnn::Tensor> *v322 = &g_cached_result_main_const_eval_10;
  ttnn::constEvalFuncWrapper(v320, v321, v322);
  ::std::vector<::ttnn::Tensor> v323 = g_cached_result_main_const_eval_10;
  ::ttnn::Tensor v324 = v323[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v325 = &main_const_eval_11;
  ::std::vector<::ttnn::Tensor> v326 = util_create_vec(v191);
  ::std::vector<::ttnn::Tensor> *v327 = &g_cached_result_main_const_eval_11;
  ttnn::constEvalFuncWrapper(v325, v326, v327);
  ::std::vector<::ttnn::Tensor> v328 = g_cached_result_main_const_eval_11;
  ::ttnn::Tensor v329 = v328[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v330 = &main_const_eval_12;
  ::std::vector<::ttnn::Tensor> v331 = util_create_vec(v10);
  ::std::vector<::ttnn::Tensor> *v332 = &g_cached_result_main_const_eval_12;
  ttnn::constEvalFuncWrapper(v330, v331, v332);
  ::std::vector<::ttnn::Tensor> v333 = g_cached_result_main_const_eval_12;
  ::ttnn::Tensor v334 = v333[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v335 = &main_const_eval_13;
  ::std::vector<::ttnn::Tensor> v336 = util_create_vec(v162);
  ::std::vector<::ttnn::Tensor> *v337 = &g_cached_result_main_const_eval_13;
  ttnn::constEvalFuncWrapper(v335, v336, v337);
  ::std::vector<::ttnn::Tensor> v338 = g_cached_result_main_const_eval_13;
  ::ttnn::Tensor v339 = v338[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v340 = &main_const_eval_14;
  ::std::vector<::ttnn::Tensor> v341 = util_create_vec(v256);
  ::std::vector<::ttnn::Tensor> *v342 = &g_cached_result_main_const_eval_14;
  ttnn::constEvalFuncWrapper(v340, v341, v342);
  ::std::vector<::ttnn::Tensor> v343 = g_cached_result_main_const_eval_14;
  ::ttnn::Tensor v344 = v343[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v345 = &main_const_eval_15;
  ::std::vector<::ttnn::Tensor> v346 = util_create_vec(v5);
  ::std::vector<::ttnn::Tensor> *v347 = &g_cached_result_main_const_eval_15;
  ttnn::constEvalFuncWrapper(v345, v346, v347);
  ::std::vector<::ttnn::Tensor> v348 = g_cached_result_main_const_eval_15;
  ::ttnn::Tensor v349 = v348[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v350 = &main_const_eval_16;
  ::std::vector<::ttnn::Tensor> v351 = util_create_vec(v111);
  ::std::vector<::ttnn::Tensor> *v352 = &g_cached_result_main_const_eval_16;
  ttnn::constEvalFuncWrapper(v350, v351, v352);
  ::std::vector<::ttnn::Tensor> v353 = g_cached_result_main_const_eval_16;
  ::ttnn::Tensor v354 = v353[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v355 = &main_const_eval_17;
  ::std::vector<::ttnn::Tensor> v356 = util_create_vec(v110);
  ::std::vector<::ttnn::Tensor> *v357 = &g_cached_result_main_const_eval_17;
  ttnn::constEvalFuncWrapper(v355, v356, v357);
  ::std::vector<::ttnn::Tensor> v358 = g_cached_result_main_const_eval_17;
  ::ttnn::Tensor v359 = v358[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v360 = &main_const_eval_18;
  ::std::vector<::ttnn::Tensor> v361 = util_create_vec(v35);
  ::std::vector<::ttnn::Tensor> *v362 = &g_cached_result_main_const_eval_18;
  ttnn::constEvalFuncWrapper(v360, v361, v362);
  ::std::vector<::ttnn::Tensor> v363 = g_cached_result_main_const_eval_18;
  ::ttnn::Tensor v364 = v363[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v365 = &main_const_eval_19;
  ::std::vector<::ttnn::Tensor> v366 = util_create_vec(v176);
  ::std::vector<::ttnn::Tensor> *v367 = &g_cached_result_main_const_eval_19;
  ttnn::constEvalFuncWrapper(v365, v366, v367);
  ::std::vector<::ttnn::Tensor> v368 = g_cached_result_main_const_eval_19;
  ::ttnn::Tensor v369 = v368[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v370 = &main_const_eval_20;
  ::std::vector<::ttnn::Tensor> v371 = util_create_vec(v211);
  ::std::vector<::ttnn::Tensor> *v372 = &g_cached_result_main_const_eval_20;
  ttnn::constEvalFuncWrapper(v370, v371, v372);
  ::std::vector<::ttnn::Tensor> v373 = g_cached_result_main_const_eval_20;
  ::ttnn::Tensor v374 = v373[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v375 = &main_const_eval_21;
  ::std::vector<::ttnn::Tensor> v376 = util_create_vec(v155);
  ::std::vector<::ttnn::Tensor> *v377 = &g_cached_result_main_const_eval_21;
  ttnn::constEvalFuncWrapper(v375, v376, v377);
  ::std::vector<::ttnn::Tensor> v378 = g_cached_result_main_const_eval_21;
  ::ttnn::Tensor v379 = v378[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v380 = &main_const_eval_22;
  ::std::vector<::ttnn::Tensor> v381 = util_create_vec(v20);
  ::std::vector<::ttnn::Tensor> *v382 = &g_cached_result_main_const_eval_22;
  ttnn::constEvalFuncWrapper(v380, v381, v382);
  ::std::vector<::ttnn::Tensor> v383 = g_cached_result_main_const_eval_22;
  ::ttnn::Tensor v384 = v383[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v385 = &main_const_eval_23;
  ::std::vector<::ttnn::Tensor> v386 = util_create_vec(v65);
  ::std::vector<::ttnn::Tensor> *v387 = &g_cached_result_main_const_eval_23;
  ttnn::constEvalFuncWrapper(v385, v386, v387);
  ::std::vector<::ttnn::Tensor> v388 = g_cached_result_main_const_eval_23;
  ::ttnn::Tensor v389 = v388[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v390 = &main_const_eval_24;
  ::std::vector<::ttnn::Tensor> v391 = util_create_vec(v222);
  ::std::vector<::ttnn::Tensor> *v392 = &g_cached_result_main_const_eval_24;
  ttnn::constEvalFuncWrapper(v390, v391, v392);
  ::std::vector<::ttnn::Tensor> v393 = g_cached_result_main_const_eval_24;
  ::ttnn::Tensor v394 = v393[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v395 = &main_const_eval_25;
  ::std::vector<::ttnn::Tensor> v396 = util_create_vec(v107);
  ::std::vector<::ttnn::Tensor> *v397 = &g_cached_result_main_const_eval_25;
  ttnn::constEvalFuncWrapper(v395, v396, v397);
  ::std::vector<::ttnn::Tensor> v398 = g_cached_result_main_const_eval_25;
  ::ttnn::Tensor v399 = v398[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v400 = &main_const_eval_26;
  ::std::vector<::ttnn::Tensor> v401 = util_create_vec(v47);
  ::std::vector<::ttnn::Tensor> *v402 = &g_cached_result_main_const_eval_26;
  ttnn::constEvalFuncWrapper(v400, v401, v402);
  ::std::vector<::ttnn::Tensor> v403 = g_cached_result_main_const_eval_26;
  ::ttnn::Tensor v404 = v403[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v405 = &main_const_eval_27;
  ::std::vector<::ttnn::Tensor> v406 = util_create_vec(v196);
  ::std::vector<::ttnn::Tensor> *v407 = &g_cached_result_main_const_eval_27;
  ttnn::constEvalFuncWrapper(v405, v406, v407);
  ::std::vector<::ttnn::Tensor> v408 = g_cached_result_main_const_eval_27;
  ::ttnn::Tensor v409 = v408[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v410 = &main_const_eval_28;
  ::std::vector<::ttnn::Tensor> v411 = util_create_vec(v202);
  ::std::vector<::ttnn::Tensor> *v412 = &g_cached_result_main_const_eval_28;
  ttnn::constEvalFuncWrapper(v410, v411, v412);
  ::std::vector<::ttnn::Tensor> v413 = g_cached_result_main_const_eval_28;
  ::ttnn::Tensor v414 = v413[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v415 = &main_const_eval_29;
  ::std::vector<::ttnn::Tensor> v416 = util_create_vec(v87);
  ::std::vector<::ttnn::Tensor> *v417 = &g_cached_result_main_const_eval_29;
  ttnn::constEvalFuncWrapper(v415, v416, v417);
  ::std::vector<::ttnn::Tensor> v418 = g_cached_result_main_const_eval_29;
  ::ttnn::Tensor v419 = v418[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v420 = &main_const_eval_30;
  ::std::vector<::ttnn::Tensor> v421 = util_create_vec(v63);
  ::std::vector<::ttnn::Tensor> *v422 = &g_cached_result_main_const_eval_30;
  ttnn::constEvalFuncWrapper(v420, v421, v422);
  ::std::vector<::ttnn::Tensor> v423 = g_cached_result_main_const_eval_30;
  ::ttnn::Tensor v424 = v423[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v425 = &main_const_eval_31;
  ::std::vector<::ttnn::Tensor> v426 = util_create_vec(v183);
  ::std::vector<::ttnn::Tensor> *v427 = &g_cached_result_main_const_eval_31;
  ttnn::constEvalFuncWrapper(v425, v426, v427);
  ::std::vector<::ttnn::Tensor> v428 = g_cached_result_main_const_eval_31;
  ::ttnn::Tensor v429 = v428[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v430 = &main_const_eval_32;
  ::std::vector<::ttnn::Tensor> v431 = util_create_vec(v19);
  ::std::vector<::ttnn::Tensor> *v432 = &g_cached_result_main_const_eval_32;
  ttnn::constEvalFuncWrapper(v430, v431, v432);
  ::std::vector<::ttnn::Tensor> v433 = g_cached_result_main_const_eval_32;
  ::ttnn::Tensor v434 = v433[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v435 = &main_const_eval_33;
  ::std::vector<::ttnn::Tensor> v436 = util_create_vec(v81);
  ::std::vector<::ttnn::Tensor> *v437 = &g_cached_result_main_const_eval_33;
  ttnn::constEvalFuncWrapper(v435, v436, v437);
  ::std::vector<::ttnn::Tensor> v438 = g_cached_result_main_const_eval_33;
  ::ttnn::Tensor v439 = v438[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v440 = &main_const_eval_34;
  ::std::vector<::ttnn::Tensor> v441 = util_create_vec(v77);
  ::std::vector<::ttnn::Tensor> *v442 = &g_cached_result_main_const_eval_34;
  ttnn::constEvalFuncWrapper(v440, v441, v442);
  ::std::vector<::ttnn::Tensor> v443 = g_cached_result_main_const_eval_34;
  ::ttnn::Tensor v444 = v443[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v445 = &main_const_eval_35;
  ::std::vector<::ttnn::Tensor> v446 = util_create_vec(v51);
  ::std::vector<::ttnn::Tensor> *v447 = &g_cached_result_main_const_eval_35;
  ttnn::constEvalFuncWrapper(v445, v446, v447);
  ::std::vector<::ttnn::Tensor> v448 = g_cached_result_main_const_eval_35;
  ::ttnn::Tensor v449 = v448[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v450 = &main_const_eval_36;
  ::std::vector<::ttnn::Tensor> v451 = util_create_vec(v75);
  ::std::vector<::ttnn::Tensor> *v452 = &g_cached_result_main_const_eval_36;
  ttnn::constEvalFuncWrapper(v450, v451, v452);
  ::std::vector<::ttnn::Tensor> v453 = g_cached_result_main_const_eval_36;
  ::ttnn::Tensor v454 = v453[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v455 = &main_const_eval_37;
  ::std::vector<::ttnn::Tensor> v456 = util_create_vec(v210);
  ::std::vector<::ttnn::Tensor> *v457 = &g_cached_result_main_const_eval_37;
  ttnn::constEvalFuncWrapper(v455, v456, v457);
  ::std::vector<::ttnn::Tensor> v458 = g_cached_result_main_const_eval_37;
  ::ttnn::Tensor v459 = v458[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v460 = &main_const_eval_38;
  ::std::vector<::ttnn::Tensor> v461 = util_create_vec(v235);
  ::std::vector<::ttnn::Tensor> *v462 = &g_cached_result_main_const_eval_38;
  ttnn::constEvalFuncWrapper(v460, v461, v462);
  ::std::vector<::ttnn::Tensor> v463 = g_cached_result_main_const_eval_38;
  ::ttnn::Tensor v464 = v463[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v465 = &main_const_eval_39;
  ::std::vector<::ttnn::Tensor> v466 = util_create_vec(v175);
  ::std::vector<::ttnn::Tensor> *v467 = &g_cached_result_main_const_eval_39;
  ttnn::constEvalFuncWrapper(v465, v466, v467);
  ::std::vector<::ttnn::Tensor> v468 = g_cached_result_main_const_eval_39;
  ::ttnn::Tensor v469 = v468[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v470 = &main_const_eval_40;
  ::std::vector<::ttnn::Tensor> v471 = util_create_vec(v71);
  ::std::vector<::ttnn::Tensor> *v472 = &g_cached_result_main_const_eval_40;
  ttnn::constEvalFuncWrapper(v470, v471, v472);
  ::std::vector<::ttnn::Tensor> v473 = g_cached_result_main_const_eval_40;
  ::ttnn::Tensor v474 = v473[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v475 = &main_const_eval_41;
  ::std::vector<::ttnn::Tensor> v476 = util_create_vec(v220);
  ::std::vector<::ttnn::Tensor> *v477 = &g_cached_result_main_const_eval_41;
  ttnn::constEvalFuncWrapper(v475, v476, v477);
  ::std::vector<::ttnn::Tensor> v478 = g_cached_result_main_const_eval_41;
  ::ttnn::Tensor v479 = v478[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v480 = &main_const_eval_42;
  ::std::vector<::ttnn::Tensor> v481 = util_create_vec(v73);
  ::std::vector<::ttnn::Tensor> *v482 = &g_cached_result_main_const_eval_42;
  ttnn::constEvalFuncWrapper(v480, v481, v482);
  ::std::vector<::ttnn::Tensor> v483 = g_cached_result_main_const_eval_42;
  ::ttnn::Tensor v484 = v483[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v485 = &main_const_eval_43;
  ::std::vector<::ttnn::Tensor> v486 = util_create_vec(v118);
  ::std::vector<::ttnn::Tensor> *v487 = &g_cached_result_main_const_eval_43;
  ttnn::constEvalFuncWrapper(v485, v486, v487);
  ::std::vector<::ttnn::Tensor> v488 = g_cached_result_main_const_eval_43;
  ::ttnn::Tensor v489 = v488[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v490 = &main_const_eval_44;
  ::std::vector<::ttnn::Tensor> v491 = util_create_vec(v197);
  ::std::vector<::ttnn::Tensor> *v492 = &g_cached_result_main_const_eval_44;
  ttnn::constEvalFuncWrapper(v490, v491, v492);
  ::std::vector<::ttnn::Tensor> v493 = g_cached_result_main_const_eval_44;
  ::ttnn::Tensor v494 = v493[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v495 = &main_const_eval_45;
  ::std::vector<::ttnn::Tensor> v496 = util_create_vec(v181);
  ::std::vector<::ttnn::Tensor> *v497 = &g_cached_result_main_const_eval_45;
  ttnn::constEvalFuncWrapper(v495, v496, v497);
  ::std::vector<::ttnn::Tensor> v498 = g_cached_result_main_const_eval_45;
  ::ttnn::Tensor v499 = v498[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v500 = &main_const_eval_46;
  ::std::vector<::ttnn::Tensor> v501 = util_create_vec(v96);
  ::std::vector<::ttnn::Tensor> *v502 = &g_cached_result_main_const_eval_46;
  ttnn::constEvalFuncWrapper(v500, v501, v502);
  ::std::vector<::ttnn::Tensor> v503 = g_cached_result_main_const_eval_46;
  ::ttnn::Tensor v504 = v503[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v505 = &main_const_eval_47;
  ::std::vector<::ttnn::Tensor> v506 = util_create_vec(v218);
  ::std::vector<::ttnn::Tensor> *v507 = &g_cached_result_main_const_eval_47;
  ttnn::constEvalFuncWrapper(v505, v506, v507);
  ::std::vector<::ttnn::Tensor> v508 = g_cached_result_main_const_eval_47;
  ::ttnn::Tensor v509 = v508[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v510 = &main_const_eval_48;
  ::std::vector<::ttnn::Tensor> v511 = util_create_vec(v100);
  ::std::vector<::ttnn::Tensor> *v512 = &g_cached_result_main_const_eval_48;
  ttnn::constEvalFuncWrapper(v510, v511, v512);
  ::std::vector<::ttnn::Tensor> v513 = g_cached_result_main_const_eval_48;
  ::ttnn::Tensor v514 = v513[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v515 = &main_const_eval_49;
  ::std::vector<::ttnn::Tensor> v516 = util_create_vec(v186);
  ::std::vector<::ttnn::Tensor> *v517 = &g_cached_result_main_const_eval_49;
  ttnn::constEvalFuncWrapper(v515, v516, v517);
  ::std::vector<::ttnn::Tensor> v518 = g_cached_result_main_const_eval_49;
  ::ttnn::Tensor v519 = v518[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v520 = &main_const_eval_50;
  ::std::vector<::ttnn::Tensor> v521 = util_create_vec(v167);
  ::std::vector<::ttnn::Tensor> *v522 = &g_cached_result_main_const_eval_50;
  ttnn::constEvalFuncWrapper(v520, v521, v522);
  ::std::vector<::ttnn::Tensor> v523 = g_cached_result_main_const_eval_50;
  ::ttnn::Tensor v524 = v523[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v525 = &main_const_eval_51;
  ::std::vector<::ttnn::Tensor> v526 = util_create_vec(v198);
  ::std::vector<::ttnn::Tensor> *v527 = &g_cached_result_main_const_eval_51;
  ttnn::constEvalFuncWrapper(v525, v526, v527);
  ::std::vector<::ttnn::Tensor> v528 = g_cached_result_main_const_eval_51;
  ::ttnn::Tensor v529 = v528[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v530 = &main_const_eval_52;
  ::std::vector<::ttnn::Tensor> v531 = util_create_vec(v105);
  ::std::vector<::ttnn::Tensor> *v532 = &g_cached_result_main_const_eval_52;
  ttnn::constEvalFuncWrapper(v530, v531, v532);
  ::std::vector<::ttnn::Tensor> v533 = g_cached_result_main_const_eval_52;
  ::ttnn::Tensor v534 = v533[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v535 = &main_const_eval_53;
  ::std::vector<::ttnn::Tensor> v536 = util_create_vec(v178);
  ::std::vector<::ttnn::Tensor> *v537 = &g_cached_result_main_const_eval_53;
  ttnn::constEvalFuncWrapper(v535, v536, v537);
  ::std::vector<::ttnn::Tensor> v538 = g_cached_result_main_const_eval_53;
  ::ttnn::Tensor v539 = v538[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v540 = &main_const_eval_54;
  ::std::vector<::ttnn::Tensor> v541 = util_create_vec(v22);
  ::std::vector<::ttnn::Tensor> *v542 = &g_cached_result_main_const_eval_54;
  ttnn::constEvalFuncWrapper(v540, v541, v542);
  ::std::vector<::ttnn::Tensor> v543 = g_cached_result_main_const_eval_54;
  ::ttnn::Tensor v544 = v543[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v545 = &main_const_eval_55;
  ::std::vector<::ttnn::Tensor> v546 = util_create_vec(v251);
  ::std::vector<::ttnn::Tensor> *v547 = &g_cached_result_main_const_eval_55;
  ttnn::constEvalFuncWrapper(v545, v546, v547);
  ::std::vector<::ttnn::Tensor> v548 = g_cached_result_main_const_eval_55;
  ::ttnn::Tensor v549 = v548[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v550 = &main_const_eval_56;
  ::std::vector<::ttnn::Tensor> v551 = util_create_vec(v261);
  ::std::vector<::ttnn::Tensor> *v552 = &g_cached_result_main_const_eval_56;
  ttnn::constEvalFuncWrapper(v550, v551, v552);
  ::std::vector<::ttnn::Tensor> v553 = g_cached_result_main_const_eval_56;
  ::ttnn::Tensor v554 = v553[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v555 = &main_const_eval_57;
  ::std::vector<::ttnn::Tensor> v556 = util_create_vec(v125);
  ::std::vector<::ttnn::Tensor> *v557 = &g_cached_result_main_const_eval_57;
  ttnn::constEvalFuncWrapper(v555, v556, v557);
  ::std::vector<::ttnn::Tensor> v558 = g_cached_result_main_const_eval_57;
  ::ttnn::Tensor v559 = v558[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v560 = &main_const_eval_58;
  ::std::vector<::ttnn::Tensor> v561 = util_create_vec(v95);
  ::std::vector<::ttnn::Tensor> *v562 = &g_cached_result_main_const_eval_58;
  ttnn::constEvalFuncWrapper(v560, v561, v562);
  ::std::vector<::ttnn::Tensor> v563 = g_cached_result_main_const_eval_58;
  ::ttnn::Tensor v564 = v563[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v565 = &main_const_eval_59;
  ::std::vector<::ttnn::Tensor> v566 = util_create_vec(v31);
  ::std::vector<::ttnn::Tensor> *v567 = &g_cached_result_main_const_eval_59;
  ttnn::constEvalFuncWrapper(v565, v566, v567);
  ::std::vector<::ttnn::Tensor> v568 = g_cached_result_main_const_eval_59;
  ::ttnn::Tensor v569 = v568[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v570 = &main_const_eval_60;
  ::std::vector<::ttnn::Tensor> v571 = util_create_vec(v215);
  ::std::vector<::ttnn::Tensor> *v572 = &g_cached_result_main_const_eval_60;
  ttnn::constEvalFuncWrapper(v570, v571, v572);
  ::std::vector<::ttnn::Tensor> v573 = g_cached_result_main_const_eval_60;
  ::ttnn::Tensor v574 = v573[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v575 = &main_const_eval_61;
  ::std::vector<::ttnn::Tensor> v576 = util_create_vec(v17);
  ::std::vector<::ttnn::Tensor> *v577 = &g_cached_result_main_const_eval_61;
  ttnn::constEvalFuncWrapper(v575, v576, v577);
  ::std::vector<::ttnn::Tensor> v578 = g_cached_result_main_const_eval_61;
  ::ttnn::Tensor v579 = v578[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v580 = &main_const_eval_62;
  ::std::vector<::ttnn::Tensor> v581 = util_create_vec(v126);
  ::std::vector<::ttnn::Tensor> *v582 = &g_cached_result_main_const_eval_62;
  ttnn::constEvalFuncWrapper(v580, v581, v582);
  ::std::vector<::ttnn::Tensor> v583 = g_cached_result_main_const_eval_62;
  ::ttnn::Tensor v584 = v583[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v585 = &main_const_eval_63;
  ::std::vector<::ttnn::Tensor> v586 = util_create_vec(v82);
  ::std::vector<::ttnn::Tensor> *v587 = &g_cached_result_main_const_eval_63;
  ttnn::constEvalFuncWrapper(v585, v586, v587);
  ::std::vector<::ttnn::Tensor> v588 = g_cached_result_main_const_eval_63;
  ::ttnn::Tensor v589 = v588[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v590 = &main_const_eval_64;
  ::std::vector<::ttnn::Tensor> v591 = util_create_vec(v143);
  ::std::vector<::ttnn::Tensor> *v592 = &g_cached_result_main_const_eval_64;
  ttnn::constEvalFuncWrapper(v590, v591, v592);
  ::std::vector<::ttnn::Tensor> v593 = g_cached_result_main_const_eval_64;
  ::ttnn::Tensor v594 = v593[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v595 = &main_const_eval_65;
  ::std::vector<::ttnn::Tensor> v596 = util_create_vec(v108);
  ::std::vector<::ttnn::Tensor> *v597 = &g_cached_result_main_const_eval_65;
  ttnn::constEvalFuncWrapper(v595, v596, v597);
  ::std::vector<::ttnn::Tensor> v598 = g_cached_result_main_const_eval_65;
  ::ttnn::Tensor v599 = v598[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v600 = &main_const_eval_66;
  ::std::vector<::ttnn::Tensor> v601 = util_create_vec(v228);
  ::std::vector<::ttnn::Tensor> *v602 = &g_cached_result_main_const_eval_66;
  ttnn::constEvalFuncWrapper(v600, v601, v602);
  ::std::vector<::ttnn::Tensor> v603 = g_cached_result_main_const_eval_66;
  ::ttnn::Tensor v604 = v603[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v605 = &main_const_eval_67;
  ::std::vector<::ttnn::Tensor> v606 = util_create_vec(v14);
  ::std::vector<::ttnn::Tensor> *v607 = &g_cached_result_main_const_eval_67;
  ttnn::constEvalFuncWrapper(v605, v606, v607);
  ::std::vector<::ttnn::Tensor> v608 = g_cached_result_main_const_eval_67;
  ::ttnn::Tensor v609 = v608[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v610 = &main_const_eval_68;
  ::std::vector<::ttnn::Tensor> v611 = util_create_vec(v205);
  ::std::vector<::ttnn::Tensor> *v612 = &g_cached_result_main_const_eval_68;
  ttnn::constEvalFuncWrapper(v610, v611, v612);
  ::std::vector<::ttnn::Tensor> v613 = g_cached_result_main_const_eval_68;
  ::ttnn::Tensor v614 = v613[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v615 = &main_const_eval_69;
  ::std::vector<::ttnn::Tensor> v616 = util_create_vec(v255);
  ::std::vector<::ttnn::Tensor> *v617 = &g_cached_result_main_const_eval_69;
  ttnn::constEvalFuncWrapper(v615, v616, v617);
  ::std::vector<::ttnn::Tensor> v618 = g_cached_result_main_const_eval_69;
  ::ttnn::Tensor v619 = v618[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v620 = &main_const_eval_70;
  ::std::vector<::ttnn::Tensor> v621 = util_create_vec(v241);
  ::std::vector<::ttnn::Tensor> *v622 = &g_cached_result_main_const_eval_70;
  ttnn::constEvalFuncWrapper(v620, v621, v622);
  ::std::vector<::ttnn::Tensor> v623 = g_cached_result_main_const_eval_70;
  ::ttnn::Tensor v624 = v623[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v625 = &main_const_eval_71;
  ::std::vector<::ttnn::Tensor> v626 = util_create_vec(v240);
  ::std::vector<::ttnn::Tensor> *v627 = &g_cached_result_main_const_eval_71;
  ttnn::constEvalFuncWrapper(v625, v626, v627);
  ::std::vector<::ttnn::Tensor> v628 = g_cached_result_main_const_eval_71;
  ::ttnn::Tensor v629 = v628[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v630 = &main_const_eval_72;
  ::std::vector<::ttnn::Tensor> v631 = util_create_vec(v68);
  ::std::vector<::ttnn::Tensor> *v632 = &g_cached_result_main_const_eval_72;
  ttnn::constEvalFuncWrapper(v630, v631, v632);
  ::std::vector<::ttnn::Tensor> v633 = g_cached_result_main_const_eval_72;
  ::ttnn::Tensor v634 = v633[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v635 = &main_const_eval_73;
  ::std::vector<::ttnn::Tensor> v636 = util_create_vec(v61);
  ::std::vector<::ttnn::Tensor> *v637 = &g_cached_result_main_const_eval_73;
  ttnn::constEvalFuncWrapper(v635, v636, v637);
  ::std::vector<::ttnn::Tensor> v638 = g_cached_result_main_const_eval_73;
  ::ttnn::Tensor v639 = v638[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v640 = &main_const_eval_74;
  ::std::vector<::ttnn::Tensor> v641 = util_create_vec(v146);
  ::std::vector<::ttnn::Tensor> *v642 = &g_cached_result_main_const_eval_74;
  ttnn::constEvalFuncWrapper(v640, v641, v642);
  ::std::vector<::ttnn::Tensor> v643 = g_cached_result_main_const_eval_74;
  ::ttnn::Tensor v644 = v643[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v645 = &main_const_eval_75;
  ::std::vector<::ttnn::Tensor> v646 = util_create_vec(v250);
  ::std::vector<::ttnn::Tensor> *v647 = &g_cached_result_main_const_eval_75;
  ttnn::constEvalFuncWrapper(v645, v646, v647);
  ::std::vector<::ttnn::Tensor> v648 = g_cached_result_main_const_eval_75;
  ::ttnn::Tensor v649 = v648[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v650 = &main_const_eval_76;
  ::std::vector<::ttnn::Tensor> v651 = util_create_vec(v265);
  ::std::vector<::ttnn::Tensor> *v652 = &g_cached_result_main_const_eval_76;
  ttnn::constEvalFuncWrapper(v650, v651, v652);
  ::std::vector<::ttnn::Tensor> v653 = g_cached_result_main_const_eval_76;
  ::ttnn::Tensor v654 = v653[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v655 = &main_const_eval_77;
  ::std::vector<::ttnn::Tensor> v656 = util_create_vec(v12);
  ::std::vector<::ttnn::Tensor> *v657 = &g_cached_result_main_const_eval_77;
  ttnn::constEvalFuncWrapper(v655, v656, v657);
  ::std::vector<::ttnn::Tensor> v658 = g_cached_result_main_const_eval_77;
  ::ttnn::Tensor v659 = v658[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v660 = &main_const_eval_78;
  ::std::vector<::ttnn::Tensor> v661 = util_create_vec(v83);
  ::std::vector<::ttnn::Tensor> *v662 = &g_cached_result_main_const_eval_78;
  ttnn::constEvalFuncWrapper(v660, v661, v662);
  ::std::vector<::ttnn::Tensor> v663 = g_cached_result_main_const_eval_78;
  ::ttnn::Tensor v664 = v663[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v665 = &main_const_eval_79;
  ::std::vector<::ttnn::Tensor> v666 = util_create_vec(v227);
  ::std::vector<::ttnn::Tensor> *v667 = &g_cached_result_main_const_eval_79;
  ttnn::constEvalFuncWrapper(v665, v666, v667);
  ::std::vector<::ttnn::Tensor> v668 = g_cached_result_main_const_eval_79;
  ::ttnn::Tensor v669 = v668[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v670 = &main_const_eval_80;
  ::std::vector<::ttnn::Tensor> v671 = util_create_vec(v136);
  ::std::vector<::ttnn::Tensor> *v672 = &g_cached_result_main_const_eval_80;
  ttnn::constEvalFuncWrapper(v670, v671, v672);
  ::std::vector<::ttnn::Tensor> v673 = g_cached_result_main_const_eval_80;
  ::ttnn::Tensor v674 = v673[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v675 = &main_const_eval_81;
  ::std::vector<::ttnn::Tensor> v676 = util_create_vec(v258);
  ::std::vector<::ttnn::Tensor> *v677 = &g_cached_result_main_const_eval_81;
  ttnn::constEvalFuncWrapper(v675, v676, v677);
  ::std::vector<::ttnn::Tensor> v678 = g_cached_result_main_const_eval_81;
  ::ttnn::Tensor v679 = v678[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v680 = &main_const_eval_82;
  ::std::vector<::ttnn::Tensor> v681 = util_create_vec(v27);
  ::std::vector<::ttnn::Tensor> *v682 = &g_cached_result_main_const_eval_82;
  ttnn::constEvalFuncWrapper(v680, v681, v682);
  ::std::vector<::ttnn::Tensor> v683 = g_cached_result_main_const_eval_82;
  ::ttnn::Tensor v684 = v683[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v685 = &main_const_eval_83;
  ::std::vector<::ttnn::Tensor> v686 = util_create_vec(v91);
  ::std::vector<::ttnn::Tensor> *v687 = &g_cached_result_main_const_eval_83;
  ttnn::constEvalFuncWrapper(v685, v686, v687);
  ::std::vector<::ttnn::Tensor> v688 = g_cached_result_main_const_eval_83;
  ::ttnn::Tensor v689 = v688[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v690 = &main_const_eval_84;
  ::std::vector<::ttnn::Tensor> v691 = util_create_vec(v190);
  ::std::vector<::ttnn::Tensor> *v692 = &g_cached_result_main_const_eval_84;
  ttnn::constEvalFuncWrapper(v690, v691, v692);
  ::std::vector<::ttnn::Tensor> v693 = g_cached_result_main_const_eval_84;
  ::ttnn::Tensor v694 = v693[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v695 = &main_const_eval_85;
  ::std::vector<::ttnn::Tensor> v696 = util_create_vec(v230);
  ::std::vector<::ttnn::Tensor> *v697 = &g_cached_result_main_const_eval_85;
  ttnn::constEvalFuncWrapper(v695, v696, v697);
  ::std::vector<::ttnn::Tensor> v698 = g_cached_result_main_const_eval_85;
  ::ttnn::Tensor v699 = v698[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v700 = &main_const_eval_86;
  ::std::vector<::ttnn::Tensor> v701 = util_create_vec(v98);
  ::std::vector<::ttnn::Tensor> *v702 = &g_cached_result_main_const_eval_86;
  ttnn::constEvalFuncWrapper(v700, v701, v702);
  ::std::vector<::ttnn::Tensor> v703 = g_cached_result_main_const_eval_86;
  ::ttnn::Tensor v704 = v703[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v705 = &main_const_eval_87;
  ::std::vector<::ttnn::Tensor> v706 = util_create_vec(v182);
  ::std::vector<::ttnn::Tensor> *v707 = &g_cached_result_main_const_eval_87;
  ttnn::constEvalFuncWrapper(v705, v706, v707);
  ::std::vector<::ttnn::Tensor> v708 = g_cached_result_main_const_eval_87;
  ::ttnn::Tensor v709 = v708[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v710 = &main_const_eval_88;
  ::std::vector<::ttnn::Tensor> v711 = util_create_vec(v238);
  ::std::vector<::ttnn::Tensor> *v712 = &g_cached_result_main_const_eval_88;
  ttnn::constEvalFuncWrapper(v710, v711, v712);
  ::std::vector<::ttnn::Tensor> v713 = g_cached_result_main_const_eval_88;
  ::ttnn::Tensor v714 = v713[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v715 = &main_const_eval_89;
  ::std::vector<::ttnn::Tensor> v716 = util_create_vec(v92);
  ::std::vector<::ttnn::Tensor> *v717 = &g_cached_result_main_const_eval_89;
  ttnn::constEvalFuncWrapper(v715, v716, v717);
  ::std::vector<::ttnn::Tensor> v718 = g_cached_result_main_const_eval_89;
  ::ttnn::Tensor v719 = v718[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v720 = &main_const_eval_90;
  ::std::vector<::ttnn::Tensor> v721 = util_create_vec(v185);
  ::std::vector<::ttnn::Tensor> *v722 = &g_cached_result_main_const_eval_90;
  ttnn::constEvalFuncWrapper(v720, v721, v722);
  ::std::vector<::ttnn::Tensor> v723 = g_cached_result_main_const_eval_90;
  ::ttnn::Tensor v724 = v723[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v725 = &main_const_eval_91;
  ::std::vector<::ttnn::Tensor> v726 = util_create_vec(v117);
  ::std::vector<::ttnn::Tensor> *v727 = &g_cached_result_main_const_eval_91;
  ttnn::constEvalFuncWrapper(v725, v726, v727);
  ::std::vector<::ttnn::Tensor> v728 = g_cached_result_main_const_eval_91;
  ::ttnn::Tensor v729 = v728[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v730 = &main_const_eval_92;
  ::std::vector<::ttnn::Tensor> v731 = util_create_vec(v41);
  ::std::vector<::ttnn::Tensor> *v732 = &g_cached_result_main_const_eval_92;
  ttnn::constEvalFuncWrapper(v730, v731, v732);
  ::std::vector<::ttnn::Tensor> v733 = g_cached_result_main_const_eval_92;
  ::ttnn::Tensor v734 = v733[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v735 = &main_const_eval_93;
  ::std::vector<::ttnn::Tensor> v736 = util_create_vec(v208);
  ::std::vector<::ttnn::Tensor> *v737 = &g_cached_result_main_const_eval_93;
  ttnn::constEvalFuncWrapper(v735, v736, v737);
  ::std::vector<::ttnn::Tensor> v738 = g_cached_result_main_const_eval_93;
  ::ttnn::Tensor v739 = v738[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v740 = &main_const_eval_94;
  ::std::vector<::ttnn::Tensor> v741 = util_create_vec(v50);
  ::std::vector<::ttnn::Tensor> *v742 = &g_cached_result_main_const_eval_94;
  ttnn::constEvalFuncWrapper(v740, v741, v742);
  ::std::vector<::ttnn::Tensor> v743 = g_cached_result_main_const_eval_94;
  ::ttnn::Tensor v744 = v743[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v745 = &main_const_eval_95;
  ::std::vector<::ttnn::Tensor> v746 = util_create_vec(v262);
  ::std::vector<::ttnn::Tensor> *v747 = &g_cached_result_main_const_eval_95;
  ttnn::constEvalFuncWrapper(v745, v746, v747);
  ::std::vector<::ttnn::Tensor> v748 = g_cached_result_main_const_eval_95;
  ::ttnn::Tensor v749 = v748[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v750 = &main_const_eval_96;
  ::std::vector<::ttnn::Tensor> v751 = util_create_vec(v86);
  ::std::vector<::ttnn::Tensor> *v752 = &g_cached_result_main_const_eval_96;
  ttnn::constEvalFuncWrapper(v750, v751, v752);
  ::std::vector<::ttnn::Tensor> v753 = g_cached_result_main_const_eval_96;
  ::ttnn::Tensor v754 = v753[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v755 = &main_const_eval_97;
  ::std::vector<::ttnn::Tensor> v756 = util_create_vec(v7);
  ::std::vector<::ttnn::Tensor> *v757 = &g_cached_result_main_const_eval_97;
  ttnn::constEvalFuncWrapper(v755, v756, v757);
  ::std::vector<::ttnn::Tensor> v758 = g_cached_result_main_const_eval_97;
  ::ttnn::Tensor v759 = v758[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v760 = &main_const_eval_98;
  ::std::vector<::ttnn::Tensor> v761 = util_create_vec(v103);
  ::std::vector<::ttnn::Tensor> *v762 = &g_cached_result_main_const_eval_98;
  ttnn::constEvalFuncWrapper(v760, v761, v762);
  ::std::vector<::ttnn::Tensor> v763 = g_cached_result_main_const_eval_98;
  ::ttnn::Tensor v764 = v763[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v765 = &main_const_eval_99;
  ::std::vector<::ttnn::Tensor> v766 = util_create_vec(v166);
  ::std::vector<::ttnn::Tensor> *v767 = &g_cached_result_main_const_eval_99;
  ttnn::constEvalFuncWrapper(v765, v766, v767);
  ::std::vector<::ttnn::Tensor> v768 = g_cached_result_main_const_eval_99;
  ::ttnn::Tensor v769 = v768[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v770 = &main_const_eval_100;
  ::std::vector<::ttnn::Tensor> v771 = util_create_vec(v147);
  ::std::vector<::ttnn::Tensor> *v772 = &g_cached_result_main_const_eval_100;
  ttnn::constEvalFuncWrapper(v770, v771, v772);
  ::std::vector<::ttnn::Tensor> v773 = g_cached_result_main_const_eval_100;
  ::ttnn::Tensor v774 = v773[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v775 = &main_const_eval_101;
  ::std::vector<::ttnn::Tensor> v776 = util_create_vec(v62);
  ::std::vector<::ttnn::Tensor> *v777 = &g_cached_result_main_const_eval_101;
  ttnn::constEvalFuncWrapper(v775, v776, v777);
  ::std::vector<::ttnn::Tensor> v778 = g_cached_result_main_const_eval_101;
  ::ttnn::Tensor v779 = v778[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v780 = &main_const_eval_102;
  ::std::vector<::ttnn::Tensor> v781 = util_create_vec(v151);
  ::std::vector<::ttnn::Tensor> *v782 = &g_cached_result_main_const_eval_102;
  ttnn::constEvalFuncWrapper(v780, v781, v782);
  ::std::vector<::ttnn::Tensor> v783 = g_cached_result_main_const_eval_102;
  ::ttnn::Tensor v784 = v783[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v785 = &main_const_eval_103;
  ::std::vector<::ttnn::Tensor> v786 = util_create_vec(v268);
  ::std::vector<::ttnn::Tensor> *v787 = &g_cached_result_main_const_eval_103;
  ttnn::constEvalFuncWrapper(v785, v786, v787);
  ::std::vector<::ttnn::Tensor> v788 = g_cached_result_main_const_eval_103;
  ::ttnn::Tensor v789 = v788[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v790 = &main_const_eval_104;
  ::std::vector<::ttnn::Tensor> v791 = util_create_vec(v70);
  ::std::vector<::ttnn::Tensor> *v792 = &g_cached_result_main_const_eval_104;
  ttnn::constEvalFuncWrapper(v790, v791, v792);
  ::std::vector<::ttnn::Tensor> v793 = g_cached_result_main_const_eval_104;
  ::ttnn::Tensor v794 = v793[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v795 = &main_const_eval_105;
  ::std::vector<::ttnn::Tensor> v796 = util_create_vec(v160);
  ::std::vector<::ttnn::Tensor> *v797 = &g_cached_result_main_const_eval_105;
  ttnn::constEvalFuncWrapper(v795, v796, v797);
  ::std::vector<::ttnn::Tensor> v798 = g_cached_result_main_const_eval_105;
  ::ttnn::Tensor v799 = v798[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v800 = &main_const_eval_106;
  ::std::vector<::ttnn::Tensor> v801 = util_create_vec(v245);
  ::std::vector<::ttnn::Tensor> *v802 = &g_cached_result_main_const_eval_106;
  ttnn::constEvalFuncWrapper(v800, v801, v802);
  ::std::vector<::ttnn::Tensor> v803 = g_cached_result_main_const_eval_106;
  ::ttnn::Tensor v804 = v803[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v805 = &main_const_eval_107;
  ::std::vector<::ttnn::Tensor> v806 = util_create_vec(v4);
  ::std::vector<::ttnn::Tensor> *v807 = &g_cached_result_main_const_eval_107;
  ttnn::constEvalFuncWrapper(v805, v806, v807);
  ::std::vector<::ttnn::Tensor> v808 = g_cached_result_main_const_eval_107;
  ::ttnn::Tensor v809 = v808[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v810 = &main_const_eval_108;
  ::std::vector<::ttnn::Tensor> v811 = util_create_vec(v113);
  ::std::vector<::ttnn::Tensor> *v812 = &g_cached_result_main_const_eval_108;
  ttnn::constEvalFuncWrapper(v810, v811, v812);
  ::std::vector<::ttnn::Tensor> v813 = g_cached_result_main_const_eval_108;
  ::ttnn::Tensor v814 = v813[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v815 = &main_const_eval_109;
  ::std::vector<::ttnn::Tensor> v816 = util_create_vec(v223);
  ::std::vector<::ttnn::Tensor> *v817 = &g_cached_result_main_const_eval_109;
  ttnn::constEvalFuncWrapper(v815, v816, v817);
  ::std::vector<::ttnn::Tensor> v818 = g_cached_result_main_const_eval_109;
  ::ttnn::Tensor v819 = v818[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v820 = &main_const_eval_110;
  ::std::vector<::ttnn::Tensor> v821 = util_create_vec(v48);
  ::std::vector<::ttnn::Tensor> *v822 = &g_cached_result_main_const_eval_110;
  ttnn::constEvalFuncWrapper(v820, v821, v822);
  ::std::vector<::ttnn::Tensor> v823 = g_cached_result_main_const_eval_110;
  ::ttnn::Tensor v824 = v823[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v825 = &main_const_eval_111;
  ::std::vector<::ttnn::Tensor> v826 = util_create_vec(v78);
  ::std::vector<::ttnn::Tensor> *v827 = &g_cached_result_main_const_eval_111;
  ttnn::constEvalFuncWrapper(v825, v826, v827);
  ::std::vector<::ttnn::Tensor> v828 = g_cached_result_main_const_eval_111;
  ::ttnn::Tensor v829 = v828[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v830 = &main_const_eval_112;
  ::std::vector<::ttnn::Tensor> v831 = util_create_vec(v193);
  ::std::vector<::ttnn::Tensor> *v832 = &g_cached_result_main_const_eval_112;
  ttnn::constEvalFuncWrapper(v830, v831, v832);
  ::std::vector<::ttnn::Tensor> v833 = g_cached_result_main_const_eval_112;
  ::ttnn::Tensor v834 = v833[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v835 = &main_const_eval_113;
  ::std::vector<::ttnn::Tensor> v836 = util_create_vec(v67);
  ::std::vector<::ttnn::Tensor> *v837 = &g_cached_result_main_const_eval_113;
  ttnn::constEvalFuncWrapper(v835, v836, v837);
  ::std::vector<::ttnn::Tensor> v838 = g_cached_result_main_const_eval_113;
  ::ttnn::Tensor v839 = v838[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v840 = &main_const_eval_114;
  ::std::vector<::ttnn::Tensor> v841 = util_create_vec(v267);
  ::std::vector<::ttnn::Tensor> *v842 = &g_cached_result_main_const_eval_114;
  ttnn::constEvalFuncWrapper(v840, v841, v842);
  ::std::vector<::ttnn::Tensor> v843 = g_cached_result_main_const_eval_114;
  ::ttnn::Tensor v844 = v843[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v845 = &main_const_eval_115;
  ::std::vector<::ttnn::Tensor> v846 = util_create_vec(v36);
  ::std::vector<::ttnn::Tensor> *v847 = &g_cached_result_main_const_eval_115;
  ttnn::constEvalFuncWrapper(v845, v846, v847);
  ::std::vector<::ttnn::Tensor> v848 = g_cached_result_main_const_eval_115;
  ::ttnn::Tensor v849 = v848[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v850 = &main_const_eval_116;
  ::std::vector<::ttnn::Tensor> v851 = util_create_vec(v30);
  ::std::vector<::ttnn::Tensor> *v852 = &g_cached_result_main_const_eval_116;
  ttnn::constEvalFuncWrapper(v850, v851, v852);
  ::std::vector<::ttnn::Tensor> v853 = g_cached_result_main_const_eval_116;
  ::ttnn::Tensor v854 = v853[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v855 = &main_const_eval_117;
  ::std::vector<::ttnn::Tensor> v856 = util_create_vec(v233);
  ::std::vector<::ttnn::Tensor> *v857 = &g_cached_result_main_const_eval_117;
  ttnn::constEvalFuncWrapper(v855, v856, v857);
  ::std::vector<::ttnn::Tensor> v858 = g_cached_result_main_const_eval_117;
  ::ttnn::Tensor v859 = v858[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v860 = &main_const_eval_118;
  ::std::vector<::ttnn::Tensor> v861 = util_create_vec(v257);
  ::std::vector<::ttnn::Tensor> *v862 = &g_cached_result_main_const_eval_118;
  ttnn::constEvalFuncWrapper(v860, v861, v862);
  ::std::vector<::ttnn::Tensor> v863 = g_cached_result_main_const_eval_118;
  ::ttnn::Tensor v864 = v863[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v865 = &main_const_eval_119;
  ::std::vector<::ttnn::Tensor> v866 = util_create_vec(v172);
  ::std::vector<::ttnn::Tensor> *v867 = &g_cached_result_main_const_eval_119;
  ttnn::constEvalFuncWrapper(v865, v866, v867);
  ::std::vector<::ttnn::Tensor> v868 = g_cached_result_main_const_eval_119;
  ::ttnn::Tensor v869 = v868[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v870 = &main_const_eval_120;
  ::std::vector<::ttnn::Tensor> v871 = util_create_vec(v201);
  ::std::vector<::ttnn::Tensor> *v872 = &g_cached_result_main_const_eval_120;
  ttnn::constEvalFuncWrapper(v870, v871, v872);
  ::std::vector<::ttnn::Tensor> v873 = g_cached_result_main_const_eval_120;
  ::ttnn::Tensor v874 = v873[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v875 = &main_const_eval_121;
  ::std::vector<::ttnn::Tensor> v876 = util_create_vec(v57);
  ::std::vector<::ttnn::Tensor> *v877 = &g_cached_result_main_const_eval_121;
  ttnn::constEvalFuncWrapper(v875, v876, v877);
  ::std::vector<::ttnn::Tensor> v878 = g_cached_result_main_const_eval_121;
  ::ttnn::Tensor v879 = v878[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v880 = &main_const_eval_122;
  ::std::vector<::ttnn::Tensor> v881 = util_create_vec(v53);
  ::std::vector<::ttnn::Tensor> *v882 = &g_cached_result_main_const_eval_122;
  ttnn::constEvalFuncWrapper(v880, v881, v882);
  ::std::vector<::ttnn::Tensor> v883 = g_cached_result_main_const_eval_122;
  ::ttnn::Tensor v884 = v883[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v885 = &main_const_eval_123;
  ::std::vector<::ttnn::Tensor> v886 = util_create_vec(v131);
  ::std::vector<::ttnn::Tensor> *v887 = &g_cached_result_main_const_eval_123;
  ttnn::constEvalFuncWrapper(v885, v886, v887);
  ::std::vector<::ttnn::Tensor> v888 = g_cached_result_main_const_eval_123;
  ::ttnn::Tensor v889 = v888[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v890 = &main_const_eval_124;
  ::std::vector<::ttnn::Tensor> v891 = util_create_vec(v9);
  ::std::vector<::ttnn::Tensor> *v892 = &g_cached_result_main_const_eval_124;
  ttnn::constEvalFuncWrapper(v890, v891, v892);
  ::std::vector<::ttnn::Tensor> v893 = g_cached_result_main_const_eval_124;
  ::ttnn::Tensor v894 = v893[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v895 = &main_const_eval_125;
  ::std::vector<::ttnn::Tensor> v896 = util_create_vec(v213);
  ::std::vector<::ttnn::Tensor> *v897 = &g_cached_result_main_const_eval_125;
  ttnn::constEvalFuncWrapper(v895, v896, v897);
  ::std::vector<::ttnn::Tensor> v898 = g_cached_result_main_const_eval_125;
  ::ttnn::Tensor v899 = v898[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v900 = &main_const_eval_126;
  ::std::vector<::ttnn::Tensor> v901 = util_create_vec(v102);
  ::std::vector<::ttnn::Tensor> *v902 = &g_cached_result_main_const_eval_126;
  ttnn::constEvalFuncWrapper(v900, v901, v902);
  ::std::vector<::ttnn::Tensor> v903 = g_cached_result_main_const_eval_126;
  ::ttnn::Tensor v904 = v903[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v905 = &main_const_eval_127;
  ::std::vector<::ttnn::Tensor> v906 = util_create_vec(v21);
  ::std::vector<::ttnn::Tensor> *v907 = &g_cached_result_main_const_eval_127;
  ttnn::constEvalFuncWrapper(v905, v906, v907);
  ::std::vector<::ttnn::Tensor> v908 = g_cached_result_main_const_eval_127;
  ::ttnn::Tensor v909 = v908[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v910 = &main_const_eval_128;
  ::std::vector<::ttnn::Tensor> v911 = util_create_vec(v156);
  ::std::vector<::ttnn::Tensor> *v912 = &g_cached_result_main_const_eval_128;
  ttnn::constEvalFuncWrapper(v910, v911, v912);
  ::std::vector<::ttnn::Tensor> v913 = g_cached_result_main_const_eval_128;
  ::ttnn::Tensor v914 = v913[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v915 = &main_const_eval_129;
  ::std::vector<::ttnn::Tensor> v916 = util_create_vec(v232);
  ::std::vector<::ttnn::Tensor> *v917 = &g_cached_result_main_const_eval_129;
  ttnn::constEvalFuncWrapper(v915, v916, v917);
  ::std::vector<::ttnn::Tensor> v918 = g_cached_result_main_const_eval_129;
  ::ttnn::Tensor v919 = v918[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v920 = &main_const_eval_130;
  ::std::vector<::ttnn::Tensor> v921 = util_create_vec(v150);
  ::std::vector<::ttnn::Tensor> *v922 = &g_cached_result_main_const_eval_130;
  ttnn::constEvalFuncWrapper(v920, v921, v922);
  ::std::vector<::ttnn::Tensor> v923 = g_cached_result_main_const_eval_130;
  ::ttnn::Tensor v924 = v923[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v925 = &main_const_eval_131;
  ::std::vector<::ttnn::Tensor> v926 = util_create_vec(v90);
  ::std::vector<::ttnn::Tensor> *v927 = &g_cached_result_main_const_eval_131;
  ttnn::constEvalFuncWrapper(v925, v926, v927);
  ::std::vector<::ttnn::Tensor> v928 = g_cached_result_main_const_eval_131;
  ::ttnn::Tensor v929 = v928[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v930 = &main_const_eval_132;
  ::std::vector<::ttnn::Tensor> v931 = util_create_vec(v112);
  ::std::vector<::ttnn::Tensor> *v932 = &g_cached_result_main_const_eval_132;
  ttnn::constEvalFuncWrapper(v930, v931, v932);
  ::std::vector<::ttnn::Tensor> v933 = g_cached_result_main_const_eval_132;
  ::ttnn::Tensor v934 = v933[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v935 = &main_const_eval_133;
  ::std::vector<::ttnn::Tensor> v936 = util_create_vec(v263);
  ::std::vector<::ttnn::Tensor> *v937 = &g_cached_result_main_const_eval_133;
  ttnn::constEvalFuncWrapper(v935, v936, v937);
  ::std::vector<::ttnn::Tensor> v938 = g_cached_result_main_const_eval_133;
  ::ttnn::Tensor v939 = v938[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v940 = &main_const_eval_134;
  ::std::vector<::ttnn::Tensor> v941 = util_create_vec(v142);
  ::std::vector<::ttnn::Tensor> *v942 = &g_cached_result_main_const_eval_134;
  ttnn::constEvalFuncWrapper(v940, v941, v942);
  ::std::vector<::ttnn::Tensor> v943 = g_cached_result_main_const_eval_134;
  ::ttnn::Tensor v944 = v943[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v945 = &main_const_eval_135;
  ::std::vector<::ttnn::Tensor> v946 = util_create_vec(v216);
  ::std::vector<::ttnn::Tensor> *v947 = &g_cached_result_main_const_eval_135;
  ttnn::constEvalFuncWrapper(v945, v946, v947);
  ::std::vector<::ttnn::Tensor> v948 = g_cached_result_main_const_eval_135;
  ::ttnn::Tensor v949 = v948[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v950 = &main_const_eval_136;
  ::std::vector<::ttnn::Tensor> v951 = util_create_vec(v40);
  ::std::vector<::ttnn::Tensor> *v952 = &g_cached_result_main_const_eval_136;
  ttnn::constEvalFuncWrapper(v950, v951, v952);
  ::std::vector<::ttnn::Tensor> v953 = g_cached_result_main_const_eval_136;
  ::ttnn::Tensor v954 = v953[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v955 = &main_const_eval_137;
  ::std::vector<::ttnn::Tensor> v956 = util_create_vec(v6);
  ::std::vector<::ttnn::Tensor> *v957 = &g_cached_result_main_const_eval_137;
  ttnn::constEvalFuncWrapper(v955, v956, v957);
  ::std::vector<::ttnn::Tensor> v958 = g_cached_result_main_const_eval_137;
  ::ttnn::Tensor v959 = v958[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v960 = &main_const_eval_138;
  ::std::vector<::ttnn::Tensor> v961 = util_create_vec(v46);
  ::std::vector<::ttnn::Tensor> *v962 = &g_cached_result_main_const_eval_138;
  ttnn::constEvalFuncWrapper(v960, v961, v962);
  ::std::vector<::ttnn::Tensor> v963 = g_cached_result_main_const_eval_138;
  ::ttnn::Tensor v964 = v963[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v965 = &main_const_eval_139;
  ::std::vector<::ttnn::Tensor> v966 = util_create_vec(v163);
  ::std::vector<::ttnn::Tensor> *v967 = &g_cached_result_main_const_eval_139;
  ttnn::constEvalFuncWrapper(v965, v966, v967);
  ::std::vector<::ttnn::Tensor> v968 = g_cached_result_main_const_eval_139;
  ::ttnn::Tensor v969 = v968[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v970 = &main_const_eval_140;
  ::std::vector<::ttnn::Tensor> v971 = util_create_vec(v127);
  ::std::vector<::ttnn::Tensor> *v972 = &g_cached_result_main_const_eval_140;
  ttnn::constEvalFuncWrapper(v970, v971, v972);
  ::std::vector<::ttnn::Tensor> v973 = g_cached_result_main_const_eval_140;
  ::ttnn::Tensor v974 = v973[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v975 = &main_const_eval_141;
  ::std::vector<::ttnn::Tensor> v976 = util_create_vec(v153);
  ::std::vector<::ttnn::Tensor> *v977 = &g_cached_result_main_const_eval_141;
  ttnn::constEvalFuncWrapper(v975, v976, v977);
  ::std::vector<::ttnn::Tensor> v978 = g_cached_result_main_const_eval_141;
  ::ttnn::Tensor v979 = v978[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v980 = &main_const_eval_142;
  ::std::vector<::ttnn::Tensor> v981 = util_create_vec(v212);
  ::std::vector<::ttnn::Tensor> *v982 = &g_cached_result_main_const_eval_142;
  ttnn::constEvalFuncWrapper(v980, v981, v982);
  ::std::vector<::ttnn::Tensor> v983 = g_cached_result_main_const_eval_142;
  ::ttnn::Tensor v984 = v983[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v985 = &main_const_eval_143;
  ::std::vector<::ttnn::Tensor> v986 = util_create_vec(v148);
  ::std::vector<::ttnn::Tensor> *v987 = &g_cached_result_main_const_eval_143;
  ttnn::constEvalFuncWrapper(v985, v986, v987);
  ::std::vector<::ttnn::Tensor> v988 = g_cached_result_main_const_eval_143;
  ::ttnn::Tensor v989 = v988[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v990 = &main_const_eval_144;
  ::std::vector<::ttnn::Tensor> v991 = util_create_vec(v120);
  ::std::vector<::ttnn::Tensor> *v992 = &g_cached_result_main_const_eval_144;
  ttnn::constEvalFuncWrapper(v990, v991, v992);
  ::std::vector<::ttnn::Tensor> v993 = g_cached_result_main_const_eval_144;
  ::ttnn::Tensor v994 = v993[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v995 = &main_const_eval_145;
  ::std::vector<::ttnn::Tensor> v996 = util_create_vec(v135);
  ::std::vector<::ttnn::Tensor> *v997 = &g_cached_result_main_const_eval_145;
  ttnn::constEvalFuncWrapper(v995, v996, v997);
  ::std::vector<::ttnn::Tensor> v998 = g_cached_result_main_const_eval_145;
  ::ttnn::Tensor v999 = v998[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1000 = &main_const_eval_146;
  ::std::vector<::ttnn::Tensor> v1001 = util_create_vec(v24);
  ::std::vector<::ttnn::Tensor> *v1002 = &g_cached_result_main_const_eval_146;
  ttnn::constEvalFuncWrapper(v1000, v1001, v1002);
  ::std::vector<::ttnn::Tensor> v1003 = g_cached_result_main_const_eval_146;
  ::ttnn::Tensor v1004 = v1003[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1005 = &main_const_eval_147;
  ::std::vector<::ttnn::Tensor> v1006 = util_create_vec(v226);
  ::std::vector<::ttnn::Tensor> *v1007 = &g_cached_result_main_const_eval_147;
  ttnn::constEvalFuncWrapper(v1005, v1006, v1007);
  ::std::vector<::ttnn::Tensor> v1008 = g_cached_result_main_const_eval_147;
  ::ttnn::Tensor v1009 = v1008[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1010 = &main_const_eval_148;
  ::std::vector<::ttnn::Tensor> v1011 = util_create_vec(v138);
  ::std::vector<::ttnn::Tensor> *v1012 = &g_cached_result_main_const_eval_148;
  ttnn::constEvalFuncWrapper(v1010, v1011, v1012);
  ::std::vector<::ttnn::Tensor> v1013 = g_cached_result_main_const_eval_148;
  ::ttnn::Tensor v1014 = v1013[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1015 = &main_const_eval_149;
  ::std::vector<::ttnn::Tensor> v1016 = util_create_vec(v116);
  ::std::vector<::ttnn::Tensor> *v1017 = &g_cached_result_main_const_eval_149;
  ttnn::constEvalFuncWrapper(v1015, v1016, v1017);
  ::std::vector<::ttnn::Tensor> v1018 = g_cached_result_main_const_eval_149;
  ::ttnn::Tensor v1019 = v1018[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1020 = &main_const_eval_150;
  ::std::vector<::ttnn::Tensor> v1021 = util_create_vec(v133);
  ::std::vector<::ttnn::Tensor> *v1022 = &g_cached_result_main_const_eval_150;
  ttnn::constEvalFuncWrapper(v1020, v1021, v1022);
  ::std::vector<::ttnn::Tensor> v1023 = g_cached_result_main_const_eval_150;
  ::ttnn::Tensor v1024 = v1023[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1025 = &main_const_eval_151;
  ::std::vector<::ttnn::Tensor> v1026 = util_create_vec(v246);
  ::std::vector<::ttnn::Tensor> *v1027 = &g_cached_result_main_const_eval_151;
  ttnn::constEvalFuncWrapper(v1025, v1026, v1027);
  ::std::vector<::ttnn::Tensor> v1028 = g_cached_result_main_const_eval_151;
  ::ttnn::Tensor v1029 = v1028[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1030 = &main_const_eval_152;
  ::std::vector<::ttnn::Tensor> v1031 = util_create_vec(v248);
  ::std::vector<::ttnn::Tensor> *v1032 = &g_cached_result_main_const_eval_152;
  ttnn::constEvalFuncWrapper(v1030, v1031, v1032);
  ::std::vector<::ttnn::Tensor> v1033 = g_cached_result_main_const_eval_152;
  ::ttnn::Tensor v1034 = v1033[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1035 = &main_const_eval_153;
  ::std::vector<::ttnn::Tensor> v1036 = util_create_vec(v242);
  ::std::vector<::ttnn::Tensor> *v1037 = &g_cached_result_main_const_eval_153;
  ttnn::constEvalFuncWrapper(v1035, v1036, v1037);
  ::std::vector<::ttnn::Tensor> v1038 = g_cached_result_main_const_eval_153;
  ::ttnn::Tensor v1039 = v1038[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1040 = &main_const_eval_154;
  ::std::vector<::ttnn::Tensor> v1041 = util_create_vec(v33);
  ::std::vector<::ttnn::Tensor> *v1042 = &g_cached_result_main_const_eval_154;
  ttnn::constEvalFuncWrapper(v1040, v1041, v1042);
  ::std::vector<::ttnn::Tensor> v1043 = g_cached_result_main_const_eval_154;
  ::ttnn::Tensor v1044 = v1043[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1045 = &main_const_eval_155;
  ::std::vector<::ttnn::Tensor> v1046 = util_create_vec(v123);
  ::std::vector<::ttnn::Tensor> *v1047 = &g_cached_result_main_const_eval_155;
  ttnn::constEvalFuncWrapper(v1045, v1046, v1047);
  ::std::vector<::ttnn::Tensor> v1048 = g_cached_result_main_const_eval_155;
  ::ttnn::Tensor v1049 = v1048[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1050 = &main_const_eval_156;
  ::std::vector<::ttnn::Tensor> v1051 = util_create_vec(v55);
  ::std::vector<::ttnn::Tensor> *v1052 = &g_cached_result_main_const_eval_156;
  ttnn::constEvalFuncWrapper(v1050, v1051, v1052);
  ::std::vector<::ttnn::Tensor> v1053 = g_cached_result_main_const_eval_156;
  ::ttnn::Tensor v1054 = v1053[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1055 = &main_const_eval_157;
  ::std::vector<::ttnn::Tensor> v1056 = util_create_vec(v140);
  ::std::vector<::ttnn::Tensor> *v1057 = &g_cached_result_main_const_eval_157;
  ttnn::constEvalFuncWrapper(v1055, v1056, v1057);
  ::std::vector<::ttnn::Tensor> v1058 = g_cached_result_main_const_eval_157;
  ::ttnn::Tensor v1059 = v1058[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1060 = &main_const_eval_158;
  ::std::vector<::ttnn::Tensor> v1061 = util_create_vec(v237);
  ::std::vector<::ttnn::Tensor> *v1062 = &g_cached_result_main_const_eval_158;
  ttnn::constEvalFuncWrapper(v1060, v1061, v1062);
  ::std::vector<::ttnn::Tensor> v1063 = g_cached_result_main_const_eval_158;
  ::ttnn::Tensor v1064 = v1063[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1065 = &main_const_eval_159;
  ::std::vector<::ttnn::Tensor> v1066 = util_create_vec(v52);
  ::std::vector<::ttnn::Tensor> *v1067 = &g_cached_result_main_const_eval_159;
  ttnn::constEvalFuncWrapper(v1065, v1066, v1067);
  ::std::vector<::ttnn::Tensor> v1068 = g_cached_result_main_const_eval_159;
  ::ttnn::Tensor v1069 = v1068[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1070 = &main_const_eval_160;
  ::std::vector<::ttnn::Tensor> v1071 = util_create_vec(v225);
  ::std::vector<::ttnn::Tensor> *v1072 = &g_cached_result_main_const_eval_160;
  ttnn::constEvalFuncWrapper(v1070, v1071, v1072);
  ::std::vector<::ttnn::Tensor> v1073 = g_cached_result_main_const_eval_160;
  ::ttnn::Tensor v1074 = v1073[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1075 = &main_const_eval_161;
  ::std::vector<::ttnn::Tensor> v1076 = util_create_vec(v161);
  ::std::vector<::ttnn::Tensor> *v1077 = &g_cached_result_main_const_eval_161;
  ttnn::constEvalFuncWrapper(v1075, v1076, v1077);
  ::std::vector<::ttnn::Tensor> v1078 = g_cached_result_main_const_eval_161;
  ::ttnn::Tensor v1079 = v1078[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1080 = &main_const_eval_162;
  ::std::vector<::ttnn::Tensor> v1081 = util_create_vec(v43);
  ::std::vector<::ttnn::Tensor> *v1082 = &g_cached_result_main_const_eval_162;
  ttnn::constEvalFuncWrapper(v1080, v1081, v1082);
  ::std::vector<::ttnn::Tensor> v1083 = g_cached_result_main_const_eval_162;
  ::ttnn::Tensor v1084 = v1083[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1085 = &main_const_eval_163;
  ::std::vector<::ttnn::Tensor> v1086 = util_create_vec(v137);
  ::std::vector<::ttnn::Tensor> *v1087 = &g_cached_result_main_const_eval_163;
  ttnn::constEvalFuncWrapper(v1085, v1086, v1087);
  ::std::vector<::ttnn::Tensor> v1088 = g_cached_result_main_const_eval_163;
  ::ttnn::Tensor v1089 = v1088[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1090 = &main_const_eval_164;
  ::std::vector<::ttnn::Tensor> v1091 = util_create_vec(v157);
  ::std::vector<::ttnn::Tensor> *v1092 = &g_cached_result_main_const_eval_164;
  ttnn::constEvalFuncWrapper(v1090, v1091, v1092);
  ::std::vector<::ttnn::Tensor> v1093 = g_cached_result_main_const_eval_164;
  ::ttnn::Tensor v1094 = v1093[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1095 = &main_const_eval_165;
  ::std::vector<::ttnn::Tensor> v1096 = util_create_vec(v188);
  ::std::vector<::ttnn::Tensor> *v1097 = &g_cached_result_main_const_eval_165;
  ttnn::constEvalFuncWrapper(v1095, v1096, v1097);
  ::std::vector<::ttnn::Tensor> v1098 = g_cached_result_main_const_eval_165;
  ::ttnn::Tensor v1099 = v1098[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1100 = &main_const_eval_166;
  ::std::vector<::ttnn::Tensor> v1101 = util_create_vec(v37);
  ::std::vector<::ttnn::Tensor> *v1102 = &g_cached_result_main_const_eval_166;
  ttnn::constEvalFuncWrapper(v1100, v1101, v1102);
  ::std::vector<::ttnn::Tensor> v1103 = g_cached_result_main_const_eval_166;
  ::ttnn::Tensor v1104 = v1103[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1105 = &main_const_eval_167;
  ::std::vector<::ttnn::Tensor> v1106 = util_create_vec(v60);
  ::std::vector<::ttnn::Tensor> *v1107 = &g_cached_result_main_const_eval_167;
  ttnn::constEvalFuncWrapper(v1105, v1106, v1107);
  ::std::vector<::ttnn::Tensor> v1108 = g_cached_result_main_const_eval_167;
  ::ttnn::Tensor v1109 = v1108[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1110 = &main_const_eval_168;
  ::std::vector<::ttnn::Tensor> v1111 = util_create_vec(v165);
  ::std::vector<::ttnn::Tensor> *v1112 = &g_cached_result_main_const_eval_168;
  ttnn::constEvalFuncWrapper(v1110, v1111, v1112);
  ::std::vector<::ttnn::Tensor> v1113 = g_cached_result_main_const_eval_168;
  ::ttnn::Tensor v1114 = v1113[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1115 = &main_const_eval_169;
  ::std::vector<::ttnn::Tensor> v1116 = util_create_vec(v170);
  ::std::vector<::ttnn::Tensor> *v1117 = &g_cached_result_main_const_eval_169;
  ttnn::constEvalFuncWrapper(v1115, v1116, v1117);
  ::std::vector<::ttnn::Tensor> v1118 = g_cached_result_main_const_eval_169;
  ::ttnn::Tensor v1119 = v1118[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1120 = &main_const_eval_170;
  ::std::vector<::ttnn::Tensor> v1121 = util_create_vec(v72);
  ::std::vector<::ttnn::Tensor> *v1122 = &g_cached_result_main_const_eval_170;
  ttnn::constEvalFuncWrapper(v1120, v1121, v1122);
  ::std::vector<::ttnn::Tensor> v1123 = g_cached_result_main_const_eval_170;
  ::ttnn::Tensor v1124 = v1123[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1125 = &main_const_eval_171;
  ::std::vector<::ttnn::Tensor> v1126 = util_create_vec(v253);
  ::std::vector<::ttnn::Tensor> *v1127 = &g_cached_result_main_const_eval_171;
  ttnn::constEvalFuncWrapper(v1125, v1126, v1127);
  ::std::vector<::ttnn::Tensor> v1128 = g_cached_result_main_const_eval_171;
  ::ttnn::Tensor v1129 = v1128[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1130 = &main_const_eval_172;
  ::std::vector<::ttnn::Tensor> v1131 = util_create_vec(v195);
  ::std::vector<::ttnn::Tensor> *v1132 = &g_cached_result_main_const_eval_172;
  ttnn::constEvalFuncWrapper(v1130, v1131, v1132);
  ::std::vector<::ttnn::Tensor> v1133 = g_cached_result_main_const_eval_172;
  ::ttnn::Tensor v1134 = v1133[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1135 = &main_const_eval_173;
  ::std::vector<::ttnn::Tensor> v1136 = util_create_vec(v26);
  ::std::vector<::ttnn::Tensor> *v1137 = &g_cached_result_main_const_eval_173;
  ttnn::constEvalFuncWrapper(v1135, v1136, v1137);
  ::std::vector<::ttnn::Tensor> v1138 = g_cached_result_main_const_eval_173;
  ::ttnn::Tensor v1139 = v1138[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1140 = &main_const_eval_174;
  ::std::vector<::ttnn::Tensor> v1141 = util_create_vec(v192);
  ::std::vector<::ttnn::Tensor> *v1142 = &g_cached_result_main_const_eval_174;
  ttnn::constEvalFuncWrapper(v1140, v1141, v1142);
  ::std::vector<::ttnn::Tensor> v1143 = g_cached_result_main_const_eval_174;
  ::ttnn::Tensor v1144 = v1143[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1145 = &main_const_eval_175;
  ::std::vector<::ttnn::Tensor> v1146 = util_create_vec(v11);
  ::std::vector<::ttnn::Tensor> *v1147 = &g_cached_result_main_const_eval_175;
  ttnn::constEvalFuncWrapper(v1145, v1146, v1147);
  ::std::vector<::ttnn::Tensor> v1148 = g_cached_result_main_const_eval_175;
  ::ttnn::Tensor v1149 = v1148[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1150 = &main_const_eval_176;
  ::std::vector<::ttnn::Tensor> v1151 = util_create_vec(v80);
  ::std::vector<::ttnn::Tensor> *v1152 = &g_cached_result_main_const_eval_176;
  ttnn::constEvalFuncWrapper(v1150, v1151, v1152);
  ::std::vector<::ttnn::Tensor> v1153 = g_cached_result_main_const_eval_176;
  ::ttnn::Tensor v1154 = v1153[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1155 = &main_const_eval_177;
  ::std::vector<::ttnn::Tensor> v1156 = util_create_vec(v207);
  ::std::vector<::ttnn::Tensor> *v1157 = &g_cached_result_main_const_eval_177;
  ttnn::constEvalFuncWrapper(v1155, v1156, v1157);
  ::std::vector<::ttnn::Tensor> v1158 = g_cached_result_main_const_eval_177;
  ::ttnn::Tensor v1159 = v1158[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1160 = &main_const_eval_178;
  ::std::vector<::ttnn::Tensor> v1161 = util_create_vec(v187);
  ::std::vector<::ttnn::Tensor> *v1162 = &g_cached_result_main_const_eval_178;
  ttnn::constEvalFuncWrapper(v1160, v1161, v1162);
  ::std::vector<::ttnn::Tensor> v1163 = g_cached_result_main_const_eval_178;
  ::ttnn::Tensor v1164 = v1163[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1165 = &main_const_eval_179;
  ::std::vector<::ttnn::Tensor> v1166 = util_create_vec(v128);
  ::std::vector<::ttnn::Tensor> *v1167 = &g_cached_result_main_const_eval_179;
  ttnn::constEvalFuncWrapper(v1165, v1166, v1167);
  ::std::vector<::ttnn::Tensor> v1168 = g_cached_result_main_const_eval_179;
  ::ttnn::Tensor v1169 = v1168[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1170 = &main_const_eval_180;
  ::std::vector<::ttnn::Tensor> v1171 = util_create_vec(v145);
  ::std::vector<::ttnn::Tensor> *v1172 = &g_cached_result_main_const_eval_180;
  ttnn::constEvalFuncWrapper(v1170, v1171, v1172);
  ::std::vector<::ttnn::Tensor> v1173 = g_cached_result_main_const_eval_180;
  ::ttnn::Tensor v1174 = v1173[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1175 = &main_const_eval_181;
  ::std::vector<::ttnn::Tensor> v1176 = util_create_vec(v206);
  ::std::vector<::ttnn::Tensor> *v1177 = &g_cached_result_main_const_eval_181;
  ttnn::constEvalFuncWrapper(v1175, v1176, v1177);
  ::std::vector<::ttnn::Tensor> v1178 = g_cached_result_main_const_eval_181;
  ::ttnn::Tensor v1179 = v1178[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1180 = &main_const_eval_182;
  ::std::vector<::ttnn::Tensor> v1181 = util_create_vec(v38);
  ::std::vector<::ttnn::Tensor> *v1182 = &g_cached_result_main_const_eval_182;
  ttnn::constEvalFuncWrapper(v1180, v1181, v1182);
  ::std::vector<::ttnn::Tensor> v1183 = g_cached_result_main_const_eval_182;
  ::ttnn::Tensor v1184 = v1183[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1185 = &main_const_eval_183;
  ::std::vector<::ttnn::Tensor> v1186 = util_create_vec(v15);
  ::std::vector<::ttnn::Tensor> *v1187 = &g_cached_result_main_const_eval_183;
  ttnn::constEvalFuncWrapper(v1185, v1186, v1187);
  ::std::vector<::ttnn::Tensor> v1188 = g_cached_result_main_const_eval_183;
  ::ttnn::Tensor v1189 = v1188[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1190 = &main_const_eval_184;
  ::std::vector<::ttnn::Tensor> v1191 = util_create_vec(v221);
  ::std::vector<::ttnn::Tensor> *v1192 = &g_cached_result_main_const_eval_184;
  ttnn::constEvalFuncWrapper(v1190, v1191, v1192);
  ::std::vector<::ttnn::Tensor> v1193 = g_cached_result_main_const_eval_184;
  ::ttnn::Tensor v1194 = v1193[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1195 = &main_const_eval_185;
  ::std::vector<::ttnn::Tensor> v1196 = util_create_vec(v76);
  ::std::vector<::ttnn::Tensor> *v1197 = &g_cached_result_main_const_eval_185;
  ttnn::constEvalFuncWrapper(v1195, v1196, v1197);
  ::std::vector<::ttnn::Tensor> v1198 = g_cached_result_main_const_eval_185;
  ::ttnn::Tensor v1199 = v1198[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1200 = &main_const_eval_186;
  ::std::vector<::ttnn::Tensor> v1201 = util_create_vec(v56);
  ::std::vector<::ttnn::Tensor> *v1202 = &g_cached_result_main_const_eval_186;
  ttnn::constEvalFuncWrapper(v1200, v1201, v1202);
  ::std::vector<::ttnn::Tensor> v1203 = g_cached_result_main_const_eval_186;
  ::ttnn::Tensor v1204 = v1203[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1205 = &main_const_eval_187;
  ::std::vector<::ttnn::Tensor> v1206 = util_create_vec(v266);
  ::std::vector<::ttnn::Tensor> *v1207 = &g_cached_result_main_const_eval_187;
  ttnn::constEvalFuncWrapper(v1205, v1206, v1207);
  ::std::vector<::ttnn::Tensor> v1208 = g_cached_result_main_const_eval_187;
  ::ttnn::Tensor v1209 = v1208[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1210 = &main_const_eval_188;
  ::std::vector<::ttnn::Tensor> v1211 = util_create_vec(v171);
  ::std::vector<::ttnn::Tensor> *v1212 = &g_cached_result_main_const_eval_188;
  ttnn::constEvalFuncWrapper(v1210, v1211, v1212);
  ::std::vector<::ttnn::Tensor> v1213 = g_cached_result_main_const_eval_188;
  ::ttnn::Tensor v1214 = v1213[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1215 = &main_const_eval_189;
  ::std::vector<::ttnn::Tensor> v1216 = util_create_vec(v122);
  ::std::vector<::ttnn::Tensor> *v1217 = &g_cached_result_main_const_eval_189;
  ttnn::constEvalFuncWrapper(v1215, v1216, v1217);
  ::std::vector<::ttnn::Tensor> v1218 = g_cached_result_main_const_eval_189;
  ::ttnn::Tensor v1219 = v1218[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1220 = &main_const_eval_190;
  ::std::vector<::ttnn::Tensor> v1221 = util_create_vec(v158);
  ::std::vector<::ttnn::Tensor> *v1222 = &g_cached_result_main_const_eval_190;
  ttnn::constEvalFuncWrapper(v1220, v1221, v1222);
  ::std::vector<::ttnn::Tensor> v1223 = g_cached_result_main_const_eval_190;
  ::ttnn::Tensor v1224 = v1223[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1225 = &main_const_eval_191;
  ::std::vector<::ttnn::Tensor> v1226 = util_create_vec(v247);
  ::std::vector<::ttnn::Tensor> *v1227 = &g_cached_result_main_const_eval_191;
  ttnn::constEvalFuncWrapper(v1225, v1226, v1227);
  ::std::vector<::ttnn::Tensor> v1228 = g_cached_result_main_const_eval_191;
  ::ttnn::Tensor v1229 = v1228[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1230 = &main_const_eval_192;
  ::std::vector<::ttnn::Tensor> v1231 = util_create_vec(v152);
  ::std::vector<::ttnn::Tensor> *v1232 = &g_cached_result_main_const_eval_192;
  ttnn::constEvalFuncWrapper(v1230, v1231, v1232);
  ::std::vector<::ttnn::Tensor> v1233 = g_cached_result_main_const_eval_192;
  ::ttnn::Tensor v1234 = v1233[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1235 = &main_const_eval_193;
  ::std::vector<::ttnn::Tensor> v1236 = util_create_vec(v252);
  ::std::vector<::ttnn::Tensor> *v1237 = &g_cached_result_main_const_eval_193;
  ttnn::constEvalFuncWrapper(v1235, v1236, v1237);
  ::std::vector<::ttnn::Tensor> v1238 = g_cached_result_main_const_eval_193;
  ::ttnn::Tensor v1239 = v1238[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1240 = &main_const_eval_194;
  ::std::vector<::ttnn::Tensor> v1241 = util_create_vec(v88);
  ::std::vector<::ttnn::Tensor> *v1242 = &g_cached_result_main_const_eval_194;
  ttnn::constEvalFuncWrapper(v1240, v1241, v1242);
  ::std::vector<::ttnn::Tensor> v1243 = g_cached_result_main_const_eval_194;
  ::ttnn::Tensor v1244 = v1243[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1245 = &main_const_eval_195;
  ::std::vector<::ttnn::Tensor> v1246 = util_create_vec(v177);
  ::std::vector<::ttnn::Tensor> *v1247 = &g_cached_result_main_const_eval_195;
  ttnn::constEvalFuncWrapper(v1245, v1246, v1247);
  ::std::vector<::ttnn::Tensor> v1248 = g_cached_result_main_const_eval_195;
  ::ttnn::Tensor v1249 = v1248[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1250 = &main_const_eval_196;
  ::std::vector<::ttnn::Tensor> v1251 = util_create_vec(v243);
  ::std::vector<::ttnn::Tensor> *v1252 = &g_cached_result_main_const_eval_196;
  ttnn::constEvalFuncWrapper(v1250, v1251, v1252);
  ::std::vector<::ttnn::Tensor> v1253 = g_cached_result_main_const_eval_196;
  ::ttnn::Tensor v1254 = v1253[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1255 = &main_const_eval_197;
  ::std::vector<::ttnn::Tensor> v1256 = util_create_vec(v168);
  ::std::vector<::ttnn::Tensor> *v1257 = &g_cached_result_main_const_eval_197;
  ttnn::constEvalFuncWrapper(v1255, v1256, v1257);
  ::std::vector<::ttnn::Tensor> v1258 = g_cached_result_main_const_eval_197;
  ::ttnn::Tensor v1259 = v1258[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1260 = &main_const_eval_198;
  ::std::vector<::ttnn::Tensor> v1261 = util_create_vec(v45);
  ::std::vector<::ttnn::Tensor> *v1262 = &g_cached_result_main_const_eval_198;
  ttnn::constEvalFuncWrapper(v1260, v1261, v1262);
  ::std::vector<::ttnn::Tensor> v1263 = g_cached_result_main_const_eval_198;
  ::ttnn::Tensor v1264 = v1263[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1265 = &main_const_eval_199;
  ::std::vector<::ttnn::Tensor> v1266 = util_create_vec(v106);
  ::std::vector<::ttnn::Tensor> *v1267 = &g_cached_result_main_const_eval_199;
  ttnn::constEvalFuncWrapper(v1265, v1266, v1267);
  ::std::vector<::ttnn::Tensor> v1268 = g_cached_result_main_const_eval_199;
  ::ttnn::Tensor v1269 = v1268[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1270 = &main_const_eval_200;
  ::std::vector<::ttnn::Tensor> v1271 = util_create_vec(v141);
  ::std::vector<::ttnn::Tensor> *v1272 = &g_cached_result_main_const_eval_200;
  ttnn::constEvalFuncWrapper(v1270, v1271, v1272);
  ::std::vector<::ttnn::Tensor> v1273 = g_cached_result_main_const_eval_200;
  ::ttnn::Tensor v1274 = v1273[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1275 = &main_const_eval_201;
  ::std::vector<::ttnn::Tensor> v1276 = util_create_vec(v260);
  ::std::vector<::ttnn::Tensor> *v1277 = &g_cached_result_main_const_eval_201;
  ttnn::constEvalFuncWrapper(v1275, v1276, v1277);
  ::std::vector<::ttnn::Tensor> v1278 = g_cached_result_main_const_eval_201;
  ::ttnn::Tensor v1279 = v1278[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1280 = &main_const_eval_202;
  ::std::vector<::ttnn::Tensor> v1281 = util_create_vec(v16);
  ::std::vector<::ttnn::Tensor> *v1282 = &g_cached_result_main_const_eval_202;
  ttnn::constEvalFuncWrapper(v1280, v1281, v1282);
  ::std::vector<::ttnn::Tensor> v1283 = g_cached_result_main_const_eval_202;
  ::ttnn::Tensor v1284 = v1283[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1285 = &main_const_eval_203;
  ::std::vector<::ttnn::Tensor> v1286 = util_create_vec(v236);
  ::std::vector<::ttnn::Tensor> *v1287 = &g_cached_result_main_const_eval_203;
  ttnn::constEvalFuncWrapper(v1285, v1286, v1287);
  ::std::vector<::ttnn::Tensor> v1288 = g_cached_result_main_const_eval_203;
  ::ttnn::Tensor v1289 = v1288[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1290 = &main_const_eval_204;
  ::std::vector<::ttnn::Tensor> v1291 = util_create_vec(v97);
  ::std::vector<::ttnn::Tensor> *v1292 = &g_cached_result_main_const_eval_204;
  ttnn::constEvalFuncWrapper(v1290, v1291, v1292);
  ::std::vector<::ttnn::Tensor> v1293 = g_cached_result_main_const_eval_204;
  ::ttnn::Tensor v1294 = v1293[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1295 = &main_const_eval_205;
  ::std::vector<::ttnn::Tensor> v1296 = util_create_vec(v32);
  ::std::vector<::ttnn::Tensor> *v1297 = &g_cached_result_main_const_eval_205;
  ttnn::constEvalFuncWrapper(v1295, v1296, v1297);
  ::std::vector<::ttnn::Tensor> v1298 = g_cached_result_main_const_eval_205;
  ::ttnn::Tensor v1299 = v1298[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1300 = &main_const_eval_206;
  ::std::vector<::ttnn::Tensor> v1301 = util_create_vec(v217);
  ::std::vector<::ttnn::Tensor> *v1302 = &g_cached_result_main_const_eval_206;
  ttnn::constEvalFuncWrapper(v1300, v1301, v1302);
  ::std::vector<::ttnn::Tensor> v1303 = g_cached_result_main_const_eval_206;
  ::ttnn::Tensor v1304 = v1303[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1305 = &main_const_eval_207;
  ::std::vector<::ttnn::Tensor> v1306 = util_create_vec(v132);
  ::std::vector<::ttnn::Tensor> *v1307 = &g_cached_result_main_const_eval_207;
  ttnn::constEvalFuncWrapper(v1305, v1306, v1307);
  ::std::vector<::ttnn::Tensor> v1308 = g_cached_result_main_const_eval_207;
  ::ttnn::Tensor v1309 = v1308[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1310 = &main_const_eval_208;
  ::std::vector<::ttnn::Tensor> v1311 = util_create_vec(v180);
  ::std::vector<::ttnn::Tensor> *v1312 = &g_cached_result_main_const_eval_208;
  ttnn::constEvalFuncWrapper(v1310, v1311, v1312);
  ::std::vector<::ttnn::Tensor> v1313 = g_cached_result_main_const_eval_208;
  ::ttnn::Tensor v1314 = v1313[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1315 = &main_const_eval_209;
  ::std::vector<::ttnn::Tensor> v1316 = util_create_vec(v42);
  ::std::vector<::ttnn::Tensor> *v1317 = &g_cached_result_main_const_eval_209;
  ttnn::constEvalFuncWrapper(v1315, v1316, v1317);
  ::std::vector<::ttnn::Tensor> v1318 = g_cached_result_main_const_eval_209;
  ::ttnn::Tensor v1319 = v1318[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1320 = &main_const_eval_210;
  ::std::vector<::ttnn::Tensor> v1321 = util_create_vec(v115);
  ::std::vector<::ttnn::Tensor> *v1322 = &g_cached_result_main_const_eval_210;
  ttnn::constEvalFuncWrapper(v1320, v1321, v1322);
  ::std::vector<::ttnn::Tensor> v1323 = g_cached_result_main_const_eval_210;
  ::ttnn::Tensor v1324 = v1323[0];
  ::std::function<::std::vector<::ttnn::Tensor>(::std::vector<::ttnn::Tensor>)>
      v1325 = &main_const_eval_211;
  ::std::vector<::ttnn::Tensor> v1326 = util_create_vec(v203);
  ::std::vector<::ttnn::Tensor> *v1327 = &g_cached_result_main_const_eval_211;
  ttnn::constEvalFuncWrapper(v1325, v1326, v1327);
  ::std::vector<::ttnn::Tensor> v1328 = g_cached_result_main_const_eval_211;
  ::ttnn::Tensor v1329 = v1328[0];
  ttnn::distributed::MeshDevice *v1330 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v1331 = ttnn::to_layout(
      v29, ::ttnn::Layout::TILE, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v29, false);
  ::ttnn::Tensor v1332 = ttnn::permute(
      v1331, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1331, false);
  ::ttnn::Tensor v1333 = ttnn::reshape(
      v1332, ::std::vector<int32_t>{1, 1, 50176, 3},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1332, false);
  ::ttnn::Tensor v1334 = ttnn::to_layout(
      v1333, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1333, false);
  ::ttnn::Tensor v1335 = ::std::get<0>(ttnn::conv2d(
      v1334, v28, v1330, 3, 64, 1, 224, 224, ::std::array<uint32_t, 2>{7, 7},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{3, 3, 3, 3},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1334, false);
  ::ttnn::Tensor v1336 = ttnn::reshape(
      v1335, ::std::vector<int32_t>{1, 112, 112, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1335, false);
  ::ttnn::Tensor v1337 = ttnn::permute(
      v1336, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1336, false);
  ::ttnn::Tensor v1338 = ttnn::batch_norm(
      v1337, v274, v1004, false, 0.000010f, 0.100000f, v684, v1139,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1337, false);
  ::ttnn::Tensor v1339 = ttnn::relu(
      v1338, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1338, false);
  ::ttnn::Tensor v1340 = ttnn::permute(
      v1339, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1339, false);
  ::ttnn::Tensor v1341 = ttnn::reshape(
      v1340, ::std::vector<int32_t>{1, 1, 12544, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1340, false);
  ::ttnn::Tensor v1342 = ttnn::typecast(
      v1341, ::ttnn::DataType::BFLOAT16,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1341, false);
  ::ttnn::Tensor v1343 = ttnn::to_layout(
      v1342, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1342, false);
  ::std::vector<::ttnn::Tensor> v1344 = ttnn::max_pool2d(
      v1343, 1, 112, 112, 64, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, false,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt, ::std::nullopt, false, false, false,
      ::ttnn::DataType::BFLOAT16, ::ttnn::Layout::ROW_MAJOR, true);
  ::ttnn::Tensor v1345 = v1344[0];
  ttnn::deallocate(v1343, false);
  ::ttnn::Tensor v1346 = ttnn::typecast(
      v1345, ::ttnn::DataType::FLOAT32,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1345, false);
  ::ttnn::Tensor v1347 = ::std::get<0>(ttnn::conv2d(
      v1346, v44, v1330, 64, 64, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ::ttnn::Tensor v1348 = ttnn::reshape(
      v1347, ::std::vector<int32_t>{1, 56, 56, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1347, false);
  ::ttnn::Tensor v1349 = ttnn::permute(
      v1348, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1348, false);
  ::ttnn::Tensor v1350 = ttnn::batch_norm(
      v1349, v734, v954, false, 0.000010f, 0.100000f, v1084, v1319,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1349, false);
  ::ttnn::Tensor v1351 = ttnn::relu(
      v1350, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1350, false);
  ::ttnn::Tensor v1352 = ttnn::permute(
      v1351, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1351, false);
  ::ttnn::Tensor v1353 = ttnn::reshape(
      v1352, ::std::vector<int32_t>{1, 1, 3136, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1352, false);
  ::ttnn::Tensor v1354 = ttnn::to_layout(
      v1353, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1353, false);
  ::ttnn::Tensor v1355 = ::std::get<0>(ttnn::conv2d(
      v1354, v39, v1330, 64, 64, 1, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1354, false);
  ::ttnn::Tensor v1356 = ttnn::reshape(
      v1355, ::std::vector<int32_t>{1, 56, 56, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1355, false);
  ::ttnn::Tensor v1357 = ttnn::permute(
      v1356, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1356, false);
  ::ttnn::Tensor v1358 = ttnn::batch_norm(
      v1357, v849, v364, false, 0.000010f, 0.100000f, v1184, v1104,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1357, false);
  ::ttnn::Tensor v1359 = ttnn::relu(
      v1358, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1358, false);
  ::ttnn::Tensor v1360 = ttnn::permute(
      v1359, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1359, false);
  ::ttnn::Tensor v1361 = ttnn::reshape(
      v1360, ::std::vector<int32_t>{1, 1, 3136, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1360, false);
  ::ttnn::Tensor v1362 = ttnn::to_layout(
      v1361, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1361, false);
  ::ttnn::Tensor v1363 = ::std::get<0>(ttnn::conv2d(
      v1362, v34, v1330, 64, 256, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1362, false);
  ::ttnn::Tensor v1364 = ttnn::reshape(
      v1363, ::std::vector<int32_t>{1, 56, 56, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1363, false);
  ::ttnn::Tensor v1365 = ttnn::permute(
      v1364, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1364, false);
  ::ttnn::Tensor v1366 = ttnn::batch_norm(
      v1365, v569, v854, false, 0.000010f, 0.100000f, v1044, v1299,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1365, false);
  ::ttnn::Tensor v1367 = ::std::get<0>(ttnn::conv2d(
      v1346, v23, v1330, 64, 256, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1346, false);
  ::ttnn::Tensor v1368 = ttnn::reshape(
      v1367, ::std::vector<int32_t>{1, 56, 56, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1367, false);
  ::ttnn::Tensor v1369 = ttnn::permute(
      v1368, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1368, false);
  ::ttnn::Tensor v1370 = ttnn::batch_norm(
      v1369, v384, v434, false, 0.000010f, 0.100000f, v544, v909,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1369, false);
  ::ttnn::Tensor v1371 =
      ttnn::add(v1366, v1370, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1370, false);
  ttnn::deallocate(v1366, false);
  ::ttnn::Tensor v1372 = ttnn::relu(
      v1371, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1371, false);
  ::ttnn::Tensor v1373 = ttnn::permute(
      v1372, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1374 = ttnn::reshape(
      v1373, ::std::vector<int32_t>{1, 1, 3136, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1373, false);
  ::ttnn::Tensor v1375 = ttnn::to_layout(
      v1374, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1374, false);
  ::ttnn::Tensor v1376 = ::std::get<0>(ttnn::conv2d(
      v1375, v59, v1330, 256, 64, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1375, false);
  ::ttnn::Tensor v1377 = ttnn::reshape(
      v1376, ::std::vector<int32_t>{1, 56, 56, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1376, false);
  ::ttnn::Tensor v1378 = ttnn::permute(
      v1377, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1377, false);
  ::ttnn::Tensor v1379 = ttnn::batch_norm(
      v1378, v1204, v1054, false, 0.000010f, 0.100000f, v284, v879,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1378, false);
  ::ttnn::Tensor v1380 = ttnn::relu(
      v1379, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1379, false);
  ::ttnn::Tensor v1381 = ttnn::permute(
      v1380, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1380, false);
  ::ttnn::Tensor v1382 = ttnn::reshape(
      v1381, ::std::vector<int32_t>{1, 1, 3136, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1381, false);
  ::ttnn::Tensor v1383 = ttnn::to_layout(
      v1382, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1382, false);
  ::ttnn::Tensor v1384 = ::std::get<0>(ttnn::conv2d(
      v1383, v54, v1330, 64, 64, 1, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1383, false);
  ::ttnn::Tensor v1385 = ttnn::reshape(
      v1384, ::std::vector<int32_t>{1, 56, 56, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1384, false);
  ::ttnn::Tensor v1386 = ttnn::permute(
      v1385, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1385, false);
  ::ttnn::Tensor v1387 = ttnn::batch_norm(
      v1386, v449, v744, false, 0.000010f, 0.100000f, v884, v1069,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1386, false);
  ::ttnn::Tensor v1388 = ttnn::relu(
      v1387, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1387, false);
  ::ttnn::Tensor v1389 = ttnn::permute(
      v1388, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1388, false);
  ::ttnn::Tensor v1390 = ttnn::reshape(
      v1389, ::std::vector<int32_t>{1, 1, 3136, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1389, false);
  ::ttnn::Tensor v1391 = ttnn::to_layout(
      v1390, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1390, false);
  ::ttnn::Tensor v1392 = ::std::get<0>(ttnn::conv2d(
      v1391, v49, v1330, 64, 256, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1391, false);
  ::ttnn::Tensor v1393 = ttnn::reshape(
      v1392, ::std::vector<int32_t>{1, 56, 56, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1392, false);
  ::ttnn::Tensor v1394 = ttnn::permute(
      v1393, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1393, false);
  ::ttnn::Tensor v1395 = ttnn::batch_norm(
      v1394, v964, v1264, false, 0.000010f, 0.100000f, v824, v404,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1394, false);
  ::ttnn::Tensor v1396 =
      ttnn::add(v1395, v1372, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1395, false);
  ttnn::deallocate(v1372, false);
  ::ttnn::Tensor v1397 = ttnn::relu(
      v1396, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1396, false);
  ::ttnn::Tensor v1398 = ttnn::permute(
      v1397, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1399 = ttnn::reshape(
      v1398, ::std::vector<int32_t>{1, 1, 3136, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1398, false);
  ::ttnn::Tensor v1400 = ttnn::to_layout(
      v1399, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1399, false);
  ::ttnn::Tensor v1401 = ::std::get<0>(ttnn::conv2d(
      v1400, v74, v1330, 256, 64, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1400, false);
  ::ttnn::Tensor v1402 = ttnn::reshape(
      v1401, ::std::vector<int32_t>{1, 56, 56, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1401, false);
  ::ttnn::Tensor v1403 = ttnn::permute(
      v1402, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1402, false);
  ::ttnn::Tensor v1404 = ttnn::batch_norm(
      v1403, v474, v794, false, 0.000010f, 0.100000f, v484, v1124,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1403, false);
  ::ttnn::Tensor v1405 = ttnn::relu(
      v1404, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1404, false);
  ::ttnn::Tensor v1406 = ttnn::permute(
      v1405, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1405, false);
  ::ttnn::Tensor v1407 = ttnn::reshape(
      v1406, ::std::vector<int32_t>{1, 1, 3136, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1406, false);
  ::ttnn::Tensor v1408 = ttnn::to_layout(
      v1407, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1407, false);
  ::ttnn::Tensor v1409 = ::std::get<0>(ttnn::conv2d(
      v1408, v69, v1330, 64, 64, 1, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1408, false);
  ::ttnn::Tensor v1410 = ttnn::reshape(
      v1409, ::std::vector<int32_t>{1, 56, 56, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1409, false);
  ::ttnn::Tensor v1411 = ttnn::permute(
      v1410, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1410, false);
  ::ttnn::Tensor v1412 = ttnn::batch_norm(
      v1411, v314, v389, false, 0.000010f, 0.100000f, v634, v839,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1411, false);
  ::ttnn::Tensor v1413 = ttnn::relu(
      v1412, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1412, false);
  ::ttnn::Tensor v1414 = ttnn::permute(
      v1413, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1413, false);
  ::ttnn::Tensor v1415 = ttnn::reshape(
      v1414, ::std::vector<int32_t>{1, 1, 3136, 64},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1414, false);
  ::ttnn::Tensor v1416 = ttnn::to_layout(
      v1415, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1415, false);
  ::ttnn::Tensor v1417 = ::std::get<0>(ttnn::conv2d(
      v1416, v64, v1330, 64, 256, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1416, false);
  ::ttnn::Tensor v1418 = ttnn::reshape(
      v1417, ::std::vector<int32_t>{1, 56, 56, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1417, false);
  ::ttnn::Tensor v1419 = ttnn::permute(
      v1418, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1418, false);
  ::ttnn::Tensor v1420 = ttnn::batch_norm(
      v1419, v639, v1109, false, 0.000010f, 0.100000f, v424, v779,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1419, false);
  ::ttnn::Tensor v1421 =
      ttnn::add(v1420, v1397, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1420, false);
  ttnn::deallocate(v1397, false);
  ::ttnn::Tensor v1422 = ttnn::relu(
      v1421, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1421, false);
  ::ttnn::Tensor v1423 = ttnn::permute(
      v1422, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1422, false);
  ::ttnn::Tensor v1424 = ttnn::reshape(
      v1423, ::std::vector<int32_t>{1, 1, 3136, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1423, false);
  ::ttnn::Tensor v1425 = ttnn::to_layout(
      v1424, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1424, false);
  ::ttnn::Tensor v1426 = ::std::get<0>(ttnn::conv2d(
      v1425, v89, v1330, 256, 128, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ::ttnn::Tensor v1427 = ttnn::reshape(
      v1426, ::std::vector<int32_t>{1, 56, 56, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1426, false);
  ::ttnn::Tensor v1428 = ttnn::permute(
      v1427, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1427, false);
  ::ttnn::Tensor v1429 = ttnn::batch_norm(
      v1428, v754, v289, false, 0.000010f, 0.100000f, v1244, v419,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1428, false);
  ::ttnn::Tensor v1430 = ttnn::relu(
      v1429, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1429, false);
  ::ttnn::Tensor v1431 = ttnn::permute(
      v1430, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1430, false);
  ::ttnn::Tensor v1432 = ttnn::reshape(
      v1431, ::std::vector<int32_t>{1, 1, 3136, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1431, false);
  ::ttnn::Tensor v1433 = ttnn::to_layout(
      v1432, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1432, false);
  ::ttnn::Tensor v1434 = ::std::get<0>(ttnn::conv2d(
      v1433, v84, v1330, 128, 128, 1, 56, 56, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1433, false);
  ::ttnn::Tensor v1435 = ttnn::reshape(
      v1434, ::std::vector<int32_t>{1, 28, 28, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1434, false);
  ::ttnn::Tensor v1436 = ttnn::permute(
      v1435, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1435, false);
  ::ttnn::Tensor v1437 = ttnn::batch_norm(
      v1436, v439, v1154, false, 0.000010f, 0.100000f, v664, v589,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1436, false);
  ::ttnn::Tensor v1438 = ttnn::relu(
      v1437, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1437, false);
  ::ttnn::Tensor v1439 = ttnn::permute(
      v1438, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1438, false);
  ::ttnn::Tensor v1440 = ttnn::reshape(
      v1439, ::std::vector<int32_t>{1, 1, 784, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1439, false);
  ::ttnn::Tensor v1441 = ttnn::to_layout(
      v1440, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1440, false);
  ::ttnn::Tensor v1442 = ::std::get<0>(ttnn::conv2d(
      v1441, v79, v1330, 128, 512, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1441, false);
  ::ttnn::Tensor v1443 = ttnn::reshape(
      v1442, ::std::vector<int32_t>{1, 28, 28, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1442, false);
  ::ttnn::Tensor v1444 = ttnn::permute(
      v1443, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1443, false);
  ::ttnn::Tensor v1445 = ttnn::batch_norm(
      v1444, v1199, v454, false, 0.000010f, 0.100000f, v829, v444,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1444, false);
  ::ttnn::Tensor v1446 = ::std::get<0>(ttnn::conv2d(
      v1425, v18, v1330, 256, 512, 1, 56, 56, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1425, false);
  ::ttnn::Tensor v1447 = ttnn::reshape(
      v1446, ::std::vector<int32_t>{1, 28, 28, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1446, false);
  ::ttnn::Tensor v1448 = ttnn::permute(
      v1447, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1447, false);
  ::ttnn::Tensor v1449 = ttnn::batch_norm(
      v1448, v1189, v609, false, 0.000010f, 0.100000f, v579, v1284,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1448, false);
  ::ttnn::Tensor v1450 =
      ttnn::add(v1445, v1449, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1449, false);
  ttnn::deallocate(v1445, false);
  ::ttnn::Tensor v1451 = ttnn::relu(
      v1450, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1450, false);
  ::ttnn::Tensor v1452 = ttnn::permute(
      v1451, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1453 = ttnn::reshape(
      v1452, ::std::vector<int32_t>{1, 1, 784, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1452, false);
  ::ttnn::Tensor v1454 = ttnn::to_layout(
      v1453, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1453, false);
  ::ttnn::Tensor v1455 = ::std::get<0>(ttnn::conv2d(
      v1454, v104, v1330, 512, 128, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1454, false);
  ::ttnn::Tensor v1456 = ttnn::reshape(
      v1455, ::std::vector<int32_t>{1, 28, 28, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1455, false);
  ::ttnn::Tensor v1457 = ttnn::permute(
      v1456, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1456, false);
  ::ttnn::Tensor v1458 = ttnn::batch_norm(
      v1457, v319, v514, false, 0.000010f, 0.100000f, v764, v904,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1457, false);
  ::ttnn::Tensor v1459 = ttnn::relu(
      v1458, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1458, false);
  ::ttnn::Tensor v1460 = ttnn::permute(
      v1459, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1459, false);
  ::ttnn::Tensor v1461 = ttnn::reshape(
      v1460, ::std::vector<int32_t>{1, 1, 784, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1460, false);
  ::ttnn::Tensor v1462 = ttnn::to_layout(
      v1461, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1461, false);
  ::ttnn::Tensor v1463 = ::std::get<0>(ttnn::conv2d(
      v1462, v99, v1330, 128, 128, 1, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1462, false);
  ::ttnn::Tensor v1464 = ttnn::reshape(
      v1463, ::std::vector<int32_t>{1, 28, 28, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1463, false);
  ::ttnn::Tensor v1465 = ttnn::permute(
      v1464, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1464, false);
  ::ttnn::Tensor v1466 = ttnn::batch_norm(
      v1465, v504, v564, false, 0.000010f, 0.100000f, v704, v1294,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1465, false);
  ::ttnn::Tensor v1467 = ttnn::relu(
      v1466, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1466, false);
  ::ttnn::Tensor v1468 = ttnn::permute(
      v1467, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1467, false);
  ::ttnn::Tensor v1469 = ttnn::reshape(
      v1468, ::std::vector<int32_t>{1, 1, 784, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1468, false);
  ::ttnn::Tensor v1470 = ttnn::to_layout(
      v1469, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1469, false);
  ::ttnn::Tensor v1471 = ::std::get<0>(ttnn::conv2d(
      v1470, v94, v1330, 128, 512, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1470, false);
  ::ttnn::Tensor v1472 = ttnn::reshape(
      v1471, ::std::vector<int32_t>{1, 28, 28, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1471, false);
  ::ttnn::Tensor v1473 = ttnn::permute(
      v1472, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1472, false);
  ::ttnn::Tensor v1474 = ttnn::batch_norm(
      v1473, v689, v929, false, 0.000010f, 0.100000f, v299, v719,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1473, false);
  ::ttnn::Tensor v1475 =
      ttnn::add(v1474, v1451, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1474, false);
  ttnn::deallocate(v1451, false);
  ::ttnn::Tensor v1476 = ttnn::relu(
      v1475, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1475, false);
  ::ttnn::Tensor v1477 = ttnn::permute(
      v1476, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1478 = ttnn::reshape(
      v1477, ::std::vector<int32_t>{1, 1, 784, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1477, false);
  ::ttnn::Tensor v1479 = ttnn::to_layout(
      v1478, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1478, false);
  ::ttnn::Tensor v1480 = ::std::get<0>(ttnn::conv2d(
      v1479, v119, v1330, 512, 128, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1479, false);
  ::ttnn::Tensor v1481 = ttnn::reshape(
      v1480, ::std::vector<int32_t>{1, 28, 28, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1480, false);
  ::ttnn::Tensor v1482 = ttnn::permute(
      v1481, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1481, false);
  ::ttnn::Tensor v1483 = ttnn::batch_norm(
      v1482, v1019, v1324, false, 0.000010f, 0.100000f, v489, v729,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1482, false);
  ::ttnn::Tensor v1484 = ttnn::relu(
      v1483, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1483, false);
  ::ttnn::Tensor v1485 = ttnn::permute(
      v1484, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1484, false);
  ::ttnn::Tensor v1486 = ttnn::reshape(
      v1485, ::std::vector<int32_t>{1, 1, 784, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1485, false);
  ::ttnn::Tensor v1487 = ttnn::to_layout(
      v1486, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1486, false);
  ::ttnn::Tensor v1488 = ::std::get<0>(ttnn::conv2d(
      v1487, v114, v1330, 128, 128, 1, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1487, false);
  ::ttnn::Tensor v1489 = ttnn::reshape(
      v1488, ::std::vector<int32_t>{1, 28, 28, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1488, false);
  ::ttnn::Tensor v1490 = ttnn::permute(
      v1489, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1489, false);
  ::ttnn::Tensor v1491 = ttnn::batch_norm(
      v1490, v354, v359, false, 0.000010f, 0.100000f, v814, v934,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1490, false);
  ::ttnn::Tensor v1492 = ttnn::relu(
      v1491, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1491, false);
  ::ttnn::Tensor v1493 = ttnn::permute(
      v1492, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1492, false);
  ::ttnn::Tensor v1494 = ttnn::reshape(
      v1493, ::std::vector<int32_t>{1, 1, 784, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1493, false);
  ::ttnn::Tensor v1495 = ttnn::to_layout(
      v1494, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1494, false);
  ::ttnn::Tensor v1496 = ::std::get<0>(ttnn::conv2d(
      v1495, v109, v1330, 128, 512, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1495, false);
  ::ttnn::Tensor v1497 = ttnn::reshape(
      v1496, ::std::vector<int32_t>{1, 28, 28, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1496, false);
  ::ttnn::Tensor v1498 = ttnn::permute(
      v1497, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1497, false);
  ::ttnn::Tensor v1499 = ttnn::batch_norm(
      v1498, v1269, v534, false, 0.000010f, 0.100000f, v599, v399,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1498, false);
  ::ttnn::Tensor v1500 =
      ttnn::add(v1499, v1476, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1499, false);
  ttnn::deallocate(v1476, false);
  ::ttnn::Tensor v1501 = ttnn::relu(
      v1500, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1500, false);
  ::ttnn::Tensor v1502 = ttnn::permute(
      v1501, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1503 = ttnn::reshape(
      v1502, ::std::vector<int32_t>{1, 1, 784, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1502, false);
  ::ttnn::Tensor v1504 = ttnn::to_layout(
      v1503, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1503, false);
  ::ttnn::Tensor v1505 = ::std::get<0>(ttnn::conv2d(
      v1504, v134, v1330, 512, 128, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1504, false);
  ::ttnn::Tensor v1506 = ttnn::reshape(
      v1505, ::std::vector<int32_t>{1, 28, 28, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1505, false);
  ::ttnn::Tensor v1507 = ttnn::permute(
      v1506, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1506, false);
  ::ttnn::Tensor v1508 = ttnn::batch_norm(
      v1507, v889, v324, false, 0.000010f, 0.100000f, v1024, v1309,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1507, false);
  ::ttnn::Tensor v1509 = ttnn::relu(
      v1508, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1508, false);
  ::ttnn::Tensor v1510 = ttnn::permute(
      v1509, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1509, false);
  ::ttnn::Tensor v1511 = ttnn::reshape(
      v1510, ::std::vector<int32_t>{1, 1, 784, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1510, false);
  ::ttnn::Tensor v1512 = ttnn::to_layout(
      v1511, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1511, false);
  ::ttnn::Tensor v1513 = ::std::get<0>(ttnn::conv2d(
      v1512, v129, v1330, 128, 128, 1, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1512, false);
  ::ttnn::Tensor v1514 = ttnn::reshape(
      v1513, ::std::vector<int32_t>{1, 28, 28, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1513, false);
  ::ttnn::Tensor v1515 = ttnn::permute(
      v1514, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1514, false);
  ::ttnn::Tensor v1516 = ttnn::batch_norm(
      v1515, v584, v559, false, 0.000010f, 0.100000f, v1169, v974,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1515, false);
  ::ttnn::Tensor v1517 = ttnn::relu(
      v1516, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1516, false);
  ::ttnn::Tensor v1518 = ttnn::permute(
      v1517, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1517, false);
  ::ttnn::Tensor v1519 = ttnn::reshape(
      v1518, ::std::vector<int32_t>{1, 1, 784, 128},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1518, false);
  ::ttnn::Tensor v1520 = ttnn::to_layout(
      v1519, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1519, false);
  ::ttnn::Tensor v1521 = ::std::get<0>(ttnn::conv2d(
      v1520, v124, v1330, 128, 512, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1520, false);
  ::ttnn::Tensor v1522 = ttnn::reshape(
      v1521, ::std::vector<int32_t>{1, 28, 28, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1521, false);
  ::ttnn::Tensor v1523 = ttnn::permute(
      v1522, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1522, false);
  ::ttnn::Tensor v1524 = ttnn::batch_norm(
      v1523, v309, v994, false, 0.000010f, 0.100000f, v1049, v1219,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1523, false);
  ::ttnn::Tensor v1525 =
      ttnn::add(v1524, v1501, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1524, false);
  ttnn::deallocate(v1501, false);
  ::ttnn::Tensor v1526 = ttnn::relu(
      v1525, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1525, false);
  ::ttnn::Tensor v1527 = ttnn::permute(
      v1526, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1526, false);
  ::ttnn::Tensor v1528 = ttnn::reshape(
      v1527, ::std::vector<int32_t>{1, 1, 784, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1527, false);
  ::ttnn::Tensor v1529 = ttnn::to_layout(
      v1528, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1528, false);
  ::ttnn::Tensor v1530 = ::std::get<0>(ttnn::conv2d(
      v1529, v149, v1330, 512, 256, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ::ttnn::Tensor v1531 = ttnn::reshape(
      v1530, ::std::vector<int32_t>{1, 28, 28, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1530, false);
  ::ttnn::Tensor v1532 = ttnn::permute(
      v1531, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1531, false);
  ::ttnn::Tensor v1533 = ttnn::batch_norm(
      v1532, v644, v1174, false, 0.000010f, 0.100000f, v989, v774,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1532, false);
  ::ttnn::Tensor v1534 = ttnn::relu(
      v1533, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1533, false);
  ::ttnn::Tensor v1535 = ttnn::permute(
      v1534, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1534, false);
  ::ttnn::Tensor v1536 = ttnn::reshape(
      v1535, ::std::vector<int32_t>{1, 1, 784, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1535, false);
  ::ttnn::Tensor v1537 = ttnn::to_layout(
      v1536, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1536, false);
  ::ttnn::Tensor v1538 = ::std::get<0>(ttnn::conv2d(
      v1537, v144, v1330, 256, 256, 1, 28, 28, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1537, false);
  ::ttnn::Tensor v1539 = ttnn::reshape(
      v1538, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1538, false);
  ::ttnn::Tensor v1540 = ttnn::permute(
      v1539, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1539, false);
  ::ttnn::Tensor v1541 = ttnn::batch_norm(
      v1540, v1274, v1059, false, 0.000010f, 0.100000f, v594, v944,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1540, false);
  ::ttnn::Tensor v1542 = ttnn::relu(
      v1541, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1541, false);
  ::ttnn::Tensor v1543 = ttnn::permute(
      v1542, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1542, false);
  ::ttnn::Tensor v1544 = ttnn::reshape(
      v1543, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1543, false);
  ::ttnn::Tensor v1545 = ttnn::to_layout(
      v1544, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1544, false);
  ::ttnn::Tensor v1546 = ::std::get<0>(ttnn::conv2d(
      v1545, v139, v1330, 256, 1024, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1545, false);
  ::ttnn::Tensor v1547 = ttnn::reshape(
      v1546, ::std::vector<int32_t>{1, 14, 14, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1546, false);
  ::ttnn::Tensor v1548 = ttnn::permute(
      v1547, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1547, false);
  ::ttnn::Tensor v1549 = ttnn::batch_norm(
      v1548, v674, v999, false, 0.000010f, 0.100000f, v1014, v1089,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1548, false);
  ::ttnn::Tensor v1550 = ::std::get<0>(ttnn::conv2d(
      v1529, v13, v1330, 512, 1024, 1, 28, 28, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1529, false);
  ::ttnn::Tensor v1551 = ttnn::reshape(
      v1550, ::std::vector<int32_t>{1, 14, 14, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1550, false);
  ::ttnn::Tensor v1552 = ttnn::permute(
      v1551, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1551, false);
  ::ttnn::Tensor v1553 = ttnn::batch_norm(
      v1552, v334, v894, false, 0.000010f, 0.100000f, v659, v1149,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1552, false);
  ::ttnn::Tensor v1554 =
      ttnn::add(v1549, v1553, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1553, false);
  ttnn::deallocate(v1549, false);
  ::ttnn::Tensor v1555 = ttnn::relu(
      v1554, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1554, false);
  ::ttnn::Tensor v1556 = ttnn::permute(
      v1555, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1557 = ttnn::reshape(
      v1556, ::std::vector<int32_t>{1, 1, 196, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1556, false);
  ::ttnn::Tensor v1558 = ttnn::to_layout(
      v1557, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1557, false);
  ::ttnn::Tensor v1559 = ::std::get<0>(ttnn::conv2d(
      v1558, v164, v1330, 1024, 256, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1558, false);
  ::ttnn::Tensor v1560 = ttnn::reshape(
      v1559, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1559, false);
  ::ttnn::Tensor v1561 = ttnn::permute(
      v1560, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1560, false);
  ::ttnn::Tensor v1562 = ttnn::batch_norm(
      v1561, v1079, v799, false, 0.000010f, 0.100000f, v969, v339,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1561, false);
  ::ttnn::Tensor v1563 = ttnn::relu(
      v1562, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1562, false);
  ::ttnn::Tensor v1564 = ttnn::permute(
      v1563, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1563, false);
  ::ttnn::Tensor v1565 = ttnn::reshape(
      v1564, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1564, false);
  ::ttnn::Tensor v1566 = ttnn::to_layout(
      v1565, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1565, false);
  ::ttnn::Tensor v1567 = ::std::get<0>(ttnn::conv2d(
      v1566, v159, v1330, 256, 256, 1, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1566, false);
  ::ttnn::Tensor v1568 = ttnn::reshape(
      v1567, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1567, false);
  ::ttnn::Tensor v1569 = ttnn::permute(
      v1568, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1568, false);
  ::ttnn::Tensor v1570 = ttnn::batch_norm(
      v1569, v914, v379, false, 0.000010f, 0.100000f, v1224, v1094,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1569, false);
  ::ttnn::Tensor v1571 = ttnn::relu(
      v1570, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1570, false);
  ::ttnn::Tensor v1572 = ttnn::permute(
      v1571, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1571, false);
  ::ttnn::Tensor v1573 = ttnn::reshape(
      v1572, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1572, false);
  ::ttnn::Tensor v1574 = ttnn::to_layout(
      v1573, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1573, false);
  ::ttnn::Tensor v1575 = ::std::get<0>(ttnn::conv2d(
      v1574, v154, v1330, 256, 1024, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1574, false);
  ::ttnn::Tensor v1576 = ttnn::reshape(
      v1575, ::std::vector<int32_t>{1, 14, 14, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1575, false);
  ::ttnn::Tensor v1577 = ttnn::permute(
      v1576, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1576, false);
  ::ttnn::Tensor v1578 = ttnn::batch_norm(
      v1577, v784, v924, false, 0.000010f, 0.100000f, v979, v1234,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1577, false);
  ::ttnn::Tensor v1579 =
      ttnn::add(v1578, v1555, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1578, false);
  ttnn::deallocate(v1555, false);
  ::ttnn::Tensor v1580 = ttnn::relu(
      v1579, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1579, false);
  ::ttnn::Tensor v1581 = ttnn::permute(
      v1580, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1582 = ttnn::reshape(
      v1581, ::std::vector<int32_t>{1, 1, 196, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1581, false);
  ::ttnn::Tensor v1583 = ttnn::to_layout(
      v1582, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1582, false);
  ::ttnn::Tensor v1584 = ::std::get<0>(ttnn::conv2d(
      v1583, v179, v1330, 1024, 256, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1583, false);
  ::ttnn::Tensor v1585 = ttnn::reshape(
      v1584, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1584, false);
  ::ttnn::Tensor v1586 = ttnn::permute(
      v1585, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1585, false);
  ::ttnn::Tensor v1587 = ttnn::batch_norm(
      v1586, v369, v469, false, 0.000010f, 0.100000f, v539, v1249,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1586, false);
  ::ttnn::Tensor v1588 = ttnn::relu(
      v1587, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1587, false);
  ::ttnn::Tensor v1589 = ttnn::permute(
      v1588, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1588, false);
  ::ttnn::Tensor v1590 = ttnn::reshape(
      v1589, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1589, false);
  ::ttnn::Tensor v1591 = ttnn::to_layout(
      v1590, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1590, false);
  ::ttnn::Tensor v1592 = ::std::get<0>(ttnn::conv2d(
      v1591, v174, v1330, 256, 256, 1, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1591, false);
  ::ttnn::Tensor v1593 = ttnn::reshape(
      v1592, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1592, false);
  ::ttnn::Tensor v1594 = ttnn::permute(
      v1593, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1593, false);
  ::ttnn::Tensor v1595 = ttnn::batch_norm(
      v1594, v1214, v1119, false, 0.000010f, 0.100000f, v279, v869,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1594, false);
  ::ttnn::Tensor v1596 = ttnn::relu(
      v1595, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1595, false);
  ::ttnn::Tensor v1597 = ttnn::permute(
      v1596, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1596, false);
  ::ttnn::Tensor v1598 = ttnn::reshape(
      v1597, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1597, false);
  ::ttnn::Tensor v1599 = ttnn::to_layout(
      v1598, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1598, false);
  ::ttnn::Tensor v1600 = ::std::get<0>(ttnn::conv2d(
      v1599, v169, v1330, 256, 1024, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1599, false);
  ::ttnn::Tensor v1601 = ttnn::reshape(
      v1600, ::std::vector<int32_t>{1, 14, 14, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1600, false);
  ::ttnn::Tensor v1602 = ttnn::permute(
      v1601, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1601, false);
  ::ttnn::Tensor v1603 = ttnn::batch_norm(
      v1602, v769, v1114, false, 0.000010f, 0.100000f, v1259, v524,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1602, false);
  ::ttnn::Tensor v1604 =
      ttnn::add(v1603, v1580, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1603, false);
  ttnn::deallocate(v1580, false);
  ::ttnn::Tensor v1605 = ttnn::relu(
      v1604, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1604, false);
  ::ttnn::Tensor v1606 = ttnn::permute(
      v1605, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1607 = ttnn::reshape(
      v1606, ::std::vector<int32_t>{1, 1, 196, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1606, false);
  ::ttnn::Tensor v1608 = ttnn::to_layout(
      v1607, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1607, false);
  ::ttnn::Tensor v1609 = ::std::get<0>(ttnn::conv2d(
      v1608, v194, v1330, 1024, 256, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1608, false);
  ::ttnn::Tensor v1610 = ttnn::reshape(
      v1609, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1609, false);
  ::ttnn::Tensor v1611 = ttnn::permute(
      v1610, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1610, false);
  ::ttnn::Tensor v1612 = ttnn::batch_norm(
      v1611, v329, v694, false, 0.000010f, 0.100000f, v834, v1144,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1611, false);
  ::ttnn::Tensor v1613 = ttnn::relu(
      v1612, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1612, false);
  ::ttnn::Tensor v1614 = ttnn::permute(
      v1613, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1613, false);
  ::ttnn::Tensor v1615 = ttnn::reshape(
      v1614, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1614, false);
  ::ttnn::Tensor v1616 = ttnn::to_layout(
      v1615, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1615, false);
  ::ttnn::Tensor v1617 = ::std::get<0>(ttnn::conv2d(
      v1616, v189, v1330, 256, 256, 1, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1616, false);
  ::ttnn::Tensor v1618 = ttnn::reshape(
      v1617, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1617, false);
  ::ttnn::Tensor v1619 = ttnn::permute(
      v1618, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1618, false);
  ::ttnn::Tensor v1620 = ttnn::batch_norm(
      v1619, v519, v724, false, 0.000010f, 0.100000f, v1099, v1164,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1619, false);
  ::ttnn::Tensor v1621 = ttnn::relu(
      v1620, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1620, false);
  ::ttnn::Tensor v1622 = ttnn::permute(
      v1621, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1621, false);
  ::ttnn::Tensor v1623 = ttnn::reshape(
      v1622, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1622, false);
  ::ttnn::Tensor v1624 = ttnn::to_layout(
      v1623, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1623, false);
  ::ttnn::Tensor v1625 = ::std::get<0>(ttnn::conv2d(
      v1624, v184, v1330, 256, 1024, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1624, false);
  ::ttnn::Tensor v1626 = ttnn::reshape(
      v1625, ::std::vector<int32_t>{1, 14, 14, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1625, false);
  ::ttnn::Tensor v1627 = ttnn::permute(
      v1626, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1626, false);
  ::ttnn::Tensor v1628 = ttnn::batch_norm(
      v1627, v499, v1314, false, 0.000010f, 0.100000f, v429, v709,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1627, false);
  ::ttnn::Tensor v1629 =
      ttnn::add(v1628, v1605, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1628, false);
  ttnn::deallocate(v1605, false);
  ::ttnn::Tensor v1630 = ttnn::relu(
      v1629, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1629, false);
  ::ttnn::Tensor v1631 = ttnn::permute(
      v1630, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1632 = ttnn::reshape(
      v1631, ::std::vector<int32_t>{1, 1, 196, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1631, false);
  ::ttnn::Tensor v1633 = ttnn::to_layout(
      v1632, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1632, false);
  ::ttnn::Tensor v1634 = ::std::get<0>(ttnn::conv2d(
      v1633, v209, v1330, 1024, 256, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1633, false);
  ::ttnn::Tensor v1635 = ttnn::reshape(
      v1634, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1634, false);
  ::ttnn::Tensor v1636 = ttnn::permute(
      v1635, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1635, false);
  ::ttnn::Tensor v1637 = ttnn::batch_norm(
      v1636, v1179, v614, false, 0.000010f, 0.100000f, v739, v1159,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1636, false);
  ::ttnn::Tensor v1638 = ttnn::relu(
      v1637, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1637, false);
  ::ttnn::Tensor v1639 = ttnn::permute(
      v1638, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1638, false);
  ::ttnn::Tensor v1640 = ttnn::reshape(
      v1639, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1639, false);
  ::ttnn::Tensor v1641 = ttnn::to_layout(
      v1640, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1640, false);
  ::ttnn::Tensor v1642 = ::std::get<0>(ttnn::conv2d(
      v1641, v204, v1330, 256, 256, 1, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1641, false);
  ::ttnn::Tensor v1643 = ttnn::reshape(
      v1642, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1642, false);
  ::ttnn::Tensor v1644 = ttnn::permute(
      v1643, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1643, false);
  ::ttnn::Tensor v1645 = ttnn::batch_norm(
      v1644, v874, v304, false, 0.000010f, 0.100000f, v1329, v414,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1644, false);
  ::ttnn::Tensor v1646 = ttnn::relu(
      v1645, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1645, false);
  ::ttnn::Tensor v1647 = ttnn::permute(
      v1646, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1646, false);
  ::ttnn::Tensor v1648 = ttnn::reshape(
      v1647, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1647, false);
  ::ttnn::Tensor v1649 = ttnn::to_layout(
      v1648, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1648, false);
  ::ttnn::Tensor v1650 = ::std::get<0>(ttnn::conv2d(
      v1649, v199, v1330, 256, 1024, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1649, false);
  ::ttnn::Tensor v1651 = ttnn::reshape(
      v1650, ::std::vector<int32_t>{1, 14, 14, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1650, false);
  ::ttnn::Tensor v1652 = ttnn::permute(
      v1651, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1651, false);
  ::ttnn::Tensor v1653 = ttnn::batch_norm(
      v1652, v409, v1134, false, 0.000010f, 0.100000f, v529, v494,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1652, false);
  ::ttnn::Tensor v1654 =
      ttnn::add(v1653, v1630, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1653, false);
  ttnn::deallocate(v1630, false);
  ::ttnn::Tensor v1655 = ttnn::relu(
      v1654, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1654, false);
  ::ttnn::Tensor v1656 = ttnn::permute(
      v1655, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1657 = ttnn::reshape(
      v1656, ::std::vector<int32_t>{1, 1, 196, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1656, false);
  ::ttnn::Tensor v1658 = ttnn::to_layout(
      v1657, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1657, false);
  ::ttnn::Tensor v1659 = ::std::get<0>(ttnn::conv2d(
      v1658, v224, v1330, 1024, 256, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1658, false);
  ::ttnn::Tensor v1660 = ttnn::reshape(
      v1659, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1659, false);
  ::ttnn::Tensor v1661 = ttnn::permute(
      v1660, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1660, false);
  ::ttnn::Tensor v1662 = ttnn::batch_norm(
      v1661, v1194, v479, false, 0.000010f, 0.100000f, v819, v394,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1661, false);
  ::ttnn::Tensor v1663 = ttnn::relu(
      v1662, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1662, false);
  ::ttnn::Tensor v1664 = ttnn::permute(
      v1663, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1663, false);
  ::ttnn::Tensor v1665 = ttnn::reshape(
      v1664, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1664, false);
  ::ttnn::Tensor v1666 = ttnn::to_layout(
      v1665, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1665, false);
  ::ttnn::Tensor v1667 = ::std::get<0>(ttnn::conv2d(
      v1666, v219, v1330, 256, 256, 1, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1666, false);
  ::ttnn::Tensor v1668 = ttnn::reshape(
      v1667, ::std::vector<int32_t>{1, 14, 14, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1667, false);
  ::ttnn::Tensor v1669 = ttnn::permute(
      v1668, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1668, false);
  ::ttnn::Tensor v1670 = ttnn::batch_norm(
      v1669, v949, v574, false, 0.000010f, 0.100000f, v509, v1304,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1669, false);
  ::ttnn::Tensor v1671 = ttnn::relu(
      v1670, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1670, false);
  ::ttnn::Tensor v1672 = ttnn::permute(
      v1671, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1671, false);
  ::ttnn::Tensor v1673 = ttnn::reshape(
      v1672, ::std::vector<int32_t>{1, 1, 196, 256},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1672, false);
  ::ttnn::Tensor v1674 = ttnn::to_layout(
      v1673, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1673, false);
  ::ttnn::Tensor v1675 = ::std::get<0>(ttnn::conv2d(
      v1674, v214, v1330, 256, 1024, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1674, false);
  ::ttnn::Tensor v1676 = ttnn::reshape(
      v1675, ::std::vector<int32_t>{1, 14, 14, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1675, false);
  ::ttnn::Tensor v1677 = ttnn::permute(
      v1676, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1676, false);
  ::ttnn::Tensor v1678 = ttnn::batch_norm(
      v1677, v374, v459, false, 0.000010f, 0.100000f, v899, v984,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1677, false);
  ::ttnn::Tensor v1679 =
      ttnn::add(v1678, v1655, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1678, false);
  ttnn::deallocate(v1655, false);
  ::ttnn::Tensor v1680 = ttnn::relu(
      v1679, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1679, false);
  ::ttnn::Tensor v1681 = ttnn::permute(
      v1680, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1680, false);
  ::ttnn::Tensor v1682 = ttnn::reshape(
      v1681, ::std::vector<int32_t>{1, 1, 196, 1024},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1681, false);
  ::ttnn::Tensor v1683 = ttnn::to_layout(
      v1682, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1682, false);
  ::ttnn::Tensor v1684 = ::std::get<0>(ttnn::conv2d(
      v1683, v239, v1330, 1024, 512, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ::ttnn::Tensor v1685 = ttnn::reshape(
      v1684, ::std::vector<int32_t>{1, 14, 14, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1684, false);
  ::ttnn::Tensor v1686 = ttnn::permute(
      v1685, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1685, false);
  ::ttnn::Tensor v1687 = ttnn::batch_norm(
      v1686, v1289, v464, false, 0.000010f, 0.100000f, v714, v1064,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1686, false);
  ::ttnn::Tensor v1688 = ttnn::relu(
      v1687, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1687, false);
  ::ttnn::Tensor v1689 = ttnn::permute(
      v1688, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1688, false);
  ::ttnn::Tensor v1690 = ttnn::reshape(
      v1689, ::std::vector<int32_t>{1, 1, 196, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1689, false);
  ::ttnn::Tensor v1691 = ttnn::to_layout(
      v1690, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1690, false);
  ::ttnn::Tensor v1692 = ::std::get<0>(ttnn::conv2d(
      v1691, v234, v1330, 512, 512, 1, 14, 14, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1691, false);
  ::ttnn::Tensor v1693 = ttnn::reshape(
      v1692, ::std::vector<int32_t>{1, 7, 7, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1692, false);
  ::ttnn::Tensor v1694 = ttnn::permute(
      v1693, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1693, false);
  ::ttnn::Tensor v1695 = ttnn::batch_norm(
      v1694, v294, v699, false, 0.000010f, 0.100000f, v859, v919,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1694, false);
  ::ttnn::Tensor v1696 = ttnn::relu(
      v1695, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1695, false);
  ::ttnn::Tensor v1697 = ttnn::permute(
      v1696, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1696, false);
  ::ttnn::Tensor v1698 = ttnn::reshape(
      v1697, ::std::vector<int32_t>{1, 1, 49, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1697, false);
  ::ttnn::Tensor v1699 = ttnn::to_layout(
      v1698, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1698, false);
  ::ttnn::Tensor v1700 = ::std::get<0>(ttnn::conv2d(
      v1699, v229, v1330, 512, 2048, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1699, false);
  ::ttnn::Tensor v1701 = ttnn::reshape(
      v1700, ::std::vector<int32_t>{1, 7, 7, 2048},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1700, false);
  ::ttnn::Tensor v1702 = ttnn::permute(
      v1701, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1701, false);
  ::ttnn::Tensor v1703 = ttnn::batch_norm(
      v1702, v1009, v1074, false, 0.000010f, 0.100000f, v604, v669,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1702, false);
  ::ttnn::Tensor v1704 = ::std::get<0>(ttnn::conv2d(
      v1683, v8, v1330, 1024, 2048, 1, 14, 14, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{2, 2}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1683, false);
  ::ttnn::Tensor v1705 = ttnn::reshape(
      v1704, ::std::vector<int32_t>{1, 7, 7, 2048},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1704, false);
  ::ttnn::Tensor v1706 = ttnn::permute(
      v1705, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1705, false);
  ::ttnn::Tensor v1707 = ttnn::batch_norm(
      v1706, v349, v809, false, 0.000010f, 0.100000f, v759, v959,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1706, false);
  ::ttnn::Tensor v1708 =
      ttnn::add(v1703, v1707, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1707, false);
  ttnn::deallocate(v1703, false);
  ::ttnn::Tensor v1709 = ttnn::relu(
      v1708, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1708, false);
  ::ttnn::Tensor v1710 = ttnn::permute(
      v1709, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1711 = ttnn::reshape(
      v1710, ::std::vector<int32_t>{1, 1, 49, 2048},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1710, false);
  ::ttnn::Tensor v1712 = ttnn::to_layout(
      v1711, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1711, false);
  ::ttnn::Tensor v1713 = ::std::get<0>(ttnn::conv2d(
      v1712, v254, v1330, 2048, 512, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1712, false);
  ::ttnn::Tensor v1714 = ttnn::reshape(
      v1713, ::std::vector<int32_t>{1, 7, 7, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1713, false);
  ::ttnn::Tensor v1715 = ttnn::permute(
      v1714, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1714, false);
  ::ttnn::Tensor v1716 = ttnn::batch_norm(
      v1715, v549, v649, false, 0.000010f, 0.100000f, v1129, v1239,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1715, false);
  ::ttnn::Tensor v1717 = ttnn::relu(
      v1716, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1716, false);
  ::ttnn::Tensor v1718 = ttnn::permute(
      v1717, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1717, false);
  ::ttnn::Tensor v1719 = ttnn::reshape(
      v1718, ::std::vector<int32_t>{1, 1, 49, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1718, false);
  ::ttnn::Tensor v1720 = ttnn::to_layout(
      v1719, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1719, false);
  ::ttnn::Tensor v1721 = ::std::get<0>(ttnn::conv2d(
      v1720, v249, v1330, 512, 512, 1, 7, 7, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1720, false);
  ::ttnn::Tensor v1722 = ttnn::reshape(
      v1721, ::std::vector<int32_t>{1, 7, 7, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1721, false);
  ::ttnn::Tensor v1723 = ttnn::permute(
      v1722, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1722, false);
  ::ttnn::Tensor v1724 = ttnn::batch_norm(
      v1723, v1029, v804, false, 0.000010f, 0.100000f, v1034, v1229,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1723, false);
  ::ttnn::Tensor v1725 = ttnn::relu(
      v1724, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1724, false);
  ::ttnn::Tensor v1726 = ttnn::permute(
      v1725, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1725, false);
  ::ttnn::Tensor v1727 = ttnn::reshape(
      v1726, ::std::vector<int32_t>{1, 1, 49, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1726, false);
  ::ttnn::Tensor v1728 = ttnn::to_layout(
      v1727, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1727, false);
  ::ttnn::Tensor v1729 = ::std::get<0>(ttnn::conv2d(
      v1728, v244, v1330, 512, 2048, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1728, false);
  ::ttnn::Tensor v1730 = ttnn::reshape(
      v1729, ::std::vector<int32_t>{1, 7, 7, 2048},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1729, false);
  ::ttnn::Tensor v1731 = ttnn::permute(
      v1730, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1730, false);
  ::ttnn::Tensor v1732 = ttnn::batch_norm(
      v1731, v624, v629, false, 0.000010f, 0.100000f, v1254, v1039,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1731, false);
  ::ttnn::Tensor v1733 =
      ttnn::add(v1732, v1709, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1732, false);
  ttnn::deallocate(v1709, false);
  ::ttnn::Tensor v1734 = ttnn::relu(
      v1733, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1733, false);
  ::ttnn::Tensor v1735 = ttnn::permute(
      v1734, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ::ttnn::Tensor v1736 = ttnn::reshape(
      v1735, ::std::vector<int32_t>{1, 1, 49, 2048},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1735, false);
  ::ttnn::Tensor v1737 = ttnn::to_layout(
      v1736, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1736, false);
  ::ttnn::Tensor v1738 = ::std::get<0>(ttnn::conv2d(
      v1737, v269, v1330, 2048, 512, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1737, false);
  ::ttnn::Tensor v1739 = ttnn::reshape(
      v1738, ::std::vector<int32_t>{1, 7, 7, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1738, false);
  ::ttnn::Tensor v1740 = ttnn::permute(
      v1739, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1739, false);
  ::ttnn::Tensor v1741 = ttnn::batch_norm(
      v1740, v1209, v654, false, 0.000010f, 0.100000f, v789, v844,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1740, false);
  ::ttnn::Tensor v1742 = ttnn::relu(
      v1741, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1741, false);
  ::ttnn::Tensor v1743 = ttnn::permute(
      v1742, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1742, false);
  ::ttnn::Tensor v1744 = ttnn::reshape(
      v1743, ::std::vector<int32_t>{1, 1, 49, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1743, false);
  ::ttnn::Tensor v1745 = ttnn::to_layout(
      v1744, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1744, false);
  ::ttnn::Tensor v1746 = ::std::get<0>(ttnn::conv2d(
      v1745, v264, v1330, 512, 512, 1, 7, 7, ::std::array<uint32_t, 2>{3, 3},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{1, 1, 1, 1},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1745, false);
  ::ttnn::Tensor v1747 = ttnn::reshape(
      v1746, ::std::vector<int32_t>{1, 7, 7, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1746, false);
  ::ttnn::Tensor v1748 = ttnn::permute(
      v1747, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1747, false);
  ::ttnn::Tensor v1749 = ttnn::batch_norm(
      v1748, v554, v1279, false, 0.000010f, 0.100000f, v939, v749,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1748, false);
  ::ttnn::Tensor v1750 = ttnn::relu(
      v1749, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1749, false);
  ::ttnn::Tensor v1751 = ttnn::permute(
      v1750, ::ttsl::SmallVector<int64_t>{0, 2, 3, 1},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1750, false);
  ::ttnn::Tensor v1752 = ttnn::reshape(
      v1751, ::std::vector<int32_t>{1, 1, 49, 512},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1751, false);
  ::ttnn::Tensor v1753 = ttnn::to_layout(
      v1752, ::ttnn::Layout::ROW_MAJOR, ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1752, false);
  ::ttnn::Tensor v1754 = ::std::get<0>(ttnn::conv2d(
      v1753, v259, v1330, 512, 2048, 1, 7, 7, ::std::array<uint32_t, 2>{1, 1},
      ::std::array<uint32_t, 2>{1, 1}, ::std::array<uint32_t, 4>{0, 0, 0, 0},
      ::std::array<uint32_t, 2>{1, 1}, 1, ::ttnn::DataType::FLOAT32,
      ::std::nullopt,
      ::ttnn::Conv2dConfig{.config_tensors_in_dram = true,
                           .enable_kernel_stride_folding = false},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::std::nullopt));
  ttnn::deallocate(v1753, false);
  ::ttnn::Tensor v1755 = ttnn::reshape(
      v1754, ::std::vector<int32_t>{1, 7, 7, 2048},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1754, false);
  ::ttnn::Tensor v1756 = ttnn::permute(
      v1755, ::ttsl::SmallVector<int64_t>{0, 3, 1, 2},
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      0.000000f);
  ttnn::deallocate(v1755, false);
  ::ttnn::Tensor v1757 = ttnn::batch_norm(
      v1756, v344, v619, false, 0.000010f, 0.100000f, v679, v864,
      ::std::nullopt,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1756, false);
  ::ttnn::Tensor v1758 =
      ttnn::add(v1757, v1734, ::ttnn::DataType::FLOAT32,
                ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                     ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1757, false);
  ttnn::deallocate(v1734, false);
  ::ttnn::Tensor v1759 = ttnn::relu(
      v1758, ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                  ::ttnn::BufferType::DRAM, ::std::nullopt});
  ttnn::deallocate(v1758, false);
  ::ttnn::Tensor v1760 =
      ttnn::mean(v1759, ::ttsl::SmallVector<int32_t>{2, 3}, false,
                 ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                                      ::ttnn::BufferType::DRAM, ::std::nullopt},
                 ::ttnn::WormholeComputeKernelConfig{
                     .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
                     .fp32_dest_acc_en = true});
  ttnn::deallocate(v1759, false);
  ::ttnn::Tensor v1761 = ttnn::linear(
      v1760, v3, v2, false, true,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt},
      ::ttnn::DataType::FLOAT32, ::std::nullopt, ::std::nullopt,
      ::ttnn::WormholeComputeKernelConfig{
          .math_fidelity = ::tt::tt_metal::MathFidelity::HiFi4,
          .fp32_dest_acc_en = true});
  ttnn::deallocate(v1760, false);
  ::std::vector<::ttnn::Tensor> v1762 = util_create_vec(v1761);
  return v1762;
}
::std::vector<::ttnn::Tensor> load_inputs_for__main() {
  ttnn::distributed::MeshDevice *v1 = ttnn::DeviceGetter::getInstance();
  ::ttnn::Tensor v2 = ttnn::loadTensor(
      "./tensors/arg0.tensorbin", ::ttnn::Layout::TILE,
      ::ttnn::DataType::FLOAT32, v1,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v3 = ttnn::loadTensor(
      "./tensors/arg1.tensorbin", ::ttnn::Layout::TILE,
      ::ttnn::DataType::FLOAT32, v1,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v4 = ttnn::loadTensor(
      "./tensors/arg2.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v5 = ttnn::loadTensor(
      "./tensors/arg3.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v6 = ttnn::loadTensor(
      "./tensors/arg4.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v7 = ttnn::loadTensor(
      "./tensors/arg5.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v8 = ttnn::loadTensor(
      "./tensors/arg6.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v9 = ttnn::loadTensor(
      "./tensors/arg7.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v10 = ttnn::loadTensor(
      "./tensors/arg8.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v11 = ttnn::loadTensor(
      "./tensors/arg9.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v12 = ttnn::loadTensor(
      "./tensors/arg10.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v13 = ttnn::loadTensor(
      "./tensors/arg11.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v14 = ttnn::loadTensor(
      "./tensors/arg12.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v15 = ttnn::loadTensor(
      "./tensors/arg13.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v16 = ttnn::loadTensor(
      "./tensors/arg14.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v17 = ttnn::loadTensor(
      "./tensors/arg15.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v18 = ttnn::loadTensor(
      "./tensors/arg16.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v19 = ttnn::loadTensor(
      "./tensors/arg17.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v20 = ttnn::loadTensor(
      "./tensors/arg18.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v21 = ttnn::loadTensor(
      "./tensors/arg19.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v22 = ttnn::loadTensor(
      "./tensors/arg20.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v23 = ttnn::loadTensor(
      "./tensors/arg21.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v24 = ttnn::loadTensor(
      "./tensors/arg22.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v25 = ttnn::loadTensor(
      "./tensors/arg23.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v26 = ttnn::loadTensor(
      "./tensors/arg24.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v27 = ttnn::loadTensor(
      "./tensors/arg25.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v28 = ttnn::loadTensor(
      "./tensors/arg26.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v29 = ttnn::loadTensor(
      "./tensors/arg27.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, v1,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
  ::ttnn::Tensor v30 = ttnn::loadTensor(
      "./tensors/arg28.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v31 = ttnn::loadTensor(
      "./tensors/arg29.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v32 = ttnn::loadTensor(
      "./tensors/arg30.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v33 = ttnn::loadTensor(
      "./tensors/arg31.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v34 = ttnn::loadTensor(
      "./tensors/arg32.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v35 = ttnn::loadTensor(
      "./tensors/arg33.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v36 = ttnn::loadTensor(
      "./tensors/arg34.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v37 = ttnn::loadTensor(
      "./tensors/arg35.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v38 = ttnn::loadTensor(
      "./tensors/arg36.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v39 = ttnn::loadTensor(
      "./tensors/arg37.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v40 = ttnn::loadTensor(
      "./tensors/arg38.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v41 = ttnn::loadTensor(
      "./tensors/arg39.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v42 = ttnn::loadTensor(
      "./tensors/arg40.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v43 = ttnn::loadTensor(
      "./tensors/arg41.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v44 = ttnn::loadTensor(
      "./tensors/arg42.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v45 = ttnn::loadTensor(
      "./tensors/arg43.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v46 = ttnn::loadTensor(
      "./tensors/arg44.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v47 = ttnn::loadTensor(
      "./tensors/arg45.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v48 = ttnn::loadTensor(
      "./tensors/arg46.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v49 = ttnn::loadTensor(
      "./tensors/arg47.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v50 = ttnn::loadTensor(
      "./tensors/arg48.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v51 = ttnn::loadTensor(
      "./tensors/arg49.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v52 = ttnn::loadTensor(
      "./tensors/arg50.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v53 = ttnn::loadTensor(
      "./tensors/arg51.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v54 = ttnn::loadTensor(
      "./tensors/arg52.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v55 = ttnn::loadTensor(
      "./tensors/arg53.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v56 = ttnn::loadTensor(
      "./tensors/arg54.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v57 = ttnn::loadTensor(
      "./tensors/arg55.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v58 = ttnn::loadTensor(
      "./tensors/arg56.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v59 = ttnn::loadTensor(
      "./tensors/arg57.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v60 = ttnn::loadTensor(
      "./tensors/arg58.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v61 = ttnn::loadTensor(
      "./tensors/arg59.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v62 = ttnn::loadTensor(
      "./tensors/arg60.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v63 = ttnn::loadTensor(
      "./tensors/arg61.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v64 = ttnn::loadTensor(
      "./tensors/arg62.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v65 = ttnn::loadTensor(
      "./tensors/arg63.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v66 = ttnn::loadTensor(
      "./tensors/arg64.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v67 = ttnn::loadTensor(
      "./tensors/arg65.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v68 = ttnn::loadTensor(
      "./tensors/arg66.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v69 = ttnn::loadTensor(
      "./tensors/arg67.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v70 = ttnn::loadTensor(
      "./tensors/arg68.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v71 = ttnn::loadTensor(
      "./tensors/arg69.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v72 = ttnn::loadTensor(
      "./tensors/arg70.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v73 = ttnn::loadTensor(
      "./tensors/arg71.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v74 = ttnn::loadTensor(
      "./tensors/arg72.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v75 = ttnn::loadTensor(
      "./tensors/arg73.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v76 = ttnn::loadTensor(
      "./tensors/arg74.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v77 = ttnn::loadTensor(
      "./tensors/arg75.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v78 = ttnn::loadTensor(
      "./tensors/arg76.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v79 = ttnn::loadTensor(
      "./tensors/arg77.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v80 = ttnn::loadTensor(
      "./tensors/arg78.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v81 = ttnn::loadTensor(
      "./tensors/arg79.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v82 = ttnn::loadTensor(
      "./tensors/arg80.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v83 = ttnn::loadTensor(
      "./tensors/arg81.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v84 = ttnn::loadTensor(
      "./tensors/arg82.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v85 = ttnn::loadTensor(
      "./tensors/arg83.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v86 = ttnn::loadTensor(
      "./tensors/arg84.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v87 = ttnn::loadTensor(
      "./tensors/arg85.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v88 = ttnn::loadTensor(
      "./tensors/arg86.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v89 = ttnn::loadTensor(
      "./tensors/arg87.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v90 = ttnn::loadTensor(
      "./tensors/arg88.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v91 = ttnn::loadTensor(
      "./tensors/arg89.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v92 = ttnn::loadTensor(
      "./tensors/arg90.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v93 = ttnn::loadTensor(
      "./tensors/arg91.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v94 = ttnn::loadTensor(
      "./tensors/arg92.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v95 = ttnn::loadTensor(
      "./tensors/arg93.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v96 = ttnn::loadTensor(
      "./tensors/arg94.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v97 = ttnn::loadTensor(
      "./tensors/arg95.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v98 = ttnn::loadTensor(
      "./tensors/arg96.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v99 = ttnn::loadTensor(
      "./tensors/arg97.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v100 = ttnn::loadTensor(
      "./tensors/arg98.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v101 = ttnn::loadTensor(
      "./tensors/arg99.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v102 = ttnn::loadTensor(
      "./tensors/arg100.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v103 = ttnn::loadTensor(
      "./tensors/arg101.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v104 = ttnn::loadTensor(
      "./tensors/arg102.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v105 = ttnn::loadTensor(
      "./tensors/arg103.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v106 = ttnn::loadTensor(
      "./tensors/arg104.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v107 = ttnn::loadTensor(
      "./tensors/arg105.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v108 = ttnn::loadTensor(
      "./tensors/arg106.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v109 = ttnn::loadTensor(
      "./tensors/arg107.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v110 = ttnn::loadTensor(
      "./tensors/arg108.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v111 = ttnn::loadTensor(
      "./tensors/arg109.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v112 = ttnn::loadTensor(
      "./tensors/arg110.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v113 = ttnn::loadTensor(
      "./tensors/arg111.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v114 = ttnn::loadTensor(
      "./tensors/arg112.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v115 = ttnn::loadTensor(
      "./tensors/arg113.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v116 = ttnn::loadTensor(
      "./tensors/arg114.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v117 = ttnn::loadTensor(
      "./tensors/arg115.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v118 = ttnn::loadTensor(
      "./tensors/arg116.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v119 = ttnn::loadTensor(
      "./tensors/arg117.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v120 = ttnn::loadTensor(
      "./tensors/arg118.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v121 = ttnn::loadTensor(
      "./tensors/arg119.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v122 = ttnn::loadTensor(
      "./tensors/arg120.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v123 = ttnn::loadTensor(
      "./tensors/arg121.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v124 = ttnn::loadTensor(
      "./tensors/arg122.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v125 = ttnn::loadTensor(
      "./tensors/arg123.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v126 = ttnn::loadTensor(
      "./tensors/arg124.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v127 = ttnn::loadTensor(
      "./tensors/arg125.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v128 = ttnn::loadTensor(
      "./tensors/arg126.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v129 = ttnn::loadTensor(
      "./tensors/arg127.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v130 = ttnn::loadTensor(
      "./tensors/arg128.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v131 = ttnn::loadTensor(
      "./tensors/arg129.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v132 = ttnn::loadTensor(
      "./tensors/arg130.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v133 = ttnn::loadTensor(
      "./tensors/arg131.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v134 = ttnn::loadTensor(
      "./tensors/arg132.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v135 = ttnn::loadTensor(
      "./tensors/arg133.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v136 = ttnn::loadTensor(
      "./tensors/arg134.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v137 = ttnn::loadTensor(
      "./tensors/arg135.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v138 = ttnn::loadTensor(
      "./tensors/arg136.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v139 = ttnn::loadTensor(
      "./tensors/arg137.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v140 = ttnn::loadTensor(
      "./tensors/arg138.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v141 = ttnn::loadTensor(
      "./tensors/arg139.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v142 = ttnn::loadTensor(
      "./tensors/arg140.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v143 = ttnn::loadTensor(
      "./tensors/arg141.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v144 = ttnn::loadTensor(
      "./tensors/arg142.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v145 = ttnn::loadTensor(
      "./tensors/arg143.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v146 = ttnn::loadTensor(
      "./tensors/arg144.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v147 = ttnn::loadTensor(
      "./tensors/arg145.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v148 = ttnn::loadTensor(
      "./tensors/arg146.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v149 = ttnn::loadTensor(
      "./tensors/arg147.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v150 = ttnn::loadTensor(
      "./tensors/arg148.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v151 = ttnn::loadTensor(
      "./tensors/arg149.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v152 = ttnn::loadTensor(
      "./tensors/arg150.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v153 = ttnn::loadTensor(
      "./tensors/arg151.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v154 = ttnn::loadTensor(
      "./tensors/arg152.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v155 = ttnn::loadTensor(
      "./tensors/arg153.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v156 = ttnn::loadTensor(
      "./tensors/arg154.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v157 = ttnn::loadTensor(
      "./tensors/arg155.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v158 = ttnn::loadTensor(
      "./tensors/arg156.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v159 = ttnn::loadTensor(
      "./tensors/arg157.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v160 = ttnn::loadTensor(
      "./tensors/arg158.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v161 = ttnn::loadTensor(
      "./tensors/arg159.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v162 = ttnn::loadTensor(
      "./tensors/arg160.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v163 = ttnn::loadTensor(
      "./tensors/arg161.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v164 = ttnn::loadTensor(
      "./tensors/arg162.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v165 = ttnn::loadTensor(
      "./tensors/arg163.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v166 = ttnn::loadTensor(
      "./tensors/arg164.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v167 = ttnn::loadTensor(
      "./tensors/arg165.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v168 = ttnn::loadTensor(
      "./tensors/arg166.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v169 = ttnn::loadTensor(
      "./tensors/arg167.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v170 = ttnn::loadTensor(
      "./tensors/arg168.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v171 = ttnn::loadTensor(
      "./tensors/arg169.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v172 = ttnn::loadTensor(
      "./tensors/arg170.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v173 = ttnn::loadTensor(
      "./tensors/arg171.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v174 = ttnn::loadTensor(
      "./tensors/arg172.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v175 = ttnn::loadTensor(
      "./tensors/arg173.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v176 = ttnn::loadTensor(
      "./tensors/arg174.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v177 = ttnn::loadTensor(
      "./tensors/arg175.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v178 = ttnn::loadTensor(
      "./tensors/arg176.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v179 = ttnn::loadTensor(
      "./tensors/arg177.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v180 = ttnn::loadTensor(
      "./tensors/arg178.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v181 = ttnn::loadTensor(
      "./tensors/arg179.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v182 = ttnn::loadTensor(
      "./tensors/arg180.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v183 = ttnn::loadTensor(
      "./tensors/arg181.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v184 = ttnn::loadTensor(
      "./tensors/arg182.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v185 = ttnn::loadTensor(
      "./tensors/arg183.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v186 = ttnn::loadTensor(
      "./tensors/arg184.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v187 = ttnn::loadTensor(
      "./tensors/arg185.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v188 = ttnn::loadTensor(
      "./tensors/arg186.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v189 = ttnn::loadTensor(
      "./tensors/arg187.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v190 = ttnn::loadTensor(
      "./tensors/arg188.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v191 = ttnn::loadTensor(
      "./tensors/arg189.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v192 = ttnn::loadTensor(
      "./tensors/arg190.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v193 = ttnn::loadTensor(
      "./tensors/arg191.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v194 = ttnn::loadTensor(
      "./tensors/arg192.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v195 = ttnn::loadTensor(
      "./tensors/arg193.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v196 = ttnn::loadTensor(
      "./tensors/arg194.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v197 = ttnn::loadTensor(
      "./tensors/arg195.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v198 = ttnn::loadTensor(
      "./tensors/arg196.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v199 = ttnn::loadTensor(
      "./tensors/arg197.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v200 = ttnn::loadTensor(
      "./tensors/arg198.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v201 = ttnn::loadTensor(
      "./tensors/arg199.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v202 = ttnn::loadTensor(
      "./tensors/arg200.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v203 = ttnn::loadTensor(
      "./tensors/arg201.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v204 = ttnn::loadTensor(
      "./tensors/arg202.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v205 = ttnn::loadTensor(
      "./tensors/arg203.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v206 = ttnn::loadTensor(
      "./tensors/arg204.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v207 = ttnn::loadTensor(
      "./tensors/arg205.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v208 = ttnn::loadTensor(
      "./tensors/arg206.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v209 = ttnn::loadTensor(
      "./tensors/arg207.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v210 = ttnn::loadTensor(
      "./tensors/arg208.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v211 = ttnn::loadTensor(
      "./tensors/arg209.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v212 = ttnn::loadTensor(
      "./tensors/arg210.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v213 = ttnn::loadTensor(
      "./tensors/arg211.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v214 = ttnn::loadTensor(
      "./tensors/arg212.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v215 = ttnn::loadTensor(
      "./tensors/arg213.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v216 = ttnn::loadTensor(
      "./tensors/arg214.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v217 = ttnn::loadTensor(
      "./tensors/arg215.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v218 = ttnn::loadTensor(
      "./tensors/arg216.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v219 = ttnn::loadTensor(
      "./tensors/arg217.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v220 = ttnn::loadTensor(
      "./tensors/arg218.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v221 = ttnn::loadTensor(
      "./tensors/arg219.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v222 = ttnn::loadTensor(
      "./tensors/arg220.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v223 = ttnn::loadTensor(
      "./tensors/arg221.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v224 = ttnn::loadTensor(
      "./tensors/arg222.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v225 = ttnn::loadTensor(
      "./tensors/arg223.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v226 = ttnn::loadTensor(
      "./tensors/arg224.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v227 = ttnn::loadTensor(
      "./tensors/arg225.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v228 = ttnn::loadTensor(
      "./tensors/arg226.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v229 = ttnn::loadTensor(
      "./tensors/arg227.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v230 = ttnn::loadTensor(
      "./tensors/arg228.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v231 = ttnn::loadTensor(
      "./tensors/arg229.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v232 = ttnn::loadTensor(
      "./tensors/arg230.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v233 = ttnn::loadTensor(
      "./tensors/arg231.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v234 = ttnn::loadTensor(
      "./tensors/arg232.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v235 = ttnn::loadTensor(
      "./tensors/arg233.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v236 = ttnn::loadTensor(
      "./tensors/arg234.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v237 = ttnn::loadTensor(
      "./tensors/arg235.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v238 = ttnn::loadTensor(
      "./tensors/arg236.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v239 = ttnn::loadTensor(
      "./tensors/arg237.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v240 = ttnn::loadTensor(
      "./tensors/arg238.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v241 = ttnn::loadTensor(
      "./tensors/arg239.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v242 = ttnn::loadTensor(
      "./tensors/arg240.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v243 = ttnn::loadTensor(
      "./tensors/arg241.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v244 = ttnn::loadTensor(
      "./tensors/arg242.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v245 = ttnn::loadTensor(
      "./tensors/arg243.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v246 = ttnn::loadTensor(
      "./tensors/arg244.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v247 = ttnn::loadTensor(
      "./tensors/arg245.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v248 = ttnn::loadTensor(
      "./tensors/arg246.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v249 = ttnn::loadTensor(
      "./tensors/arg247.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v250 = ttnn::loadTensor(
      "./tensors/arg248.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v251 = ttnn::loadTensor(
      "./tensors/arg249.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v252 = ttnn::loadTensor(
      "./tensors/arg250.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v253 = ttnn::loadTensor(
      "./tensors/arg251.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v254 = ttnn::loadTensor(
      "./tensors/arg252.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v255 = ttnn::loadTensor(
      "./tensors/arg253.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v256 = ttnn::loadTensor(
      "./tensors/arg254.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v257 = ttnn::loadTensor(
      "./tensors/arg255.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v258 = ttnn::loadTensor(
      "./tensors/arg256.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v259 = ttnn::loadTensor(
      "./tensors/arg257.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v260 = ttnn::loadTensor(
      "./tensors/arg258.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v261 = ttnn::loadTensor(
      "./tensors/arg259.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v262 = ttnn::loadTensor(
      "./tensors/arg260.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v263 = ttnn::loadTensor(
      "./tensors/arg261.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v264 = ttnn::loadTensor(
      "./tensors/arg262.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v265 = ttnn::loadTensor(
      "./tensors/arg263.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v266 = ttnn::loadTensor(
      "./tensors/arg264.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v267 = ttnn::loadTensor(
      "./tensors/arg265.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v268 = ttnn::loadTensor(
      "./tensors/arg266.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::ttnn::Tensor v269 = ttnn::loadTensor(
      "./tensors/arg267.tensorbin", ::ttnn::Layout::ROW_MAJOR,
      ::ttnn::DataType::FLOAT32, nullptr,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::SYSTEM_MEMORY, ::std::nullopt});
  ::std::vector<::ttnn::Tensor> v270 = util_create_vec(
      v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17,
      v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32,
      v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47,
      v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62,
      v63, v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77,
      v78, v79, v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92,
      v93, v94, v95, v96, v97, v98, v99, v100, v101, v102, v103, v104, v105,
      v106, v107, v108, v109, v110, v111, v112, v113, v114, v115, v116, v117,
      v118, v119, v120, v121, v122, v123, v124, v125, v126, v127, v128, v129,
      v130, v131, v132, v133, v134, v135, v136, v137, v138, v139, v140, v141,
      v142, v143, v144, v145, v146, v147, v148, v149, v150, v151, v152, v153,
      v154, v155, v156, v157, v158, v159, v160, v161, v162, v163, v164, v165,
      v166, v167, v168, v169, v170, v171, v172, v173, v174, v175, v176, v177,
      v178, v179, v180, v181, v182, v183, v184, v185, v186, v187, v188, v189,
      v190, v191, v192, v193, v194, v195, v196, v197, v198, v199, v200, v201,
      v202, v203, v204, v205, v206, v207, v208, v209, v210, v211, v212, v213,
      v214, v215, v216, v217, v218, v219, v220, v221, v222, v223, v224, v225,
      v226, v227, v228, v229, v230, v231, v232, v233, v234, v235, v236, v237,
      v238, v239, v240, v241, v242, v243, v244, v245, v246, v247, v248, v249,
      v250, v251, v252, v253, v254, v255, v256, v257, v258, v259, v260, v261,
      v262, v263, v264, v265, v266, v267, v268, v269);
  return v270;
}
int32_t main() {
  ::std::vector<::ttnn::Tensor> v1 = load_inputs_for__main();
  ::std::vector<::ttnn::Tensor> v2 = _main(v1);
  int32_t v3 = 0;
  return v3;
}
