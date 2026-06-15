# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension


def split_env_paths(name):
    value = os.environ.get(name, "")
    return [path for path in value.split(os.pathsep) if path]


this_dir = Path(__file__).resolve().parent

setup(
    name="cpp_resnet_preprocess",
    ext_modules=[
        CppExtension(
            name="cpp_resnet_preprocess",
            sources=[str(this_dir / "cpp_resnet_preprocess.cpp")],
            include_dirs=split_env_paths("TURBOJPEG_INCLUDE_DIR"),
            library_dirs=split_env_paths("TURBOJPEG_LIB_DIR"),
            runtime_library_dirs=split_env_paths("TURBOJPEG_LIB_DIR"),
            libraries=["turbojpeg"],
            extra_compile_args=["-O3", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
