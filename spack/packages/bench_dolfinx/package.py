# Copyright 2025 Chris Richardson and Garth N. Wells.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

from spack.package import *


class BenchDolfinx(CMakePackage, CudaPackage, ROCmPackage):
    "A benchmark using DOLFINx."

    homepage = "https://github.com/ukri-bench/ukri-bench"
    git = "https://github.com/ukri-bench/benchmark-dolfinx.git"

    version("main", tag="main")

    depends_on("c", type="build")
    depends_on("cxx", type="build")

    depends_on("fenics-dolfinx@main")
    depends_on("py-fenics-ffcx@main", type="build")
    depends_on("py-fenics-ufl@main", type="build")

    depends_on("boost+program_options")
    depends_on("mpi")
    depends_on("hip", when="+rocm")
    depends_on("cuda", when="+cuda")
    depends_on("jsoncpp")

    conflicts("+cuda", when="+rocm", msg="Cannot build for both ROCm and CUDA")

    variant("fp32", default=False, description="Build for float32 scalar type")

    with when("+rocm"):
        depends_on("rocm-core")
        depends_on("rocthrust")

    def cmake_args(self):
        args = [
            self.define("SCALAR_TYPE", "float32" if "+fp32" in self.spec else "float64")
        ]
        if "+rocm" in self.spec:
            args += [self.define("HIP_ARCH", self.spec.variants["amdgpu_target"].value)]
        if "+cuda" in self.spec:
            args += [self.define("CUDA_ARCH", self.spec.variants["cuda_arch"].value)]
        return args
