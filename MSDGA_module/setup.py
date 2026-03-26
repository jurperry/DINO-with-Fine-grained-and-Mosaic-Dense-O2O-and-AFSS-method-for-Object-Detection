import glob
import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, BuildExtension

# 包信息
version = "0.1.0"
package_name = "msdga_module"

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    
    # MSDGA module
    extensions_dir = os.path.join(this_dir, "msdga_module", "csrc")
    sources = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    # using  MsDeformAttn dir
    source_cuda = glob.glob(os.path.join(extensions_dir, "MsDeformAttn", "*.cu"))
    
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # 检查CUDA是否可用
    if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
        print("Compiling MSDA with CUDA")
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        print("Compiling MSDA without CUDA")
        # if no CUDA，just compiling CPU version

    include_dirs = [extensions_dir, os.path.join(extensions_dir, "MsDeformAttn")]

    ext_modules = [
        extension(
            "msdga_module._C",  # package's name: msdga_module._C
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

if __name__ == "__main__":
    print(f"Building MSDGA module {package_name}-{version}")

    setup(
        name=package_name,
        version=version,
        author="jsr1813",
        description="Multi-Scale Deformable And Gated Attention Module",
        install_requires=["torch", "torchvision"],
        packages=find_packages(),
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
    )