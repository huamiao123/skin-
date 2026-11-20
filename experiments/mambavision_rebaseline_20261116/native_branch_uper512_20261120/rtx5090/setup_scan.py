from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension,CUDAExtension
src=Path('/home/waas/mv_branch_sm120_20261120/mamba-v1.2.2/csrc/selective_scan')
sources=[src/'selective_scan.cpp',*sorted(src.glob('*.cu'))]
assert len(sources)==10
flags=['-O3','-std=c++17','-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__','-U__CUDA_NO_BFLOAT16_OPERATORS__','-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_BFLOAT162_OPERATORS__','-U__CUDA_NO_HALF2_OPERATORS__','--expt-relaxed-constexpr','--expt-extended-lambda','--use_fast_math','-lineinfo']
setup(name='selective_scan_cuda',ext_modules=[CUDAExtension(name='selective_scan_cuda',sources=list(map(str,sources)),include_dirs=[str(src)],extra_compile_args={'cxx':['-O2','-g0','-std=c++17'],'nvcc':flags})],cmdclass={'build_ext':BuildExtension})

