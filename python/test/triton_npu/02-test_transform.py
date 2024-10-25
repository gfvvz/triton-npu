# TRITON_ENABLE_LLVM_DEBUG=1 TRITON_NPU_BACKEND=1 python3 python/test/triton_npu/01-test_empty_kernel.py

import triton
import triton.language as tl

from triton.compiler import ASTSource

# import pdb
# pdb.set_trace()

target = triton.runtime.driver.active.get_current_target()

def is_npu():
    return target.backend == "npu"

@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    foo = pid + 42
    tl.device_print("Hello, World!", foo, pid)

src = ASTSource(
    fn=add_kernel,
    constants={'BLOCK_SIZE': 32},
    signature={'x_ptr': "*fp32", 'y_ptr': "*fp32", 'output_ptr': "*fp32"},
)

handler = triton.compile(src=src, target=target)

print("Finished")