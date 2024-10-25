import torch
import os
from IPython.core.debugger import set_trace

#os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import triton
import triton.language as tl

def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_npu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "npu"


def cdiv(a,b): return (a + b - 1) // b


@triton.jit
def vector_add(a_ptr, b_ptr, c_ptr, n, bs:tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * bs + tl.arange(0, bs)
    
    mask = offset < n
    a = tl.load(a_ptr+offset, mask)
    b = tl.load(b_ptr+offset, mask)

    c = a + b
    tl.store(c_ptr+offset, c, mask)

def vector_add_host(a, b, bs, kernel_fn):
    c = torch.zeros_like(a)
    n = a.numel()
    n_blocks = cdiv(n, bs)
    grid = (n_blocks,)  # how many blocks do we have? can be 1d/2d/3d-tuple or function returning 1d/2d/3d-tuple

    # launch grid!
    # - kernel_fn is the triton kernel, which we write above
    # - grid is the grid we constructed above
    # - x,z,n,bs are paramters that are passed into each kernel function
    kernel_fn[grid](a, b, c, n, bs)

    print(a)
    print(b)
    print(c)

    return c


x = torch.tensor([1,2,3,4,5,6])
y = torch.tensor([0,1,0,1,0,1])

vector_add_host(x, y, bs=4, kernel_fn=vector_add)
