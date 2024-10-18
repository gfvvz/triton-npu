#ifndef TRITON_CONVERSION_TO_NPU_PASSES_H
#define TRITON_CONVERSION_TO_NPU_PASSES_H

#include "triton/Conversion/TritonToTritonNPU/TritonToTritonNPUPass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/TritonToTritonNPU/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
