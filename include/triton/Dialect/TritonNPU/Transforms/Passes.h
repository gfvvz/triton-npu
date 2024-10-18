#ifndef TRITON_DIALECT_TRITONNPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONNPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {
namespace npu {} // namespace npu
} // namespace triton

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/TritonNPU/Transforms/Passes.h.inc"

} // namespace mlir
#endif
