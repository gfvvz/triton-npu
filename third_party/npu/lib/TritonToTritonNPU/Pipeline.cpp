#include "npu/include/TritonToTritonNPU/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace triton {
namespace npu {

void tritonToTritonNPUPipelineBuilder(OpPassManager &pm) {
  pm.addPass(mlir::triton::npu::createConvertMemoryOps());
  pm.addPass(mlir::triton::npu::createConvertPtrOps());
  pm.addPass(mlir::triton::npu::createConvertElementwiseOps());
  pm.addPass(mlir::triton::npu::createConvertDotOp());
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void registerTritonToTritonNPUPipeline() {
  PassPipelineRegistration<>("triton-to-triton-npu",
                             "Triton to TritonNPU conversion pipeline.",
                             tritonToTritonNPUPipelineBuilder);
}

} // namespace npu
} // namespace triton
} // namespace mlir
