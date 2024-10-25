#include "npu/include/TritonNPUToLLVM/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace triton {
namespace npu {

void tritonNPUToLLVMPipelineBuilder(OpPassManager &pm) {
  pm.addPass(mlir::triton::npu::createFuncOpToLLVMPass());
  pm.addPass(mlir::triton::npu::createGetProgramIdOpToLLVMPass());
  pm.addPass(mlir::triton::npu::createMemoryOpToLLVMPass());
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void registerTritonNPUToLLVMPipeline() {
  PassPipelineRegistration<>("triton-npu-to-llvmir",
                             "TritonNPU to LLVM conversion pipeline.",
                             tritonNPUToLLVMPipelineBuilder);
}

} // namespace npu
} // namespace triton
} // namespace mlir
