#ifndef TRITONTOTRITONNPU_CONVERSION_PASSES_H
#define TRITONTOTRITONNPU_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace npu {

#define GEN_PASS_DECL
#include "npu/include/TritonToTritonNPU/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertElementwiseOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertMemoryOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertPtrOps();
std::unique_ptr<OperationPass<ModuleOp>> createConvertDotOp();

void tritonToTritonNPUPipelineBuilder(OpPassManager &pm);
void registerTritonToTritonNPUPipeline();

#define GEN_PASS_REGISTRATION
#include "npu/include/TritonToTritonNPU/Passes.h.inc"

} // namespace npu
} // namespace triton

} // namespace mlir

#endif
