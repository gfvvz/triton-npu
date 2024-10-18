#ifndef TRITON_CONVERSION_TRITONTOTRITONNPU_TRITONTOTRITONNPUPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONNPU_TRITONTOTRITONNPUPASS_H

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonNPUPass();

} // namespace triton
} // namespace mlir

#endif
