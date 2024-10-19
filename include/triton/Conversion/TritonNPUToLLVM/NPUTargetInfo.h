#ifndef TRITON_CONVERSION_TRITONNPU_TO_LLVM_TARGETINFOBASE_H
#define TRITON_CONVERSION_TRITONNPU_TO_LLVM_TARGETINFOBASE_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"

namespace mlir::triton::npu {
class NPUTargetInfo {
public:
  // Note: we may revisit for different NPU ISAs like AVX and Neon.
  NPUTargetInfo() {}

  Value programId(ConversionPatternRewriter &rewriter, Location loc,
                  LLVM::LLVMFuncOp funcOp, int axis) const;

  void printf(ConversionPatternRewriter &rewriter, Value formatStrStart,
              int formatStrByteCount, ValueRange args) const;

  ~NPUTargetInfo() {}
};
} // namespace mlir::triton::npu
#endif // TRITON_CONVERSION_TRITONNPU_TO_LLVM_TARGETINFOBASE_H
