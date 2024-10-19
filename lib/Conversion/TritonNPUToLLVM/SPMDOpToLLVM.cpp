#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonNPUToLLVM/PatternTritonNPUOpToLLVM.h"
#include "triton/Conversion/TritonNPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::npu;

struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  explicit GetProgramIdOpConversion(LLVMTypeConverter &typeConverter,
                                    const NPUTargetInfo &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::GetProgramIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value programId = targetInfo.programId(
        rewriter, op->getLoc(), op->getParentOfType<LLVM::LLVMFuncOp>(),
        op.getAxisAsInt());
    rewriter.replaceOp(op, programId);
    return success();
  }

private:
  const NPUTargetInfo &targetInfo;
};

} // namespace

void mlir::triton::npu::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const NPUTargetInfo &targetInfo, PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, targetInfo, benefit);
}
