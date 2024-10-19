#include "triton/Conversion/TritonNPUToLLVM/PatternTritonNPUOpToLLVM.h"
#include "triton/Conversion/TritonNPUToLLVM/Utility.h"
#include "llvm/Support/ErrorHandling.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

struct ReturnOpConversion : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (funcOp->hasAttr("npu.kernel")) {
      if (op.getNumOperands() > 0) {
        return rewriter.notifyMatchFailure(
            op, "Kernel functions do not support return with operands");
      }
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                  op->getAttrs());
    } else {
      llvm_unreachable("Not implemented");
    }
    return success();
  }
};

} // namespace

void mlir::triton::npu::populateControlFlowOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<ReturnOpConversion>(typeConverter, benefit);
}
