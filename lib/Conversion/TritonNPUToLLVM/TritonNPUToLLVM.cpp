#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonNPUToLLVM/Passes.h"
#include "triton/Conversion/TritonNPUToLLVM/PatternTritonNPUOpToLLVM.h"
#include "triton/Conversion/TritonNPUToLLVM/TypeConverter.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonNPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONNPUTOLLVM
#include "triton/Conversion/TritonNPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<index::IndexDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::npu::TritonNPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonNPUToLLVM
    : public triton::impl::ConvertTritonNPUToLLVMBase<ConvertTritonNPUToLLVM> {
  using ConvertTritonNPUToLLVMBase<
      ConvertTritonNPUToLLVM>::ConvertTritonNPUToLLVMBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::npu::TritonNPUDialect, LLVM::LLVMDialect>();
  }

  ConvertTritonNPUToLLVM() : ConvertTritonNPUToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonNPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      TritonNPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::npu::populateFuncOpConversionPattern(
          typeConverter, funcPatterns,
          mlir::triton::npu::patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    RewritePatternSet patterns(context);
    mlir::triton::npu::NPUTargetInfo targetInfo;
    int benefit =
        mlir::triton::npu::patternBenefitPrioritizeOverLLVMConversions;
    mlir::triton::npu::populateControlFlowOpToLLVMPattern(typeConverter,
                                                          patterns, benefit);
    mlir::triton::npu::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::npu::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                   targetInfo, benefit);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonNPUToLLVMPass() {
  return std::make_unique<ConvertTritonNPUToLLVM>();
}

} // namespace triton
} // namespace mlir
