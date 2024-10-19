#ifndef TRITON_CONVERSION_TRITONNPU_TO_LLVM_PATTERNS_TRITON_NPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONNPU_TO_LLVM_PATTERNS_TRITON_NPU_OP_TO_LLVM_H

#include "NPUTargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonNPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
// Some populate* functions have name collisions with the ones for GPUs.
namespace npu {

constexpr int patternBenefitDefault = 1;
constexpr int patternBenefitPrioritizeOverLLVMConversions = 10;
constexpr int patternBenefitClampOptimizedPattern = 20;
constexpr int patternBenefitConvertLayoutOptimizedPattern = 20;

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const npu::NPUTargetInfo &targetInfo,
                                 PatternBenefit benefit);

void populateControlFlowOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

void populateFuncOpConversionPattern(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

void populatePrintOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  const NPUTargetInfo &targetInfo,
                                  PatternBenefit benefit);

} // namespace npu
} // namespace triton
} // namespace mlir

#endif
