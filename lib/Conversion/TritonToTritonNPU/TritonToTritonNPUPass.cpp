#include "triton/Conversion/TritonToTritonNPU/TritonToTritonNPUPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNPU/IR/Dialect.h"
#include "triton/Dialect/TritonNPU/Transforms/TritonNPUConversion.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonToTritonNPU/Passes.h.inc"

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::npu;

class ConvertTritonToTritonNPU
    : public ConvertTritonToTritonNPUBase<ConvertTritonToTritonNPU> {
public:
  ConvertTritonToTritonNPU() = default;

  void runOnOperation() override {
    // TODO:
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonNPUPass() {
  return std::make_unique<::ConvertTritonToTritonNPU>();
}
