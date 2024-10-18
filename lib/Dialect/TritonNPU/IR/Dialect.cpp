#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNPU/IR/Dialect.cpp.inc"
#include "triton/Dialect/TritonNPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::npu;

//===----------------------------------------------------------------------===//
// Attribute methods
//===----------------------------------------------------------------------===//
#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/TritonNPU/IR/TritonNPUAttrDefs.cpp.inc"

void TritonNPUDialect::initialize() {
  registerTypes();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/TritonNPU/IR/TritonNPUAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/TritonNPU/IR/Ops.cpp.inc"
#include "triton/Dialect/TritonNPU/IR/OpsEnums.cpp.inc"
      >();
}

// verify TritonNPU ops
LogicalResult TritonNPUDialect::verifyOperationAttribute(Operation *op,
                                                         NamedAttribute attr) {
  // TODO: fill this.
  return success();
}
