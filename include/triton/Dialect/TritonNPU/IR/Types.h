#ifndef TRITONNPU_IR_TYPES_H_
#define TRITONNPU_IR_TYPES_H_

#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/TritonNPU/IR/Types.h.inc"

#endif // TRITON_IR_TYPES_H_
