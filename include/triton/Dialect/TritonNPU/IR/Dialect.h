#ifndef TRITON_DIALECT_TRITONNPU_IR_DIALECT_H_
#define TRITON_DIALECT_TRITONNPU_IR_DIALECT_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

// TritonNPU depends on Triton
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonNPU/IR/Attributes.h"
#include "triton/Dialect/TritonNPU/IR/Dialect.h.inc"
#include "triton/Dialect/TritonNPU/IR/Types.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonNPU/IR/Ops.h.inc"

#endif // TRITON_DIALECT_TRITONNPU_IR_DIALECT_H_
