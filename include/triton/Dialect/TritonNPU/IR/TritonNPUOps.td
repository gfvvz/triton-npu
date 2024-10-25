#ifndef TRITONNPU_OPS
#define TRITONNPU_OPS

include "triton/Dialect/TritonNPU/IR/TritonNPUDialect.td"
include "triton/Dialect/TritonNPU/IR/TritonNPUTypes.td"
include "triton/Dialect/TritonNPU/IR/TritonNPUAttrDefs.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class TTN_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonNPU_Dialect, mnemonic,
       !listconcat(traits, [])> {
}

def TTN_ExtractMemRefOp : TTN_Op<"extract_memref", [NoMemoryEffect]> {
  let summary = "Extract base memref from a block pointer";

  let description = [{
    Extract base memref from a block pointer. It covers whole base tensor memory,
    not only the block referenced. Base pointer, shape, and strides are used
    in the resulting memref. Offsets and block shape are ignored.

  }];

  let arguments = (ins TT_TensorPtr:$src);

  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

def TTN_ExtractIndicesOp : TTN_Op<"extract_indices", [NoMemoryEffect]> {
  let summary = "Extract indices from a block pointer.";

  let description = [{
    Extract indices that can be used to access the block using its base memref.
    Indices are supposed to be used for vector loads/stores with the base
    memref extracted from the same block pointer.
  }];

  let arguments = (ins TT_TensorPtr:$src);

  let results = (outs Variadic<Index>:$result);

  let builders = [
    OpBuilder<(ins "Value":$src)>
  ];

  let hasCanonicalizer = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif
