#ifndef TRITONNPU_ATTRDEFS
#define TRITONNPU_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "triton/Dialect/TritonNPU/IR/TritonNPUDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"

//===----------------------------------------------------------------------===//
// TritonNPU Attribute Definitions
//===----------------------------------------------------------------------===//
def TritonNPU_AttrTrait : AttrInterface<"TritonNPU_AttrTrait"> {
  let cppNamespace = "::mlir::triton::npu";
}

class TritonNPU_Attr<string name, string attrMnemonic, list<Trait> traits = [],
                     Dialect dialect = TritonNPU_Dialect,
                     string baseCppClass = "::mlir::Attribute">
  : AttrDef<dialect, name, !listconcat([TritonNPU_AttrTrait], traits), baseCppClass> {

  let description = [{TritonNPU attr.}];
  let attrName = "triton.npu." # attrMnemonic;
}

#endif
