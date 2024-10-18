#ifndef TRITONNPU_DIALECT
#define TRITONNPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonNPU_Dialect : Dialect {
  let name = "triton_npu";

  let cppNamespace = "::mlir::triton::npu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton NPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif
