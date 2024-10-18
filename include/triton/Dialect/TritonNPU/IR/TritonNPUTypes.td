#ifndef TRITONNPU_TYPES
#define TRITONNPU_TYPES

include "triton/Dialect/TritonNPU/IR/TritonNPUDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Triton NPU Specific Types
class TTN_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonNPU_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def TTN_TokenType : TTN_TypeDef<"Token", "token"> {
  let parameters = (ins "int32_t":$type);

  let builders = [
    TypeBuilder<(ins "unsigned":$type), [{
      return $_get($_ctxt, type);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
}

#endif
