#ifndef TRITONNPU_CONVERSION_TRITONNPUTOLLVM_TYPECONVERTER_H
#define TRITONNPU_CONVERSION_TRITONNPUTOLLVM_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/TritonNPU/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

class TritonNPUToLLVMTypeConverter : public LLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonNPUToLLVMTypeConverter(MLIRContext *ctx, LowerToLLVMOptions &option,
                               const DataLayoutAnalysis *analysis = nullptr);
};

#endif
