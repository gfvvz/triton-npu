#ifndef TRITON_CONVERSION_TRITON_TO_TRITONNPU_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITON_TO_TRITONNPU_TYPECONVERTER_H

#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

class TritonToTritonNPUTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  TritonToTritonNPUTypeConverter();

  Type convertTritonPointerType(triton::PointerType type);
};

#endif
