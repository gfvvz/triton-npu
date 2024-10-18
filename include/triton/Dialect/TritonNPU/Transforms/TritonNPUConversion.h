//===----------------------------------------------------------------------===//
//
// Defines utilities to use while converting to the TritonNPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_TRITONNPU_TRANSFORMS_TRITONNPUCONVERSION_H_
#define TRITON_DIALECT_TRITONNPU_TRANSFORMS_TRITONNPUCONVERSION_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

class TritonNPUTypeConverter : public TypeConverter {
public:
  TritonNPUTypeConverter(MLIRContext *context);

private:
  MLIRContext *context;
};

class TritonNPUConversionTarget : public ConversionTarget {

public:
  explicit TritonNPUConversionTarget(MLIRContext &ctx,
                                     TritonNPUTypeConverter &typeConverter);
};

} // namespace mlir

#endif // TRITON_DIALECT_TRITONNPU_TRANSFORMS_TRITONNPUCONVERSION_H_
