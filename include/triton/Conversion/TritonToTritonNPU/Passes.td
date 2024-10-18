#ifndef TRITON_CONVERSION_TO_NPU_PASSES
#define TRITON_CONVERSION_TO_NPU_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonNPU: Pass<"convert-triton-to-tritonnpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonNPU";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonToTritonNPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::npu::TritonNPUDialect",
                             "mlir::triton::TritonDialect"];

   let options = [
   ];
}

#endif
