#ifndef TRITONNPU_CONVERSION_PASSES
#define TRITONNPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonNPUToLLVM : Pass<"convert-triton-npu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonNPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonNPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::npu::TritonNPUDialect",
                             "mlir::triton::TritonDialect"];

    let options = [
    ];
}

#endif
