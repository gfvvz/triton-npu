#ifndef TRITONTOTRITONNPU_CONVERSION_PASSES
#define TRITONTOTRITONNPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertMemoryOps : Pass<"triton-npu-convert-memory-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory ops.";
    let description = [{

    }];
    let constructor = "mlir::triton::npu::createConvertMemoryOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::npu::TritonNPUDialect"];
}

def ConvertElementwiseOps : Pass<"triton-npu-convert-elementwise-ops", "mlir::ModuleOp"> {
    let summary = "Convert elementwise ops.";
    let description = [{

    }];
    let constructor = "mlir::triton::npu::createConvertElementwiseOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::npu::TritonNPUDialect"];
}

def ConvertPtrOps : Pass<"triton-npu-convert-ptr-ops", "mlir::ModuleOp"> {
    let summary = "Convert Triton ops related to pointer arithmetics.";
    let description = [{

    }];
    let constructor = "mlir::triton::npu::createConvertPtrOps()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::npu::TritonNPUDialect"];
}

def ConvertDotOp : Pass<"triton-npu-convert-dot-op", "mlir::ModuleOp"> {
    let summary = "Convert Triton DotOp.";
    let description = [{

    }];
    let constructor = "mlir::triton::npu::createConvertDotOp()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::vector::VectorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::npu::TritonNPUDialect"];
}

#endif
