#ifndef TRITONNPU_CONVERSION_PASSES
#define TRITONNPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FuncOpToLLVM : Pass<"triton-npu-func-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert FuncOp to LLVM for NPU.";
    let description = [{

    }];
    let constructor = "mlir::triton::npu::createFuncOpToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::npu::TritonNPUDialect"];
}

def MemoryOpToLLVM : Pass<"triton-npu-memory-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton memory operations to LLVM for NPU.";
    let description = [{

    }];
    let constructor = "mlir::triton::npu::createMemoryOpToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::memref::MemRefDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::npu::TritonNPUDialect"];
}

def GetProgramIdOpToLLVM : Pass<"triton-npu-get-program-id-op-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Triton GetProgramId to LLVM for NPU.";
    let description = [{

    }];
    let constructor = "mlir::triton::npu::createGetProgramIdOpToLLVMPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect"];
}

#endif
