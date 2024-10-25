#include "TritonNPUToLLVM/Passes.h"
#include "TritonToTritonNPU/Passes.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Conversion/TritonNPUToLLVM/Passes.h"
#include "triton/Dialect/TritonNPU/IR/Dialect.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/TargetSelect.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

void init_triton_npu_passes_ttnpuir(py::module &&m) {
  using namespace mlir::triton;
  // m.def("add_to_llvmir", [](mlir::PassManager &pm) {
  //   pm.addPass(mlir::triton::createConvertTritonNPUToLLVMPass());
  // });
  m.def("add_triton_to_triton_npu_pipeline", [](mlir::PassManager &pm) {
    mlir::triton::npu::tritonToTritonNPUPipelineBuilder(pm);
  });
  m.def("add_triton_npu_to_llvmir_pipeline", [](mlir::PassManager &pm) {
    mlir::triton::npu::tritonNPUToLLVMPipelineBuilder(pm);
  });
  m.def("add_vector_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertVectorToLLVMPass());
  });
  m.def("add_memref_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  });
  m.def("add_math_to_libm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertMathToLibmPass());
  });
  m.def("add_func_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::createConvertFuncToLLVMPass());
  });
}

void init_triton_npu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_npu_passes_ttnpuir(passes.def_submodule("ttnpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::npu::TritonNPUDialect,
                    mlir::vector::VectorDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def("find_kernel_names", [](mlir::ModuleOp &mod) {
    std::vector<std::string> res;
    mod.walk([&](mlir::FunctionOpInterface funcOp) {
      if (funcOp.getVisibility() == mlir::SymbolTable::Visibility::Public)
        res.push_back(funcOp.getName().str());
    });
    return res;
  });
}