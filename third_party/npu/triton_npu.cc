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
  m.def("add_to_llvmir", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createConvertTritonNPUToLLVMPass());
  });
}

void init_triton_npu(py::module &&m) {
  auto passes = m.def_submodule("passes");
  init_triton_npu_passes_ttnpuir(passes.def_submodule("ttnpuir"));

  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::npu::TritonNPUDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });
}