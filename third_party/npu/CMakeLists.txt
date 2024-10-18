if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonNPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_npu.cc LINK_LIBS TritonNPUToLLVM)
endif()