add_triton_library(TritonNPUToLLVM
    FuncOpToLLVM.cpp
    GetProgramIdOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    Pipeline.cpp
    TypeConverter.cpp

    DEPENDS
    TritonNPUToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRVectorToLLVMPass
)
