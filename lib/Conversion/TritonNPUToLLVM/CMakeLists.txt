add_triton_library(TritonNPUToLLVM
    ControlFlowOpToLLVM.cpp
    NPUTargetInfo.cpp
    FuncOpToLLVM.cpp
    PrintOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    TypeConverter.cpp
    TritonNPUToLLVM.cpp

    DEPENDS
    TritonNPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    TritonAnalysis
    TritonIR
    TritonNPUIR
    TritonNPUTransforms
)
