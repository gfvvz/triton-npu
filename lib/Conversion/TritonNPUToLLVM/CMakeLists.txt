add_triton_library(TritonNPUToLLVM
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
