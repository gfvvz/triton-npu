add_triton_library(TritonToTritonNPU
    TritonNPUConversion.cpp
    TritonToTritonNPUPass.cpp

    DEPENDS
    TritonConversionToNPUPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonNPUIR
    TritonNPUTransforms
)
