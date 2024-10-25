add_triton_library(TritonToTritonNPU
    ConvertDotOp.cpp
    ConvertElementwiseOps.cpp
    ConvertMemoryOps.cpp
    ConvertPtrOps.cpp
    Pipeline.cpp
    TypeConverter.cpp

    DEPENDS
    TritonToTritonNPUPassIncGen

    LINK_LIBS PUBLIC
    TritonNPUIR
    MLIRVectorDialect
)
