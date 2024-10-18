add_triton_library(TritonNPUTransforms

  DEPENDS
  TritonNPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonNPUIR
  MLIRTransformUtils
)
