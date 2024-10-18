add_triton_library(TritonNPUIR
  Dialect.cpp
  Types.cpp

  DEPENDS
  TritonNPUTableGen
  TritonNPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
