message(STATUS "Setting up examples")

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/
  DESTINATION examples
  # COMPONENT examples
  FILES_MATCHING
  #
  # Exclude folder structures: doc, doxygen, CMakeFiles,...
  #
  PATTERN "CMakeFiles*" EXCLUDE
  PATTERN "doc*" EXCLUDE
  PATTERN "build*" EXCLUDE
  #
  # Glob Includes:
  #
  PATTERN "*.cc"
  PATTERN "*.h"
  PATTERN "*.prm"
  PATTERN "*.inp"
  PATTERN "*.output"
  #
  # All CMakeLists.txt except the one in examples/
  #
  PATTERN "heat-equation/CMakeLists.txt"
  PATTERN "entanglement/CMakeLists.txt"
  #
  # Special files:
  #
  PATTERN "grid.msh"
  )

MESSAGE(STATUS "Setting up examples - Done")
