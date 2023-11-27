set(third_include ${CMAKE_CURRENT_SOURCE_DIR}/third_party/include/)
message(STATUS "third_include: ${third_include}")
include_directories(${third_include})
#
set(thirdLibs ${CMAKE_CURRENT_SOURCE_DIR}/third_party/lib/*)
file(GLOB thirdLibs ${thirdLibs})
message(STATUS "thirdLibs: ${thirdLibs}")



