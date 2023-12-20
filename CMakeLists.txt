cmake_minimum_required(VERSION 3.21.1)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAEK_BUILD_TYPE Release)
endif()

#add_definitions(-DORIG)
add_definitions(-DSIMD128) # -DORIG -DSSE
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake;${CMAKE_MODULE_PATH}")

project(ImagePRO LANGUAGES CXX)

# for Opencv
include(${PROJECT_SOURCE_DIR}/cmake/third_party.cmake)

add_subdirectory(videodenoise)
add_subdirectory(contrast)
add_subdirectory(timer)
add_subdirectory(test)





