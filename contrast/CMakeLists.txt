file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(contrast STATIC ${srcs})
target_include_directories(contrast PUBLIC include)








