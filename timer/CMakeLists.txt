file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(timer STATIC ${srcs})
target_include_directories(timer PUBLIC include)








