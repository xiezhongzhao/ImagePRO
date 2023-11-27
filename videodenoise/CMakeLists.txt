file(GLOB_RECURSE srcs CONFIGURE_DEPENDS src/*.cpp include/*.h)
add_library(videodenoise STATIC ${srcs})
target_include_directories(videodenoise PUBLIC include)
target_include_directories(videodenoise PUBLIC ../timer/include)







