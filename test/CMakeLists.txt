file(GLOB_RECURSE srcs CONFIGURE_DEPENDS *.cpp *.h)

add_executable(test ${srcs})

target_link_libraries(test PUBLIC ${thirdLibs})
target_link_libraries(test PUBLIC timer)
target_link_libraries(test PUBLIC videodenoise)
target_link_libraries(test PUBLIC contrast)
