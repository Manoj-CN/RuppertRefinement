cmake_minimum_required(VERSION 3.3...4.0)

project(CDT_root)

add_subdirectory(CDT)

option(CDT_BUILD_VISUALIZER
    "Build the CDT visualizer (requires Qt5)"
    OFF)

if(CDT_BUILD_VISUALIZER)
    add_subdirectory(visualizer)
endif()
