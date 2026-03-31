cmake_minimum_required(VERSION 3.3...4.0)

project(CDT_visualizer VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_AUTOMOC ON)

find_package(Qt5 COMPONENTS Core Widgets Gui REQUIRED)

option(CDT_USE_FIND_PACKAGE
    "If enabled prebuild CDT is consumed with find_package" OFF)

if(CDT_USE_FIND_PACKAGE)
    # add CDT as package
    find_package(CDT REQUIRED CONFIG)
elseif(NOT TARGET CDT::CDT)
    # add CDT as source (easier development)
    # Guard: skip if CDT was already added by a parent CMakeLists.txt
    add_subdirectory(../CDT CDT)
endif()

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE Qt5::Widgets)
target_link_libraries(${PROJECT_NAME} PRIVATE CDT::CDT)
