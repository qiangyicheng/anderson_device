cmake_minimum_required(VERSION 3.14)
include(CheckLanguage)
check_language(CUDA)
set(ANDERSON_DEVICE_USE_CUDA)

if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA is found. This project will target CUDA platform directly")
    set(ANDERSON_DEVICE_USE_CUDA True)
    project(anderson_device LANGUAGES CUDA CXX)
    find_package(CUDAToolkit REQUIRED)
else()
    message(STATUS "CUDA is not found. This prpject will try HIP platform")
    set(ANDERSON_DEVICE_USE_CUDA False)
    project(anderson_device LANGUAGES CXX)
    find_package(hip REQUIRED)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)
FetchContent_Declare(device_api
    GIT_REPOSITORY git@github.com:qiangyicheng/device_api.git
)
FetchContent_GetProperties(device_api)

if(NOT device_api_POPULATED)
    FetchContent_Populate(device_api)
    add_subdirectory(${device_api_SOURCE_DIR} ${device_api_BINARY_DIR})
endif()

FetchContent_Declare(qutility
    GIT_REPOSITORY git@github.com:qiangyicheng/qutility_cmake.git
    GIT_TAG main
)

if(NOT qutility_POPULATED)
    FetchContent_Populate(qutility)
    add_subdirectory(${qutility_SOURCE_DIR} ${qutility_BINARY_DIR})
endif()

FetchContent_Declare(qutility_device
    GIT_REPOSITORY git@github.com:qiangyicheng/qutility_device.git
    GIT_TAG main
)

if(NOT qutility_device_POPULATED)
    FetchContent_Populate(qutility_device)
    add_subdirectory(${qutility_device_SOURCE_DIR} ${qutility_device_BINARY_DIR})
endif()

add_library(${PROJECT_NAME} STATIC)
target_include_directories(${PROJECT_NAME} PRIVATE include/)


add_subdirectory(include)
add_subdirectory(src)

enable_testing()
add_subdirectory(test)
