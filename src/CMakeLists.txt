target_sources(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_NAME}.cpp
)

# # anderson_device_dual_history
set(CURRENT_TARGET_NAME dual_history)

target_sources(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
    PRIVATE
    ${CURRENT_TARGET_NAME}.cu
)

if(ANDERSON_DEVICE_USE_CUDA)
    target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
        INTERFACE
        CUDA::cudart_static
        CUDA::cudart
        CUDA::cublas
        CUDA::culibos
    )

    if(EXISTS "${CUDAToolkit_BIN_DIR}/../samples/common/inc/helper_cuda.h")
        message(STATUS "Header helper_cuda.h is found in ${CUDAToolkit_BIN_DIR}/../samples/common/inc/")
        target_include_directories(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE ${CUDAToolkit_BIN_DIR}/../samples/common/inc/)
    else()
        message(STATUS "Header helper_cuda.h is not found in ${CUDAToolkit_BIN_DIR}/../samples/common/inc/. Fetch cuda-samples")

        FetchContent_Declare(cuda-samples
            GIT_REPOSITORY git@github.com:NVIDIA/cuda-samples.git
            GIT_TAG master
        )

        FetchContent_GetProperties(cuda-samples)

        if(NOT cuda-samples_POPULATED)
            FetchContent_Populate(cuda-samples)
            message(STATUS "use cuda-samples headers in ${cuda-samples_SOURCE_DIR}/Common")
        endif()

        target_include_directories(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE ${cuda-samples_SOURCE_DIR}/Common)
    endif()
else()
    find_package(hip)
    find_package(hipcub)
    find_package(hipblas)
    target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
        INTERFACE
        hip::host
        hip::device
        hip::hipcub
        roc::hipblas
    )
endif()

# # anderson_device_dual_history
set(CURRENT_TARGET_NAME uv_solver)

target_sources(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
    PRIVATE
    ${CURRENT_TARGET_NAME}.cu
)

if(ANDERSON_DEVICE_USE_CUDA)
    target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
        INTERFACE
        CUDA::cudart_static
        CUDA::cudart
        CUDA::cublas
        CUDA::culibos
        CUDA::cusolver
    )

    if(EXISTS "${CUDAToolkit_BIN_DIR}/../samples/common/inc/helper_cuda.h")
        message(STATUS "Header helper_cuda.h is found in ${CUDAToolkit_BIN_DIR}/../samples/common/inc/")
        target_include_directories(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE ${CUDAToolkit_BIN_DIR}/../samples/common/inc/)
    else()
        message(STATUS "Header helper_cuda.h is not found in ${CUDAToolkit_BIN_DIR}/../samples/common/inc/. Fetch cuda-samples")

        FetchContent_Declare(cuda-samples
            GIT_REPOSITORY git@github.com:NVIDIA/cuda-samples.git
            GIT_TAG master
        )

        FetchContent_GetProperties(cuda-samples)

        if(NOT cuda-samples_POPULATED)
            FetchContent_Populate(cuda-samples)
            message(STATUS "use cuda-samples headers in ${cuda-samples_SOURCE_DIR}/Common")
        endif()

        target_include_directories(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE ${cuda-samples_SOURCE_DIR}/Common)
    endif()
else()
    find_package(hip)
    find_package(hipcub)
    find_package(hipblas)
    find_package(LAPACK)
    target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
        INTERFACE
        hip::host
        hip::device
        hip::hipcub
        roc::hipblas
        LAPACK::LAPACK
    )
    target_compile_definitions(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
        PRIVATE
        ANDERSON_DEVICE_USING_LAPACK_HOST_SOLVER
    )
endif()