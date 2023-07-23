# # anderson_device_dual_history
set(CURRENT_TARGET_NAME dual_history)
add_library(${PROJECT_NAME}_${CURRENT_TARGET_NAME} STATIC)
target_include_directories(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/)

target_sources(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PUBLIC
    ${PROJECT_NAME}.h
    ${CURRENT_TARGET_NAME}.h
)

target_sources(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE
    ${CURRENT_TARGET_NAME}.cuh
)

target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PUBLIC
    qutility
    qutility_device_def
    qutility_device_event
    qutility_device_workspace
    qutility_device_dual_field
    device_api
)

target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE
    qutility_device_sync_grid
)

if(ANDERSON_DEVICE_USE_CUDA)
else()
    target_compile_definitions(${PROJECT_NAME}_${CURRENT_TARGET_NAME}
        PRIVATE
        ANDERSON_DEVICE_USING_HIP_DYNAMIC_SHARED_INSTEAD_OF_EXTERN_SHARED
    )
endif()

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_${CURRENT_TARGET_NAME})

# # anderson_device_dual_history
set(CURRENT_TARGET_NAME uv_solver)
add_library(${PROJECT_NAME}_${CURRENT_TARGET_NAME} STATIC)
target_include_directories(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include/)

target_sources(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PUBLIC
    ${PROJECT_NAME}.h
    ${CURRENT_TARGET_NAME}.h
)

target_sources(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PRIVATE
    ${CURRENT_TARGET_NAME}.cuh
)

target_link_libraries(${PROJECT_NAME}_${CURRENT_TARGET_NAME} PUBLIC
    qutility
    qutility_device_def
    qutility_device_event
    qutility_device_workspace
    qutility_device_dual_field
    device_api
)

target_link_libraries(${PROJECT_NAME} ${PROJECT_NAME}_${CURRENT_TARGET_NAME})

