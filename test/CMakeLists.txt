include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/58d77fa8070e8cec2dc1ed015d66b454c8d78850.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

include(GoogleTest)

# test info
set(TEST_NAME anderson_device)

if(ANDERSON_DEVICE_USE_CUDA)
    set_property(SOURCE ${TEST_NAME}_test.cpp PROPERTY LANGUAGE CUDA)
else()
endif()

add_executable(
    ${TEST_NAME}_test
    ${TEST_NAME}_test.cpp
)
target_link_libraries(${TEST_NAME}_test
    PRIVATE
    ${PROJECT_NAME}
    qutility
    qutility_device
    device_api
    gtest_main
)
gtest_discover_tests(${TEST_NAME}_test)
