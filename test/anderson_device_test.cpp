// c++ headers
#include <type_traits>
#include <cmath>
#include <iomanip>
#include <memory>

// dependencies headers

// project headers
#include "anderson_device/dual_history.h"
#include "anderson_device/uv_solver.h"

// gtest headers
#include <gtest/gtest.h>

TEST(AndersonDevice, TrivialAdd)
{
    EXPECT_EQ(1 + 1, 2);
}
