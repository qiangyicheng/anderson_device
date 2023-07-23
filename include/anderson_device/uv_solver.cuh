#pragma once

#include "qutility/traits.h"

#include "device_api/device_api_cuda_device.h"
#include "device_api/device_api_cub.h"

#include "qutility_device/def.h"

namespace anderson
{
    namespace device
    {
        namespace kernel
        {
            QUTILITY_DEVICE_DEVICE_ACCESSIBLE QUTILITY_DEVICE_FORCE_INLINE double weightedAdd(size_t index) { return 0; }

            template <typename... PointerWeightTypePair>
            QUTILITY_DEVICE_DEVICE_ACCESSIBLE QUTILITY_DEVICE_FORCE_INLINE double weightedAdd(size_t index, double *FirstPtr, double FirstWeight, PointerWeightTypePair... Rest)
            {
                using qutility::traits::is_type_T;
                static_assert(
                    qutility::traits::is_correct_list_of_n<2, is_type_T<double *>, is_type_T<double>, double *, double, PointerWeightTypePair...>::value,
                    "Parameters must be pairs of double* and double");
                return FirstWeight * FirstPtr[index] + weightedAdd(index, Rest...);
            }

            template <typename doublePtrT, typename doubleT, typename... PointerWeightTypePair>
            __global__ void combineUV(size_t n_hist, doublePtrT FirstPtr, doubleT FirstWeight, PointerWeightTypePair... Rest)
            {
                static_assert(std::is_same<doublePtrT, double *>::value);
                static_assert(std::is_same<doubleT, double>::value);
                int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                int grid_size = gridDim.x * blockDim.x;
                for (int itr = thread_rank; itr < n_hist * (n_hist + 1); itr += grid_size)
                {
                    FirstPtr[itr] *= FirstWeight;
                    FirstPtr[itr] += weightedAdd(itr, Rest...);
                }
            }

            QUTILITY_DEVICE_DEVICE_ACCESSIBLE QUTILITY_DEVICE_FORCE_INLINE bool isTooOld(size_t nhist, size_t pos, size_t pos_push, size_t max_age)
            {
                return (pos < pos_push ? pos_push - pos : (pos_push + nhist) - pos) > max_age;
            }

            inline __global__ void maskOldUV(size_t nhist, double *UV, size_t pos_push, size_t max_age)
            {
                for (size_t itr_block = blockIdx.x; itr_block < nhist; itr_block += gridDim.x)
                {
                    if (threadIdx.x == 0)
                    {
                        if (isTooOld(nhist, itr_block, pos_push, max_age))
                            UV[nhist * nhist + itr_block] = 0;
                    }
                    for (size_t itr_thread = threadIdx.x; itr_thread < nhist; itr_thread += blockDim.x)
                    {
                        if (isTooOld(nhist, itr_thread, pos_push, max_age) || isTooOld(nhist, itr_block, pos_push, max_age))
                        {
                            UV[itr_block * nhist + itr_thread] = (itr_block == itr_thread) * 1.;
                        }
                    }
                }
            }

            template <size_t ThreadsPerBlock>
            __global__ void modifyCoef(size_t nhist, double *coef)
            {
                typedef dapi_cub::BlockReduce<double, ThreadsPerBlock, dapi_cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceT;
                if (blockIdx.x == 0)
                {
                    __shared__ typename BlockReduceT::TempStorage temp_storage;
                    double data = 0;
                    for (int itr = threadIdx.x; itr < nhist; itr += blockDim.x)
                        data += coef[itr];
                    dapi___syncthreads();
                    double agg = BlockReduceT(temp_storage).Sum(data);
                    if (threadIdx.x == 0)
                        coef[nhist] = 1 - agg;
                }
            }

        } // namespace kernel

    }
}