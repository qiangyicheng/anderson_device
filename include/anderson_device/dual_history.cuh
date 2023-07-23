#pragma once

#include "device_api/device_api_cuda_device.h"
#include "device_api/device_api_cub.h"

#include "qutility_device/def.h"
#include "qutility_device/sync_grid.cuh"


namespace anderson
{
    namespace device
    {
        namespace kernel
        {
            template <size_t ThreadsPerBlock>
            __global__ void andersonDualHistoryCalcUV(
                size_t single_size, size_t nhist,
                const double *new_diff,
                double *inner, double *UV, const int *mask,
                double *working,
                size_t N_ava, size_t pos_push)
            {
                QUTILITY_DEVICE_SYNC_GRID_PREPARE;
                int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                int grid_size = gridDim.x * blockDim.x;
                typedef dapi_cub::BlockReduce<double, ThreadsPerBlock, dapi_cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduceT;
                {
                    __shared__ typename BlockReduceT::TempStorage temp_storage;
                    double data = 0.0;
                    for (int itr = thread_rank; itr < single_size; itr += grid_size)
                        data += new_diff[itr] * new_diff[itr];
                    dapi___syncthreads();
                    double agg = BlockReduceT(temp_storage).Sum(data);
                    if (threadIdx.x == 0)
                        working[blockIdx.x] = agg;
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                    if (thread_rank == 0)
                    {
                        for (int itr_block = 1; itr_block < gridDim.x; ++itr_block)
                            agg += working[itr_block];
                        inner[nhist * (nhist + 1)] = agg;
                    }
                    QUTILITY_DEVICE_SYNC_GRID_SYNC((unsigned int *)(working));
                }
                {
                    for (size_t itr_block = blockIdx.x; itr_block < nhist; itr_block += gridDim.x)
                    {
                        double new_new = inner[nhist * (nhist + 1)];
                        double new_old = inner[nhist * nhist + itr_block];
                        if (threadIdx.x == 0)
                        {
                            UV[nhist * nhist + itr_block] = itr_block < N_ava ? (new_new - new_old) : 0;
                        }
                        for (size_t itr_thread = threadIdx.x; itr_thread < nhist; itr_thread += blockDim.x)
                        {
                            double new_old_2 = inner[nhist * nhist + itr_thread];
                            UV[itr_block * nhist + itr_thread] =
                                (itr_block < N_ava &&
                                 itr_thread < N_ava &&
                                 mask[itr_block] >= 0 &&
                                 mask[itr_thread] >= 0)
                                    ? (inner[itr_block * nhist + itr_thread] + new_new - new_old - new_old_2)
                                    : (itr_thread == itr_block) * 1.;
                            if (itr_block == pos_push && itr_thread == pos_push)
                            {
                                inner[itr_block * nhist + itr_thread] = new_new;
                            }
                            else if (itr_block == pos_push)
                            {
                                inner[itr_block * nhist + itr_thread] = new_old_2;
                            }
                            else if (itr_thread == pos_push)
                            {
                                inner[itr_block * nhist + itr_thread] = new_old;
                            }
                        }
                    }
                }
            }

            __global__ void andersonDualHistoryCalcUVInvalid(
                size_t nhist,
                double *inner, double *UV,
                size_t pos_push)
            {
                {
                    for (size_t itr_block = blockIdx.x; itr_block < nhist; itr_block += gridDim.x)
                    {
                        if (threadIdx.x == 0)
                        {
                            inner[nhist * nhist + itr_block] = 0;
                            UV[nhist * nhist + itr_block] = 0;
                        }
                        for (size_t itr_thread = threadIdx.x; itr_thread < nhist; itr_thread += blockDim.x)
                        {
                            UV[itr_block * nhist + itr_thread] = (itr_thread == itr_block) * 1.;
                            if (itr_block == pos_push || itr_thread == pos_push)
                            {
                                inner[itr_block * nhist + itr_thread] = 0;
                            }
                        }
                    }
                }
            }

            __global__ void andersonDualHistoryMix(
                size_t single_size, size_t nhist,
                double *hist, double *hist_diff,
                double *new_field, const double *new_field_diff,
                const double *coef,
                int *mask,
                size_t N_ava, size_t pos_push,
                double acceptance)
            {
                int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                int grid_size = gridDim.x * blockDim.x;
#ifdef ANDERSON_DEVICE_USING_HIP_DYNAMIC_SHARED_INSTEAD_OF_EXTERN_SHARED
                HIP_DYNAMIC_SHARED(double, coef_s);
#else
                extern __shared__ double coef_s[];
#endif
                for (size_t itr = threadIdx.x; itr <= nhist; itr += blockDim.x)
                {
                    coef_s[itr] = coef[itr];
                }
                dapi___syncthreads();
                for (size_t itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    double save_val[2] = {new_field[itr], new_field_diff[itr]};
                    double new_val[2] = {coef_s[nhist] * save_val[0], coef_s[nhist] * save_val[1]};
                    for (size_t itr_ava = 0; itr_ava < N_ava; ++itr_ava)
                    {
                        if (dapi_fabs(coef_s[itr_ava]) < 1e-10)
                            continue;
                        new_val[0] += coef_s[itr_ava] * hist[itr_ava * single_size + itr];
                        new_val[1] += coef_s[itr_ava] * hist_diff[itr_ava * single_size + itr];
                    }
                    hist[pos_push * single_size + itr] = save_val[0];
                    hist_diff[pos_push * single_size + itr] = save_val[1];
                    new_field[itr] = new_val[0] + acceptance * new_val[1];
                }

                if (thread_rank == 0)
                    mask[pos_push] = N_ava;
            }

            __global__ void andersonDualHistoryMixSimple(
                size_t single_size,
                double *hist, double *hist_diff,
                double *new_field, const double *new_field_diff,
                int *mask,
                size_t pos_push,
                double acceptance)
            {
                int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                int grid_size = gridDim.x * blockDim.x;
                for (size_t itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    hist[pos_push * single_size + itr] = new_field[itr];
                    hist_diff[pos_push * single_size + itr] = new_field_diff[itr];
                    new_field[itr] += acceptance * new_field_diff[itr];
                }

                if (thread_rank == 0)
                    mask[pos_push] = 0;
            }

            __global__ void andersonDualHistoryDuplicate(
                size_t single_size,
                double *hist, double *hist_diff,
                const double *new_field,
                int *mask,
                size_t pos_push)
            {
                int thread_rank = blockIdx.x * blockDim.x + threadIdx.x;
                int grid_size = gridDim.x * blockDim.x;
                for (size_t itr = thread_rank; itr < single_size; itr += grid_size)
                {
                    hist[pos_push * single_size + itr] = new_field[itr];
                    hist_diff[pos_push * single_size + itr] = 0;
                }

                if (thread_rank == 0)
                    mask[pos_push] = -1;
            }
        }
    }
}