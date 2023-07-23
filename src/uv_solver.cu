#include <string>
#include <string_view>

#include "anderson_device/dual_history.h"
#include "anderson_device/uv_solver.h"
#include "anderson_device/uv_solver.cuh"

#ifdef ANDERSON_DEVICE_USING_LAPACK_HOST_SOLVER
#include <lapacke/lapacke_config.h>
#include <lapacke/lapacke_mangling.h>
#include <lapacke/lapacke_utils.h>
#include <lapacke/lapacke.h>
#else
#include <cusolver_common.h>
#include <cusolverDn.h>
#endif // ANDERSON_DEVICE_USING_LAPACK_HOST_SOLVER

#include "device_api/device_api_helper.h"
#include "qutility/array_wrapper/array_wrapper_gpu.h"

namespace anderson
{
    namespace device
    {
        using namespace std::literals;

        UVSolver::UVSolver(
            int device,
            size_t nhist,
            const std::shared_ptr<WorkspaceT> &workspace) : StreamEventHelper(device),
                                                            nhist_(nhist),
                                                            workspace_(workspace),
                                                            coef_ptr_(std::make_unique<ArrayT>(0, (nhist + 1), device)),
                                                            data_ptr_(std::make_unique<DataArrayT>(0, (nhist + 1), device))
        {
#ifndef ANDERSON_DEVICE_USING_LAPACK_HOST_SOLVER
            set_device();
            dapi_checkCudaErrors(cusolverDnCreate((cusolverDnHandle_t *)(&lapack_solver_)));
            dapi_checkCudaErrors(cusolverDnSetStream((cusolverDnHandle_t)lapack_solver_, stream_));
            dapi_checkCudaErrors(cusolverDnDDgesv_bufferSize((cusolverDnHandle_t)lapack_solver_, nhist_, 1, nullptr, nhist_, nullptr, nullptr, nhist_, nullptr, nhist_, nullptr, &working_size_));
            if (working_size_ > workspace_->size_in_bytes())
            {
                qutility::message::exit_with_message("UVSolver Error: not enough workspace. The given workspace has size of "s + std::to_string(workspace->size_in_bytes()) + ", while " + std::to_string(max_threads_per_block_) + " on device "s + std::to_string(working_size_) + " is reauired.", __FILE__, __LINE__);
            }
#else
            // fall back to lapacke on host.
            // no initialization required
#endif
        }

        UVSolver::~UVSolver()
        {
#ifndef ANDERSON_DEVICE_USING_LAPACK_HOST_SOLVER
            set_device();
            dapi_checkCudaErrors(cusolverDnDestroy((cusolverDnHandle_t)lapack_solver_));
#else
            // fall back to lapacke on host.
            // no cleanning required
#endif
        }

        double *UVSolver::to_raw_data(DualHistory &dh) { return dh.get_uv(); }
        double UVSolver::to_raw_data(double &val) { return val; }
        dapi_cudaStream_t UVSolver::get_stream(DualHistory &dh) { return dh.stream_; }
        dapi_cudaStream_t UVSolver::get_stream(double &val) { return nullptr; }

        void UVSolver::mask_old_uv(double *uv, std::size_t current_pos, std::size_t using_history_count)
        {
            launch_kernel<threads_per_block_>(kernel::maskOldUV, {nhist_, uv, current_pos, using_history_count}, 0);
        }

        void UVSolver::solve_eigen_system(double *uv)
        {
            ValT *U = uv;
            ValT *V = U + nhist_ * nhist_;

#ifndef ANDERSON_DEVICE_USING_LAPACK_HOST_SOLVER
            this_wait_other(workspace_->stream_);

            int *ipiv = *data_ptr_;
            int *d_info = *data_ptr_ + nhist_;
            dapi_checkCudaErrors(cusolverDnDDgesv(
                (cusolverDnHandle_t)lapack_solver_, nhist_, 1,
                U, nhist_,
                ipiv,
                V, nhist_,
                *coef_ptr_, nhist_,
                (ValT *)*workspace_, working_size_,
                &iter_, d_info));

            other_wait_this(workspace_->stream_);

#else
            // fallback to host host routines

            using qutility::array_wrapper::DArrayDDR;
            using qutility::array_wrapper::DArrayDDRPinned;
            DArrayDDRPinned<ValT> first_uv_host(nhist_ * (nhist_ + 1));
            DArrayDDR<int> ipiv_host(nhist_);
            // copy UV to host
            dapi_checkCudaErrors(dapi_cudaMemcpyAsync(
                first_uv_host.pointer(), U,
                sizeof(ValT) * nhist_ * (nhist_ + 1),
                dapi_cudaMemcpyDeviceToHost, stream_));
            // wait for copy to complete
            dapi_checkCudaErrors(dapi_cudaStreamSynchronize(stream_));
            // lauch gesv on host
            ValT *U_host = first_uv_host.pointer();
            ValT *V_host = U_host + nhist_ * nhist_;
            auto info = LAPACKE_dgesv(LAPACK_COL_MAJOR, nhist_, 1, U_host, nhist_, ipiv_host.pointer(), V_host, nhist_);
            // cpoy results in V to coef_ on device
            dapi_checkCudaErrors(dapi_cudaMemcpyAsync(
                coef_, V_host,
                sizeof(ValT) * nhist_,
                dapi_cudaMemcpyHostToDevice, stream_));
#endif
        }

        void UVSolver::modify_coef()
        {
            launch_kernel(kernel::modifyCoef<threads_per_block_>, {1, 1, 1}, {threads_per_block_, 1, 1}, {nhist_, *coef_ptr_}, 0);
        }
    }
}