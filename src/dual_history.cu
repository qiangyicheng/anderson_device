#include "anderson_device/dual_history.h"
#include "anderson_device/dual_history.cuh"
#include "anderson_device/uv_solver.h"

#include "qutility_device/field.h"

#include "device_api/device_api_helper.h"

#include "qutility/array_wrapper/array_wrapper_gpu.h"

namespace anderson
{
    namespace device
    {
        DualHistory::DualHistory(
            int device,
            std::size_t single_size,
            std::size_t nhist,
            const std::shared_ptr<WorkspaceT> &workspace)
            : StreamEventHelper(device),
              single_size_(single_size),
              nhist_(nhist),
              workspace_(workspace),
              data_ptr_(std::make_unique<ArrayT>(0., (nhist + 1) * (nhist + 1), device)),
              uv_ptr_(std::make_unique<ArrayT>(0., (nhist + 1) * (nhist), device)),
              hist_ptr_(std::make_unique<ArrayT>(0., single_size * nhist, device)),
              hist_diff_ptr_(std::make_unique<ArrayT>(0., single_size * nhist, device)),
              field_(hist_ptr_->pointer(), single_size, nhist),
              field_diff_(hist_diff_ptr_->pointer(), single_size, nhist),
              is_valid_data_ptr_(std::make_unique<MaskArrayT>(-1, nhist, device)) /*-1 means at the beginning all record is invalid*/,
              is_valid_(is_valid_data_ptr_->pointer(), 1, nhist),
              old_old_(data_ptr_->pointer()),
              new_old_(data_ptr_->pointer() + nhist * nhist),
              new_new_(data_ptr_->pointer() + nhist * (nhist + 1)),
              u_(uv_ptr_->pointer()),
              v_(uv_ptr_->pointer() + nhist * nhist)
        {
            set_device();
            dapi_checkCudaErrors(dapi_cublasCreate_v2(&dapi_cublashandle));
            dapi_checkCudaErrors(dapi_cublasSetStream_v2(dapi_cublashandle, stream_));
        }

        DualHistory::~DualHistory()
        {
            set_device();
            dapi_checkCudaErrors(dapi_cublasDestroy_v2(dapi_cublashandle));
        }

        auto DualHistory::get_valid_sequence_count() const -> std::size_t { return valid_sequence_count_; }
        auto DualHistory::get_available_count() const -> std::size_t { return field_.available(); }
        auto DualHistory::get_current_pos() const -> std::size_t { return field_.pos(); }
        auto DualHistory::get_uv() const -> double * { return uv_ptr_->pointer(); }

        auto DualHistory::update_inner_and_calc_uv(DualFieldT &field) -> dapi_cudaEvent_t
        {
            set_device();
            this_wait_other(field.stream_);
            calc_new_old_impl(field.field_diff_);
            calc_uv_impl(field.field_diff_);
            other_wait_this(field.stream_);
            return record_event();
        }
        auto DualHistory::update_inner_and_calc_uv_invalid() -> dapi_cudaEvent_t
        {
            set_device();
            calc_uv_invalid_impl();
            return record_event();
        }

        auto DualHistory::mix_and_push(DualFieldT &field, UVSolver &uv_solver, double acceptance) -> dapi_cudaEvent_t
        {
            set_device();
            this_wait_other(field.stream_);
            this_wait_other(uv_solver.stream_);
            mix_impl(field.field_, field.field_diff_, *uv_solver.coef_ptr_, acceptance);
            push_impl();
            other_wait_this(field.stream_);
            other_wait_this(uv_solver.stream_);
            return record_event();
        }

        auto DualHistory::mix_simple_and_push(DualFieldT &field, double acceptance) -> dapi_cudaEvent_t
        {
            set_device();
            this_wait_other(field.stream_);
            mix_simple_impl(field.field_, field.field_diff_, acceptance);
            push_impl();
            other_wait_this(field.stream_);
            return record_event();
        }

        auto DualHistory::duplicate_and_push_invalid(DualFieldT &field) -> dapi_cudaEvent_t
        {
            set_device();
            this_wait_other(field.stream_);
            duplicate_impl(field.field_);
            push_invalid_impl();
            other_wait_this(field.stream_);
            return record_event();
        }

        void DualHistory::calc_new_old_impl(double *new_field_diff)
        {
            if (field_diff_.available() == 0)
                return;
            double alpha = 1.;
            double beta = 0.;
            dapi_checkCudaErrors(dapi_cublasDgemv_v2(
                dapi_cublashandle, DAPI_CUBLAS_OP_T,
                single_size_, (int)(field_diff_.available()),
                &alpha,
                field_diff_.begin(), single_size_,
                new_field_diff, 1,
                &beta,
                new_old_, 1));
            return;
        }

        void DualHistory::calc_uv_impl(double *new_field_diff)
        {
            this_wait_other(workspace_->stream_);
            launch_kernel_cg<threads_per_block_>(
                kernel::andersonDualHistoryCalcUV<threads_per_block_>,
                {single_size_,
                 nhist_,
                 new_field_diff,
                 *data_ptr_,
                 *uv_ptr_,
                 *is_valid_data_ptr_,
                 *workspace_,
                 field_.available(),
                 field_.pos()},
                0);
            other_wait_this(workspace_->stream_);
            return;
        }

        void DualHistory::calc_uv_invalid_impl()
        {
            this_wait_other(workspace_->stream_);
            launch_kernel<threads_per_block_>(
                kernel::andersonDualHistoryCalcUVInvalid,
                {nhist_,
                 *data_ptr_,
                 *uv_ptr_,
                 field_.pos()},
                0);
            other_wait_this(workspace_->stream_);
            return;
        }

        void DualHistory::mix_impl(double *new_field, const double *new_field_diff, const double *coef, double acceptance)
        {
            launch_kernel<threads_per_block_>(
                kernel::andersonDualHistoryMix,
                {single_size_,
                 nhist_,
                 *hist_ptr_,
                 *hist_diff_ptr_,
                 new_field,
                 new_field_diff,
                 coef,
                 *is_valid_data_ptr_,
                 field_.available(),
                 field_.pos(),
                 acceptance},
                (nhist_ + 1) * sizeof(double));
            return;
        }

        void DualHistory::mix_simple_impl(double *new_field, const double *new_field_diff, double acceptance)
        {
            launch_kernel<threads_per_block_>(
                kernel::andersonDualHistoryMixSimple,
                {single_size_,
                 *hist_ptr_,
                 *hist_diff_ptr_,
                 new_field,
                 new_field_diff,
                 *is_valid_data_ptr_,
                 field_.pos(),
                 acceptance},
                0);
            return;
        }

        void DualHistory::duplicate_impl(const double *new_field)
        {
            launch_kernel<threads_per_block_>(
                kernel::andersonDualHistoryDuplicate,
                {single_size_,
                 *hist_ptr_,
                 *hist_diff_ptr_,
                 new_field,
                 *is_valid_data_ptr_,
                 field_.pos()},
                0);
            return;
        }

        void DualHistory::push_impl()
        {
            field_.push();
            field_diff_.push();
            valid_sequence_count_ += 1;
            if (valid_sequence_count_ > nhist_)
                valid_sequence_count_ = nhist_;
        }

        void DualHistory::push_invalid_impl()
        {
            field_.push();
            field_diff_.push();
            valid_sequence_count_ = 0;
        }
    }
}