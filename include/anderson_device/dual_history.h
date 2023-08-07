#pragma once

#include "anderson_device.h"

#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include "device_api/device_api_cuda_runtime.h"
#include "device_api/device_api_cublas.h"

#include "qutility_device/def.h"
#include "qutility_device/event.h"
#include "qutility_device/workspace.h"
#include "qutility_device/field/field_declare.h" //only include the declaration of the templates, avoiding cuda_samplers.h since it has side effects

#include "qutility/history.h"
#include "qutility/message.h"
#include "qutility/array_wrapper/array_wrapper_gpu_declare.h" //only include the declaration of the templates, avoiding cuda_samplers.h since it has side effects

namespace anderson
{
    namespace device
    {

        class DualHistory : public qutility::device::event::StreamEventHelper
        {
        public:
            constexpr static size_t threads_per_block_ = ANDERSON_DEVICE_THREADS_PER_BLOCK;

            using StreamEventHelper = qutility::device::event::StreamEventHelper;
            using ValT = double;
            using ArrayT = qutility::array_wrapper::DArrayGPU<ValT>;
            using HistoryT = qutility::history::DHistory<ValT>;
            using DualFieldT = qutility::device::field::DualField<ValT>;
            using WorkspaceT = qutility::device::workspace::Workspace<ValT>;

            using MaskValT = int;
            using MaskArrayT = qutility::array_wrapper::DArrayGPU<MaskValT>;
            using MaskHistoryT = qutility::history::DHistory<MaskValT>;

            DualHistory() = delete;
            DualHistory(const DualHistory &) = delete;
            DualHistory &operator=(const DualHistory &) = delete;
            DualHistory(DualHistory &&) = delete;
            DualHistory &operator=(DualHistory &&) = delete;
            DualHistory(int device, std::size_t single_size, std::size_t nhist, const std::shared_ptr<WorkspaceT> &working);
            ~DualHistory();

            auto get_valid_sequence_count() const -> std::size_t;
            auto get_available_count() const -> std::size_t;
            auto get_current_pos() const -> std::size_t;
            auto get_uv() const -> double *;

            auto update_inner_and_calc_uv(DualFieldT &field) ->dapi_cudaEvent_t;
            auto update_inner_and_calc_uv_invalid() ->dapi_cudaEvent_t;
            auto mix_and_push(DualFieldT &field, UVSolver &uv_solver, double acceptance) ->dapi_cudaEvent_t;
            auto mix_simple_and_push(DualFieldT &field, double acceptance) ->dapi_cudaEvent_t;
            auto duplicate_and_push_invalid(DualFieldT &field) ->dapi_cudaEvent_t;

            const std::size_t single_size_;
            const std::size_t nhist_;

        private:
            friend class UVSolver;

            std::unique_ptr<ArrayT> hist_ptr_;
            std::unique_ptr<ArrayT> hist_diff_ptr_;

            std::unique_ptr<ArrayT> data_ptr_;
            std::unique_ptr<ArrayT> uv_ptr_;

            // these HistoryT members are only adaptors for the internal data
            HistoryT field_;
            HistoryT field_diff_;

            std::unique_ptr<MaskArrayT> is_valid_data_ptr_;
            MaskHistoryT is_valid_;

            std::size_t valid_sequence_count_ = 0;

            const std::shared_ptr<WorkspaceT> workspace_;

            // derived ptrs
            ValT *const old_old_;
            ValT *const new_old_;
            ValT *const new_new_;
            ValT *const u_;
            ValT *const v_;

            dapi_cublasHandle_t dapi_cublashandle = nullptr;

            void calc_new_old_impl(double *new_field_diff);
            void calc_uv_impl(double *new_field_diff);
            void calc_uv_invalid_impl();
            void mix_impl(double *new_field, const double *new_field_diff, const double *coef, double acceptance);
            void mix_simple_impl(double *new_field, const double *new_field_diff, double acceptance);
            void duplicate_impl(const double *new_field);
            void push_impl();
            void push_invalid_impl();
        };

    }
}