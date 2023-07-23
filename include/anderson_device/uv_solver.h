#pragma once

#include "anderson_device.h"

#include <memory>
#include <vector>
#include <stdexcept>
#include <type_traits>

#include "device_api/device_api_cuda_runtime.h"

#include "qutility/traits.h"
#include "qutility/array_wrapper/array_wrapper_gpu_declare.h"

#include "qutility_device/def.h"
#include "qutility_device/event.h"
#include "qutility_device/workspace.h"

#include "uv_solver.cuh"

namespace anderson
{
    namespace device
    {
        class UVSolver : public qutility::device::event::StreamEventHelper
        {
        public:
            constexpr static size_t threads_per_block_ = ANDERSON_DEVICE_THREADS_PER_BLOCK;

            using StreamEventHelper = qutility::device::event::StreamEventHelper;
            using ValT = double;
            using ArrayT = qutility::array_wrapper::DArrayGPU<ValT>;
            using WorkspaceT = qutility::device::workspace::Workspace<ValT>;

            using DataValT = int;
            using DataArrayT = qutility::array_wrapper::DArrayGPU<DataValT>;

            UVSolver() = delete;
            UVSolver(const UVSolver &) = delete;
            UVSolver &operator=(const UVSolver &) = delete;
            UVSolver(UVSolver &&) = delete;
            UVSolver &operator=(UVSolver &&) = delete;
            ~UVSolver();

            UVSolver(int device, size_t nhist, const std::shared_ptr<WorkspaceT> &working);

            template <typename... Ts>
            dapi_cudaEvent_t solve(
                size_t using_history_count_suggestion,
                DualHistory &first_dh, double first_weight, Ts &&...Args);

        private:
            friend class DualHistory;
            const size_t nhist_;
            std::unique_ptr<ArrayT> coef_ptr_;
            std::unique_ptr<DataArrayT> data_ptr_;
            const std::shared_ptr<WorkspaceT> workspace_;

            void *lapack_solver_ = nullptr;

            int iter_;
            size_t working_size_ = 0;

            template <typename... T>
            struct type_seq
            {
            };
            template <size_t N, typename... T>
            struct gen_type_seq : gen_type_seq<N - 1, ValT *, ValT, T...>
            {
            };
            template <typename... T>
            struct gen_type_seq<0, T...> : type_seq<T...>
            {
            };

            inline auto get_using_history_count(size_t count_suggestion) const -> std::size_t;
            template <typename First, typename... Rest>
            auto get_using_history_count(size_t count_suggestion, First &&first, Rest &&...rest) const -> std::size_t;

            double *to_raw_data(DualHistory &dh);
            double to_raw_data(double &val);
            dapi_cudaStream_t get_stream(DualHistory &dh);
            dapi_cudaStream_t get_stream(double &val);

            template <typename... PointerWeightTypePair>
            inline auto get_kernel_pointer_impl(type_seq<PointerWeightTypePair...>)
            {
                return kernel::combineUV<PointerWeightTypePair...>;
            }
            template <size_t N>
            inline auto get_kernel_pointer()
            {
                return get_kernel_pointer_impl(gen_type_seq<N>());
            }

            void mask_old_uv(double *uv, std::size_t current_pos, std::size_t using_history_count);
            void solve_eigen_system(double *uv);
            void modify_coef();

            template <typename... Ts>
            void solve_impl(size_t using_history_count, size_t current_pos, double *first_uv, double first_weight, Ts &&...Args);
        };

        template <typename... Ts>
        dapi_cudaEvent_t UVSolver::solve(
            size_t using_history_count_suggestion,
            DualHistory &first_dh, double first_weight, Ts &&...Args)
        {
            static_assert(qutility::traits::is_correct_list_of_n<
                          2, qutility::traits::is_type_T<DualHistory>, qutility::traits::is_type_T<ValT>, DualHistory, double, std::remove_reference_t<Ts>...>::value);

            set_device();

            dapi_cudaStream_t streams_to_depend_on[] = {get_stream(first_dh), get_stream(first_weight), get_stream(Args)...};
            for (const auto &stream : streams_to_depend_on)
            {
                if (stream != nullptr)
                    this_wait_other(stream); // stream created will never be nullptr (namely default stream)
            }

            size_t using_history_count = get_using_history_count(using_history_count_suggestion, first_dh, first_weight, Args...);
            size_t pos = first_dh.get_current_pos();
            solve_impl(using_history_count, pos, to_raw_data(first_dh), to_raw_data(first_weight), to_raw_data(Args)...);

            for (const auto &stream : streams_to_depend_on)
            {
                if (stream != nullptr)
                    other_wait_this(stream); // stream created will never be nullptr (namely default stream)
            }

            return record_event();
        }

        inline auto UVSolver::get_using_history_count(size_t count_suggestion) const -> size_t
        {
            return std::min(count_suggestion, nhist_);
        }
        template <typename First, typename... Rest>
        auto UVSolver::get_using_history_count(size_t count_suggestion, First &&first, Rest &&...rest) const -> size_t
        {
            if constexpr (std::is_same<std::remove_reference_t<First>, DualHistory>::value)
            {
                size_t val_rest = get_using_history_count(count_suggestion, rest...);
                size_t val_first_1 = first.get_valid_sequence_count();
                size_t val_first_2 = first.get_available_count();
                return std::min(std::min(val_first_1, val_first_2), val_rest);
            }
            else
            {
                return get_using_history_count(count_suggestion, rest...);
            }
            return 0;
        }

        template <typename... Ts>
        void UVSolver::solve_impl(size_t using_history_count, size_t current_pos, double *first_uv, double first_weight, Ts &&...Args)
        {
            static_assert(qutility::traits::is_correct_list_of_n<
                          2, qutility::traits::is_type_T<ValT *>, qutility::traits::is_type_T<ValT>, double *, double, std::remove_reference_t<Ts>...>::value);
            {
                constexpr size_t dual_history_count = (sizeof...(Args) / 2) + 1;
                auto kernel_ptr = get_kernel_pointer<dual_history_count>();
                launch_kernel<threads_per_block_>(kernel_ptr, {nhist_, first_uv, first_weight, Args...}, 0);
            }

            mask_old_uv(first_uv, current_pos, using_history_count);
            solve_eigen_system(first_uv);
            modify_coef();
        }
    }
}