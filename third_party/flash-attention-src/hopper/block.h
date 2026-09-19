/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

template <class SeqlenInfo_t, int kBlockM, int kBlockN, bool Is_causal, bool Is_local, bool PackGQA=false, bool Split=false>
struct BlockMN {

    static
    CUTLASS_DEVICE
    int warp_allreduce_max(int value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value = std::max(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        return __shfl_sync(0xffffffff, value, 0);
    }

    static
    CUTLASS_DEVICE
    int warp_allreduce_min(int value) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            value = std::min(value, __shfl_down_sync(0xffffffff, value, offset));
        }
        return __shfl_sync(0xffffffff, value, 0);
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod,
            int const* const cross_kv_boundary = nullptr) {

        int const seqlen_k = seqlen_info.seqlen_k;
        int const seqlen_q = seqlen_info.seqlen_q;
        int n_block_max = cute::ceil_div(seqlen_k, kBlockN);
        if constexpr (Is_causal || Is_local) {
            if (cross_kv_boundary != nullptr) {
                int max_kv = 0;
                int m_start = m_block * kBlockM;
                int m_end = (m_block + 1) * kBlockM;
                if constexpr (PackGQA) {
                    // MMA rows are packed with query heads; convert to actual Q rows.
                    m_start = qhead_per_khead_divmod.divide(m_start);
                    m_end = qhead_per_khead_divmod.divide(m_end - 1) + 1;
                }
                m_start = std::min(m_start, seqlen_q);
                m_end = std::min(m_end, seqlen_q);
                int const lane = threadIdx.x % 32;
                for (int i = m_start + lane; i < m_end; i += 32) {
                    max_kv = std::max(max_kv, cross_kv_boundary[i]);
                }
                max_kv = warp_allreduce_max(max_kv);
                n_block_max = std::min(n_block_max, cute::ceil_div(max_kv, kBlockN));
            } else {
                int m_idx_max = (m_block + 1) * kBlockM;
                // TODO: check off-by-1 error
                if (PackGQA) { m_idx_max = qhead_per_khead_divmod.divide(m_idx_max - 1) + 1 ; }
                int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
                int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
                if (Is_local && attention_chunk_divmod.divisor > 0) {
                    n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
                }
                n_block_max = std::min(n_block_max, cute::ceil_div(n_idx_right, kBlockN));
            }
        }
        int n_block_min = 0;
        if constexpr (Is_local) {
            if (cross_kv_boundary == nullptr) {
                int m_idx_min = m_block * kBlockM;
                if (PackGQA) { m_idx_min = qhead_per_khead_divmod.divide(m_idx_min); }
                int const n_idx = m_idx_min + seqlen_k - seqlen_q;
                int n_idx_left = n_idx - window_size_left;
                if (attention_chunk_divmod.divisor > 0) {
                    n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
                }
                n_block_min = std::max(int(0), n_idx_left / kBlockN);
            }
            // When cross_kv_boundary != nullptr, n_block_min stays 0 (staircase has no left boundary)
        }
        // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        if constexpr (Split) {
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16; // first 16 bits are for num_splits
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            int split_idx_actual = split_idx & 0x0000FFFF;
            int num_splits_actual = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            int num_n_blocks_per_split = n_block_max <= n_block_min ? 0 : cute::ceil_div(n_block_max - n_block_min, num_splits_actual);
            n_block_min = n_block_min + split_idx_actual * num_n_blocks_per_split;
            n_block_max = std::min(n_block_min + num_n_blocks_per_split, n_block_max);
            // if (threadIdx.x == 128) { printf("Inside, bid.x = %d, bid.y = %d, bid.z = %d, split_idx = %d, num_splits_dynamic = %d, num_splits_actual = %d, num_n_blocks_per_split = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.x, blockIdx.y, blockIdx.z, split_idx, num_splits_dynamic, num_splits_actual, num_n_blocks_per_split, n_block_min, n_block_max); }
        }
        // if (threadIdx.x == 128) { printf("After split, inside, bid.y = %d, bid.z = %d, split_idx = %d, n_block_min: %d, n_block_max: %d\n", blockIdx.y, blockIdx.z, split_idx, n_block_min, n_block_max); }
        return {n_block_min, n_block_max};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_k_new_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const bidb, int const split_idx, int const num_splits,
            int const window_size_left, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod,
            int const* const cross_kv_boundary = nullptr) {

        auto [n_block_min, n_block_max] = get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, num_splits,
            window_size_left, window_size_right, attention_chunk_divmod, qhead_per_khead_divmod, cross_kv_boundary);
        int const idx_k_new_min = std::max(n_block_min * kBlockN - seqlen_info.seqlen_k_og, 0);
        int const idx_k_new_max = std::min(n_block_max * kBlockN - seqlen_info.seqlen_k_og, seqlen_info.seqlen_k_new);
        int const n_block_new_min = idx_k_new_min / kBlockN;
        int const n_block_new_max = idx_k_new_max > idx_k_new_min ? cute::ceil_div(idx_k_new_max, kBlockN) : n_block_new_min;
        // if (threadIdx.x == 128 && m_block == 0) { printf("bidb = %d, seqlen_k_new = %d, seqlen_k_og = %d, n_block_min = %d, n_block_max = %d, idx_k_new_min = %d, idx_k_new_max = %d, n_block_new_min = %d, n_block_new_max = %d\n", bidb, seqlen_k_new, seqlen_k_og, n_block_min, n_block_max, idx_k_new_min, idx_k_new_max, n_block_new_min, n_block_new_max);}
        return {n_block_new_min, n_block_new_max};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_n_block_k_new_store_min_max(
            SeqlenInfo_t const& seqlen_info, int const split_idx, int const num_splits) {
        int n_block_new_min = 0;
        int n_block_new_max = cute::ceil_div(seqlen_info.seqlen_k_new, kBlockN);
        if constexpr (Split) {
            uint32_t num_splits_dynamic_u = reinterpret_cast<uint32_t const&>(split_idx) >> 16;
            int num_splits_dynamic = reinterpret_cast<int&>(num_splits_dynamic_u);
            int split_idx_actual = split_idx & 0x0000FFFF;
            int num_splits_actual = num_splits_dynamic > 0 ? num_splits_dynamic : num_splits;
            int num_n_blocks_per_split = cute::ceil_div(n_block_new_max, num_splits_actual);
            n_block_new_min = split_idx_actual * num_n_blocks_per_split;
            n_block_new_max = std::min(n_block_new_min + num_n_blocks_per_split, n_block_new_max);
        }
        return {n_block_new_min, n_block_new_max};
    }

    static
    CUTLASS_DEVICE
    cute::tuple<int, int> get_m_block_min_max(
            SeqlenInfo_t const& seqlen_info,
            int const n_block, int const bidb,
            int const window_size_left, int const window_size_right, int const sink_token_length,
            int const* const cross_kv_boundary = nullptr) {
        // TODO: support attention_chunk
        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int m_block_max = cute::ceil_div(seqlen_q, kBlockM);
        int m_block_min = 0;
        if constexpr (Is_causal || Is_local) {
            if (cross_kv_boundary != nullptr) {
                // Staircase: for K-block [n_start, n_start + kBlockN), a Q row
                // contributes iff cross_kv_boundary[row] > n_start. Raw cross_kv_boundary can
                // drop back to 0 for right padding, so do not binary-search it.
                int const n_start = n_block * kBlockN;
                int first_row = seqlen_q;
                int last_row = -1;
                int const lane = threadIdx.x % 32;
                for (int row = lane; row < seqlen_q; row += 32) {
                    if (cross_kv_boundary[row] > n_start) {
                        first_row = std::min(first_row, row);
                        last_row = row;
                    }
                }
                first_row = warp_allreduce_min(first_row);
                last_row = warp_allreduce_max(last_row);
                if (last_row < first_row) {
                    return {m_block_max, m_block_max};
                }
                m_block_min = std::max(m_block_min, first_row / kBlockM);
                m_block_max = std::min(m_block_max, last_row / kBlockM + 1);
            } else {
                if constexpr (Is_local) {
                    if (n_block >= cute::ceil_div(sink_token_length, kBlockN)) {
                        m_block_max = std::min(m_block_max, cute::ceil_div((n_block + 1) * kBlockN + seqlen_q - seqlen_k + window_size_left, kBlockM));
                    }
                }
                m_block_min = std::max(m_block_min, (n_block * kBlockN + seqlen_q - seqlen_k - window_size_right) / kBlockM);
            }
        }
        return {m_block_min, m_block_max};
    }

    // If we have separate iterations with causal or local masking at the start, where do we stop
    static
    CUTLASS_DEVICE
    int get_n_block_min_causal_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_right,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod,
            int const* const cross_kv_boundary = nullptr) {
        if (cross_kv_boundary != nullptr) {
            int const seqlen_q = seqlen_info.seqlen_q;
            int m_start = m_block * kBlockM;
            int m_end = (m_block + 1) * kBlockM;
            if constexpr (PackGQA) {
                // MMA rows are packed with query heads; convert to actual Q rows.
                m_start = qhead_per_khead_divmod.divide(m_start);
                m_end = qhead_per_khead_divmod.divide(m_end - 1) + 1;
            }
            m_start = std::min(m_start, seqlen_q);
            m_end = std::min(m_end, seqlen_q);
            if (m_start >= m_end) { return n_block_min; }
            int min_kv = 0x7fffffff;
            int const lane = threadIdx.x % 32;
            for (int i = m_start + lane; i < m_end; i += 32) {
                min_kv = std::min(min_kv, cross_kv_boundary[i]);
            }
            min_kv = warp_allreduce_min(min_kv);
            return std::max(n_block_min, min_kv / kBlockN);
        }
        int const m_idx_min = !PackGQA ? m_block * kBlockM : qhead_per_khead_divmod.divide(m_block * kBlockM);
        int const n_idx = m_idx_min + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        int n_idx_right = !Is_local ? n_idx : n_idx + window_size_right;
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_right = std::min(n_idx_right, flash::round_up(attention_chunk_divmod, n_idx));
        }
        return std::max(n_block_min, n_idx_right / kBlockN);
    }

    // If we have separate iterations with local masking at the end, where do we stop the non-masked iterations
    static
    CUTLASS_DEVICE
    int get_n_block_min_before_local_mask(
            SeqlenInfo_t const& seqlen_info,
            int const m_block, int const n_block_min, int const window_size_left,
            cutlass::FastDivmod const& attention_chunk_divmod,
            cutlass::FastDivmod const& qhead_per_khead_divmod,
            int const* const cross_kv_boundary = nullptr) {
        if (cross_kv_boundary != nullptr) {
            // Staircase: no left boundary, all KV from position 0 are visible
            return n_block_min;
        }
        int const m_idx_max = !PackGQA ? (m_block + 1) * kBlockM : qhead_per_khead_divmod.divide((m_block + 1) * kBlockM - 1) + 1;
        int const n_idx = m_idx_max + seqlen_info.seqlen_k - seqlen_info.seqlen_q;
        int n_idx_left = !Is_local ? n_idx : n_idx - window_size_left;
        if (Is_local && attention_chunk_divmod.divisor > 0) {
            n_idx_left = std::max(n_idx_left, flash::round_down(attention_chunk_divmod, n_idx));
        }
        return !Is_local ? n_block_min : std::max(n_block_min, cute::ceil_div(n_idx_left, kBlockN));
    }

};

} // namespace flash
