/*
 * @Author: Guyue
 * @Date: 2026-04-01 18:49:27
 * @LastEditTime: 2026-04-01 19:09:15
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/src/hit_counter.cpp
 */
#include <hit_counter.h>

HitCounter::HitCounter(int window_size, int total_vectors)
    : window_size_(window_size),
      total_vectors_(total_vectors),
      curr_query_index_(0),              // 当前写入索引的位置
      num_queries_recorded_(0),          // 已记录的查询总数
      running_sum_scan_fraction_(0.0f),  // 窗口内扫描比例的总和(用于快速计算平均值)
      current_scan_fraction_(1.0f)       // 当前平均扫描比例
{
    if (window_size_ <= 0 || total_vectors_ <= 0)
    {
        throw std::invalid_argument("[HitCounter] HitCounter: Window size and Total vectors must be positive");
    }
    per_query_hits_.resize(window_size_);
    per_query_scanned_sizes_.resize(window_size_);
}

void HitCounter::reset()
{
    curr_query_index_ = 0;
    num_queries_recorded_ = 0;
    running_sum_scan_fraction_ = 0.0f;
    current_scan_fraction_ = 1.0f;
    per_query_hits_.clear();
    per_query_hits_.resize(window_size_);
    per_query_scanned_sizes_.clear();
    per_query_scanned_sizes_.resize(window_size_);
}

float HitCounter::compute_scan_fraction(const std::vector<int64_t>& scanned_sizes) const
{
    // 计算单次查询的扫描比例: 扫描的向量数之和 / 总向量数
    int sum = std::accumulate(scanned_sizes.begin(), scanned_sizes.end(), 0);
    return static_cast<float>(sum) / static_cast<float>(total_vectors_);
}

void HitCounter::add_query_data(const std::vector<int64_t>& hit_partition_ids, const std::vector<int64_t>& scanned_sizes)
{
    // 添加新的一次查询数据
    if (hit_partition_ids.size() != scanned_sizes.size())
    {
        throw std::invalid_argument("[HitCounter] add_query_data: Input vector lengths must match");
    }

    // 1> 计算本次扫描的扫描比例
    float query_fraction = compute_scan_fraction(scanned_sizes);
    // 2> 维护滑动窗口逻辑
    if (num_queries_recorded_ < window_size_)
    {
        // 窗口未满, 直接追加记录
        per_query_hits_[num_queries_recorded_] = hit_partition_ids;
        per_query_scanned_sizes_[num_queries_recorded_] = scanned_sizes;
        running_sum_scan_fraction_ += query_fraction;
        num_queries_recorded_++;
    } else {
        // 窗口已满, 先进先出
        
        // a> 从总和中减去即将被覆盖的最旧数据的比例
        running_sum_scan_fraction_ -= compute_scan_fraction(per_query_scanned_sizes_[curr_query_index_]);
        
        // b> 覆盖旧数据
        per_query_hits_[curr_query_index_] = hit_partition_ids;
        per_query_scanned_sizes_[curr_query_index_] = scanned_sizes;
        
        // c> 加上新数据的比例
        running_sum_scan_fraction_ += query_fraction;

        // d> 移动指针, 指向下一个最旧的位置
        curr_query_index_ = (curr_query_index_ + 1) % window_size_;
    }
    // 3> 更新当前的平均扫描比例
    int effective_window = (num_queries_recorded_ < window_size_) ? num_queries_recorded_ : window_size_;
    current_scan_fraction_ = running_sum_scan_fraction_ / static_cast<float>(effective_window);
}

void HitCounter::set_total_vectors(int total_vectors)
{
    // 设置总向量数
    if (total_vectors <= 0) {
        throw std::invalid_argument("[HitCounter] set_total_vectors: Total vectors must be positive");
    }
    total_vectors_ = total_vectors;
}

float HitCounter::get_current_scan_fraction() const {
    // 获取当前窗口记录扫描比例均值
    return current_scan_fraction_;
}

const std::vector<std::vector<int64_t>>& HitCounter::get_per_query_hits() const {
    // 获取查询记录命中分区IDs记录
    return per_query_hits_;
}

const std::vector<std::vector<int64_t>>& HitCounter::get_per_query_scanned_sizes() const {
    // 获取查询记录扫描分区规模记录
    return per_query_scanned_sizes_;
}

int HitCounter::get_window_size() const {
    // 获取窗口大小
    return window_size_;
}

int64_t HitCounter::get_num_queries_recorded() const {
    // 获取查询记录数量
    return num_queries_recorded_;
}

