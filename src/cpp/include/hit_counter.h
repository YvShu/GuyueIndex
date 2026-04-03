/*
 * @Author: Guyue
 * @Date: 2026-04-01 17:55:57
 * @LastEditTime: 2026-04-01 18:47:14
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/hit_counter.h
 */
#ifndef HIT_COUNTER_H
#define HIT_COUNTER_H

#include <numeric>
#include <common.h>

class HitCounter {
private:
    int window_size_;                                               // 滑动窗口大小, 决定了统计平均值时参考的最近查询次数
    int64_t total_vectors_;                                         // 向量总数, 作为分母用于计算扫描比例
    int64_t curr_query_index_;                                      // 当前写入索引, 在循环缓冲区中指向下一个待覆盖区域
    int64_t num_queries_recorded_;                                  // 已记录查询数, 记录当前窗口内实际有效的查询数量(0到window_size_)

    std::vector<std::vector<int64_t>> per_query_hits_;              // 分区命中历史, 存储窗口内查询分别命中了那些分区ID
    std::vector<std::vector<int64_t>> per_query_scanned_sizes_;     // 扫描规模历史, 存储窗口内每次查询在各个分区扫描的具体向量数量

    float running_sum_scan_fraction_;                               // 扫描比例累加和, 为了高效计算平均值, 维护当前窗口内所有查询扫描比例的总和
    float current_scan_fraction_;                                   // 当前平均扫描比例, 最近N次查询的平均扫描深度(扫描向量数/总向量数)

    /**
     * @brief: 计算一个查询的扫描比例
     * @param {std::vector<int64_t>&} scanned_sizes 单个查询扫描分区的规模
     * @return {*}
     */    
    float compute_scan_fraction(const std::vector<int64_t>& scanned_sizes) const;

public:
    /**
     * @brief: 命中计数器构造函数
     * @param {int} window_size 滑动窗口大小
     * @param {int} total_vectors 索引向量总数
     * @return {*}
     */
    HitCounter(int window_size, int total_vectors);

    /**
     * @brief: 通过清空所有的查询记录数据重置计数器
     * @return {*}
     */    
    void reset();

    /**
     * @brief: 设置索引中的向量总数
     * @param {int} total_vectors 新的向量总数
     * @return {*}
     */    
    void set_total_vectors(int total_vectors);

    /**
     * @brief: 输入接口
     * 1> 计算本次查询的扫描比例(单次扫描数 / 向量总数)
     * 2> 如果窗口已满, 则减去最旧的一条记录, 并用新记录覆盖(循环缓冲区逻辑)
     * 3> 更新current_scan_fraction的平均值
     * @param {const std::vector<int64_t>&} hit_partition_ids 查询期间命中的分区IDs
     * @param {const std::vector<int64_t>&} scanned_sizes 命中的每个分区的向量扫描大小
     * @return {*}
     */    
    void add_query_data(const std::vector<int64_t>& hit_partition_ids, const std::vector<int64_t>& scanned_sizes);

    /**
     * @brief: 获取当前滑动窗口的平均扫描比例
     * @return {*} 当前扫描比例
     */    
    float get_current_scan_fraction() const;

    /**
     * @brief: 获取查询命中计数
     * @return {*} 关于查询命中计数值
     */    
    const std::vector<std::vector<int64_t>>& get_per_query_hits() const;

    /**
     * @brief: 获取查询扫描分区大小
     * @return {*} 关于查询扫描分区的大小
     */    
    const std::vector<std::vector<int64_t>>& get_per_query_scanned_sizes() const;

    /**
     * @brief: 返回滑动窗口大小
     * @return {*} 滑动窗口大小
     */    
    int get_window_size() const;

    /**
     * @brief: 返回目前查询记录总数
     * @return {*} 查询记录总数
     */    
    int64_t get_num_queries_recorded() const;
};

#endif // HIT_COUNTER_H