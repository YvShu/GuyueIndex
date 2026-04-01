/*
 * @Author: Guyue
 * @Date: 2026-03-23 10:12:56
 * @LastEditTime: 2026-04-01 15:17:57
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/common.h
 */
#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
#include <faiss/MetricType.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <memory>
#include <math.h>
#include <set>
#include <queue>
#include <cassert>
#include <mutex>
#include <algorithm>
#include <omp.h>

// --- 关于索引构建的默认参数 ---
constexpr int DEFAULT_NLIST = 0;                            // 默认索引构建聚类个数
constexpr int DEFAULT_NITER = 10;                           // 默认索引构建聚类迭代次数
constexpr const char* DEFAULT_MERTIC = "l2";                // 默认距离度量
constexpr bool DEFAULT_TREE_BUILD = false;                  // 是否构建分区树

// --- 关于维护策略的默认参数 ---
constexpr const char* DEFAULT_REINDEXING_STRATEGY = "None"; // 默认索引维护策略
constexpr int DEFAULT_WINDOW_SIZE = 1000;                   // 默认窗口大小
constexpr int DEFAULT_REINDEXING_RADIUS = 15;               // 默认重索引半径
constexpr int DEFAULT_REINDEXING_ITERATIONS = 5;            // 默认重索引迭代次数
constexpr int DEFAULT_MAX_PARTITION_SIZE = 460;             // 默认最大分区大小
constexpr int DEFAULT_MIN_PARTITION_SIZE = 100;             // 默认最小分区大小
constexpr int DEFAULT_TARGET_PARTITION_SIZE = 500;          // 默认目标分区大小
constexpr int DEFAULT_TARGET_NLIST = -1;                    // 目标分区数量
constexpr int DEFAULT_TOPK_LARGEST_PARTITIONS = 32;         // 默认获取最大分区个数
constexpr bool DEFAULT_CENTROIDS_UPDATE = false;            // 默认分区中心是否更新
constexpr float DEFAULT_HEATING_PARAM = 0.005;              // 默认升温因子
constexpr float DEFAULT_COOLING_PARAM = 0.05;               // 默认降温因子
constexpr float DEFAULT_TEMPERATURE_PARAM = 1.0;            // 默认温度权重
constexpr float DEFAULT_IMBALANCE_PARAM = 1.0;              // 默认不平衡权重
constexpr float DEFAULT_REINDEXING_THRESHOLD = 5;           // 默认重索引阈值
constexpr int DEFAULT_NUM_THREADS = 16;                     // 默认线程数量

// --- 关于量化的默认参数 ---
constexpr size_t DEFAULT_BYTES_PER_DIM = 8;                 // 默认每维度量化后的大小
constexpr size_t DEFAULT_EXTRA_BYTES = 64;                  // 额外开销

const std::vector<int> DEFAULT_LATENCY_ESTIMATOR_RANGE_N = {1, 2, 4, 16, 64, 256, 1024, 4096, 16384, 65536};   ///< Default range of n values for latency estimator.
const std::vector<int> DEFAULT_LATENCY_ESTIMATOR_RANGE_K = {1, 4, 16, 64, 256};                                ///< Default range of k values for latency estimator.
constexpr int DEFAULT_LATENCY_ESTIMATOR_NTRIALS = 5;  

// --- 关于搜索的默认参数 ---
constexpr int DEFAULT_K = 10;                                // 默认结果集大小
constexpr int DEFAULT_NPROBE = 1;                            // 默认查询分区数量
constexpr int DEFAULT_BEAM_SIZE = 4;                         // 搜索束大小

/**
 * @brief: 索引构建参数
 */
struct IndexBuildParams
{
    int dimension = 0;
    int nlist = DEFAULT_NLIST;
    int niter = DEFAULT_NITER;
    std::string metric = DEFAULT_MERTIC;
    bool tree_build = DEFAULT_TREE_BUILD;

    IndexBuildParams() = default;
};

/**
 * @brief: 索引维护参数
 */
struct PartitionStates
{
    std::vector<float> initial_centroids;
    float temperature = 1.0;
    float reindex_score = 0.0;

    PartitionStates() = default;
    PartitionStates(const std::vector<float>& centroids, float T = 1.0, float score = 0.0)
        : initial_centroids(centroids), temperature(T), reindex_score(score) {}
};

/**
 * @brief: 量化参数
 */
struct PQParams
{
    size_t bytes_per_dim = DEFAULT_BYTES_PER_DIM;
    size_t extra_bytes = DEFAULT_EXTRA_BYTES;

    PQParams() = default;
};

struct ReindexingParams
{
    std::string reindexing_strategy = DEFAULT_REINDEXING_STRATEGY;
    int window_size = DEFAULT_WINDOW_SIZE;
    int reindexing_radius = DEFAULT_REINDEXING_RADIUS;
    int refinement_iterations = DEFAULT_REINDEXING_ITERATIONS;
    int max_partition_size = DEFAULT_MAX_PARTITION_SIZE;
    int min_partition_size = DEFAULT_MIN_PARTITION_SIZE;
    int target_partition_size = DEFAULT_TARGET_PARTITION_SIZE;
    int topk_largest_partitions = DEFAULT_TOPK_LARGEST_PARTITIONS;
    bool centroids_update = DEFAULT_CENTROIDS_UPDATE;
    float heating_param = DEFAULT_HEATING_PARAM;
    float cooling_param = DEFAULT_COOLING_PARAM;
    float temperature_param = DEFAULT_TOPK_LARGEST_PARTITIONS;
    float imbalance_param = DEFAULT_IMBALANCE_PARAM;
    float reindexing_threshold = DEFAULT_REINDEXING_THRESHOLD;
    int target_nlist = DEFAULT_TARGET_NLIST;
    // float delete_threshold = 
    // float split_threshold = 
    int num_threads = DEFAULT_NUM_THREADS;

    ReindexingParams() = default;
};

/**
 * @brief: 搜索参数
 */
struct SearchParams
{
    int k = DEFAULT_K;
    int beam_size = DEFAULT_BEAM_SIZE;
    int nprobe = DEFAULT_NPROBE;
    int num_threads = DEFAULT_NUM_THREADS;
    float target_recall;
};

/**
 * @brief: 搜索返回结果
 */
struct SearchResult
{
    std::vector<std::vector<int64_t>> indices;  // 搜索结果IDs
    std::vector<std::vector<float>> distances;  // 搜索结果距离
    double search_time;                         // 搜索时间
    double c_search_time;                       // 中心搜索时间
    double p_search_time;                       // 分区搜索时间
    float search_nprobe;                        // 平均搜索分区数量
    int64_t search_points;                      // 平均扫描点数
};

/**
 * @brief: 插入时的搜索结果—每个插入向量所属的分区
 */
struct InsertSearchResult
{
    std::unordered_map<int64_t, std::vector<int64_t>> assignment;   // 向量分配情况: [分区A,属于分区A的向量ids]
    // new
    // std::vector<float> dists;                                       // 距离信息: 向量到其所在分区中心的距离
};

inline faiss::MetricType str_to_metric_type(std::string metric)
{
    std::transform(metric.begin(), metric.end(), metric.begin(), ::tolower);
    if (metric == "l2")
    {
        return faiss::METRIC_L2;
    } else if (metric == "ip") {
        return faiss::METRIC_INNER_PRODUCT;
    } else {
        throw std::invalid_argument("[common] str_to_metric_type : Invalid metric type: " + metric);
    }
}

inline std::string metric_type_to_str(faiss::MetricType metric)
{
    if (metric == faiss::METRIC_L2)
    {
        return "l2";
    } else if (metric == faiss::METRIC_INNER_PRODUCT) {
        return "ip";
    } else {
        throw std::invalid_argument("[common] str_to_metric_type : Invalid metric type.");
    }
}

#endif // COMMON_H