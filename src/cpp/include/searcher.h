/*
 * @Author: Guyue
 * @Date: 2026-03-23 14:27:36
 * @LastEditTime: 2026-04-03 10:55:13
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/searcher.h
 */
#ifndef SEARCHER_H
#define SEARCHER_H

#include <common.h>
// #include <matrix_utils.h>
// #include <list_scanning.h>
#include <partition_manager.h>
#include <partition_tree.h>
// #include <geometry.h>
#include <parallel.h>
#include <immintrin.h>
#include "faiss/utils/Heap.h"
#include "faiss/utils/distances.h"

class Searcher {
public:
    faiss::MetricType metric_;

    /**
     * @brief: 查询器构造函数
     * @param {MetricType} metric 距离度量
     * @return {*}
     */    
    Searcher(faiss::MetricType metric);

    /**
     * @brief: 查询器析构函数
     * @return {*}
     */    
    ~Searcher();

    size_t div_roundup(size_t num, size_t denom) 
    {
        return (num + static_cast<size_t>(denom) - static_cast<size_t>(1)) / static_cast<size_t>(denom);
    }
    
    /**
     * @brief: 对分区中心进行搜索
     * @param {shared_ptr<PartitionManager>} centroids_manager 分区中心管理器
     * @param {int64_t} n_queries 查询数量
     * @param {vector<float>& queries} 查询向量
     * @param {std::shared_ptr<SearchParams>} search_params 搜索参数设置
     * @return {*} 分区中心搜索结果
     */    
    std::shared_ptr<SearchResult> search_centers(std::shared_ptr<PartitionManager> centroids_manager, int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params);

    /**
     * @brief: 对分区树进行搜索
     * @param {shared_ptr<PartitionTree>} partition_tree 分区树
     * @param {int64_t} n_queries 查询数量
     * @param {vector<float>& queries} 查询向量
     * @param {std::shared_ptr<SearchParams>} search_params 搜索参数设置
     * @return {*} 分区中心搜索结果
     */     
    std::shared_ptr<InsertSearchResult> search_tree(std::shared_ptr<PartitionTree> partition_tree, int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params);

    /**
     * @brief: 对分区树进行搜索
     * @param {shared_ptr<PartitionTree>} partition_tree 分区树
     * @param {int64_t} n_queries 查询数量
     * @param {vector<float>& queries} 查询向量
     * @param {std::shared_ptr<SearchParams>} search_params 搜索参数设置
     * @return {*} 分区中心搜索结果
     */     
    std::shared_ptr<InsertSearchResult> search_greedy(std::shared_ptr<PartitionTree> partition_tree, int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params);

    /**
     * @brief: 对分区执行搜索
     * @param {shared_ptr<PartitionManager>} partition_manager 分区管理器
     * @param {int64_t} n_queries 查询数量
     * @param {vector<float>&} queries 查询向量
     * @param {std::vector<std::vector<int64_t>>&} scan_lists 每个查询扫描的分区ids
     * @param {std::shared_ptr<SearchParams>} search_params 搜索参数设置
     * @return {*} 搜索结果
     */
    std::shared_ptr<SearchResult> search_partitions(std::shared_ptr<PartitionManager> partition_manager, int64_t n_queries, std::vector<float>& queries, std::vector<std::vector<int64_t>>& scan_lists, std::shared_ptr<SearchParams> search_params, std::shared_ptr<PQParams> pq_params = nullptr);

    /**
     * @brief: 对分区执行搜索(SIMD优化)
     * @param {shared_ptr<PartitionManager>} partition_manager 分区管理器
     * @param {int64_t} n_queries 查询数量
     * @param {vector<float>&} queries 查询向量
     * @param {std::vector<std::vector<int64_t>>&} scan_lists 每个查询扫描的分区ids
     * @param {std::shared_ptr<SearchParams>} search_params 搜索参数设置
     * @return {*} 搜索结果
     */
    std::shared_ptr<SearchResult> search_partitions_acc(std::shared_ptr<PartitionManager> partition_manager, int64_t n_queries, std::vector<float>& queries, std::vector<std::vector<int64_t>>& scan_lists, std::shared_ptr<SearchParams> search_params, std::shared_ptr<PQParams> pq_params = nullptr);

    /**
     * @brief: 对分区执行批量搜索
     * @param {shared_ptr<PartitionManager>} partition_manager 分区管理器
     * @param {int64_t} n_queries 查询数量
     * @param {vector<float>&} queries 查询向量
     * @param {std::vector<std::vector<int64_t>>&} scan_lists 每个查询扫描的分区ids
     * @param {std::shared_ptr<SearchParams>} search_params 搜索参数设置
     * @return {*} 搜索结果
     */
    std::shared_ptr<SearchResult> search_partitions_batch(std::shared_ptr<PartitionManager> partition_manager, int64_t n_queries, std::vector<float>& queries, std::vector<std::vector<int64_t>>& scan_lists, std::shared_ptr<SearchParams> search_params, std::shared_ptr<PQParams> pq_params = nullptr);

    /**
     * @brief: 搜索待插入向量的所属分区
     * @param {shared_ptr<PartitionManager>} centroids_manager 分区中心管理器
     * @param {int64_t} n_queries 查询数量
     * @param {vector<float>& queries} 查询向量
     * @param {std::shared_ptr<SearchParams>} search_params 搜索参数设置
     * @return {*} 分区中心搜索结果
     */    
    std::shared_ptr<InsertSearchResult> search_insert(std::shared_ptr<PartitionManager> centroids_manager, int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params);

    /**
     * @brief: 计算查询与分区向量的距离
     * @param {float*} query_vec 查询向量
     * @param {float*} list_vecs 分区向量
     * @param {int64_t*} list_ids 向量ids
     * @param {int} list_size 分区规模
     * @param {int} d 向量维度
     * @param {float*} simi 距离结果
     * @param {int64_t*} idxi id结果
     * @param {size_t} k 结果集大小
     * @param {MetricType} metric 距离度量
     * @return {*}
     */    
    void scan_one_list(const float* query_vec, const uint8_t* list_vecs, const int64_t* list_ids, int list_size, int d, float* simi, int64_t* idxi, size_t k, faiss::MetricType metric, std::shared_ptr<PQParams> pq_params = nullptr);


    void accumulating_one2manyl2_avx2(const float* query, int dim, const float* dataset, std::vector<float>& result);
};

#endif // SEARCHER_H