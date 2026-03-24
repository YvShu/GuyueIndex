/*
 * @Author: Guyue
 * @Date: 2026-03-23 15:39:29
 * @LastEditTime: 2026-03-24 15:16:38
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/guyue_index.h
 */
#ifndef GUYUE_INDEX_H
#define GUYUE_INDEX_H

#include <partition_manager.h>
#include <partition_tree.h>
#include <searcher.h>
#include <math.h>
// #include <matrix_utils.h>
// #include <hit_counter.h>
// #include <cost_model.h>
#include <faiss/utils/distances.h>

class GuyueIndex {
public:
    std::shared_ptr<PartitionManager> centroids_manager_;       // 指向分区中心管理器的指针
    std::shared_ptr<PartitionManager> partition_manager_;       // 指向分区管理器的指针
    bool tree_build_;                                           // 树构建？
    std::shared_ptr<PartitionTree> partition_tree_;             // 指向分区树的指针

    std::shared_ptr<Searcher> searcher_;                        // 索引搜索器
    std::shared_ptr<ReindexingParams> reindexing_params_;       // 重索引参数
    
    faiss::MetricType metric_;                                  // 距离度量

    /**
     * @brief: 索引构造函数
     * @return {*}
     */    
    GuyueIndex();

    /**
     * @brief: 索引析构函数
     * @return {*}
     */    
    ~GuyueIndex();

    /**
     * @brief: 索引构建函数
     * @param {vector<float>&} vectors 构建索引所使用的向量，[n_vectors * dim]
     * @param {std::vector<int64_t>&} ids 构建索引的向量ids
     * @param {std::shared_ptr<IndexBuildParams>} build_params 索引构建参数
     * @param {shared_ptr<ReindexingParams>} reindexing_params 索引维护参数
     * @return {*}
     */    
    void build(std::vector<float>& vectors, std::vector<int64_t>& ids, std::shared_ptr<IndexBuildParams> build_params, std::shared_ptr<ReindexingParams> reindexing_params);

    /**
     * @brief: 向量搜索
     * @param {int64_t} n_queries 查询向量个数
     * @param {vector<float>&} queries 查询向量，[n_queries * dim]
     * @param {std::shared_ptr<SearchParams>} search_params 查询参数配置
     * @return {*} 查询结果
     */    
    std::shared_ptr<SearchResult> search(int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params);

    /**
     * @brief: 向索引中添加向量
     * @param {std::vector<float>&} vectors 待添加的向量
     * @param {std::vector<int64_t>&} ids 添加向量的ids
     * @return {*}
     */    
    void add(std::vector<float>& vectors, std::vector<int64_t>& ids);

    /**
     * @brief: 从索引中移除向量
     * @param {std::vector<int64_t>&} ids 要移除的向量ids
     * @return {*}
     */    
    void remove(std::vector<int64_t>& ids);

    int64_t ntotal();

    int64_t nlist();

    int dim();

    // ========= 以上为索引基础相关方法 =========
    
    // ========= 以下为索引维护相关方法 =========
    
    /**
     * @brief: 添加分区
     * @param {shared_ptr<Clustering>} partitions 包含要添加分区的聚类对象
     * @return {*} 插入分区的IDs
     */
    std::vector<int64_t> add_partitions(std::shared_ptr<Clustering> partitions);

    /**
     * @brief: 删除多个分区并重分配向量
     * @param {std::vector<int64_t>&} partition_ids 要删除的分区IDs
     * @return {*}
     */    
    void delete_partitions(const std::vector<int64_t>& partition_ids);

    /**
     * @brief: 获取要分类的分区IDs
     * @return {*} 待进行分裂的分区IDs
     */    
    std::vector<int64_t> get_ids_to_split();

    /**
     * @brief: 获取要进行缩减的分区IDs
     * @return {*} 待进行缩减的分区IDs
     */    
    std::vector<int64_t> get_ids_to_shrink();

    /**
     * @brief: 对一定半径范围内的分区进行重分配
     * @param {std::vector<int64_t>&} partition_ids 待重分配的分区IDs
     * @return {*}
     */    
    void local_reassign(std::vector<int64_t>& partition_ids);

    /**
     * @brief: 对分区进行重分配
     * @param {std::vector<int64_t>&} partition_ids 待冲分配的分区IDs
     * @return {*}
     */    
    void reassign(std::vector<int64_t>& partition_ids);
    
    /**
     * @brief: 重索引器执行维护策略
     * @return {*}
     */    
    void ReindexingPolicy();

    /**
     * @brief: 执行索引维护策略:DeDrift
     * @return {*}
     */    
    void DeDrift();
    
    /**
     * @brief: 执行索引维护策略:LIRE(增量轻量重平衡协议)
     * @return {*}
     */    
    void LIRE();

    /**
     * @brief: 执行索引维护策略:TreeLIRE(增量轻量重平衡协议)
     * @return {*}
     */    
    void TreeLIRE();

    /**
     * @brief: 执行索引维护策略:AdaIVF
     * @return {*}
     */    
    void AdaIVF();

    /**
     * @brief: 执行索引维护策略:CostModel
     * @return {*}
     */    
    void CM();
};

#endif