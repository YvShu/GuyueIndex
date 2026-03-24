/*
 * @Author: Guyue
 * @Date: 2026-03-23 10:54:47
 * @LastEditTime: 2026-03-23 11:08:58
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/partition_manager.h
 */
#ifndef PARTITION_MANAGER_H
#define PARTITION_MANAGER_H

#include <common.h>
#include <clustering.h>
#include <dynamic_inverted_lists.h>

class PartitionManager {
public:
    std::shared_ptr<faiss::DynamicInvertedLists> partition_store_ = nullptr;
    int64_t curr_partition_id_ = 0;
    faiss::MetricType metric_;
    std::set<int64_t> resident_ids;
    std::vector<float> mu;
    std::vector<std::vector<float>> covs;

    /**
     * @brief: PartitionManager的构造函数
     * @return {*}
     */    
    PartitionManager();

    /**
     * @brief: PartitionManager的析构函数
     * @return {*}
     */    
    ~PartitionManager();

    /**
     * @brief: 利用聚类初始化分区
     * @param {shared_ptr<Clustering>} partitions 包含要初始化的分区的聚类对象
     * @param {MetricType} metric 距离度量
     * @return {*}
     */    
    void init_partitions(std::shared_ptr<Clustering> partitions, faiss::MetricType metric);

    /**
     * @brief: 将向量添加到相应的分区中
     * @param {int64_t} n_vectors 添加向量个数
     * @param {vector<float>} vectors 向量, [n_vectors * dim]
     * @param {vector<int64_t>} vector_ids 向量ids
     * @param {vector<float>} assignments 包含n_vectors个向量所分配到的分区ID
     * @return {*}
     */    
    void add(int64_t n_vectors, const std::vector<float>& vectors, const std::vector<int64_t>& vector_ids, const std::vector<int64_t>& assignments);

    /**
     * @brief: 修改分区中心(只对中心管理器生效)
     * @param {int64_t} n_vectors 向量个数
     * @param {vector<float>} vectors 向量, [n_vectors * dim]
     * @param {vector<int64_t>} partition_size 分区大小
     * @param {vector<int64_t>} assignments 向量分配
     * @param {int} delta 1表示添加、-1表示删除
     * @return {*}
     */    
    void update_centroids(int64_t n_vectors, const std::vector<float>& vectors, std::unordered_map<int64_t, int64_t>& partition_size, const std::vector<int64_t>& assignments, int delta = 1);

    /**
     * @brief: 通过ids将向量移除
     * @param {vector<int64_t>} ids 要移除向量的ids
     * @param {unordered_map<int64_t, std::vector<int64_t>>} assignments 包含n_vectors个向量所分配到的分区ID【可选】
     * @return {vector<float>} 要删除的向量拷贝【如果assignment不为nullptr】
     */
    std::vector<float> remove(const std::vector<int64_t>& ids, std::unordered_map<int64_t, std::vector<int64_t>>* assignments = nullptr);

    /**
     * @brief: 通过向量ids获取向量(结果拷贝)
     * @param {vector<int64_t>} ids 向量ids
     * @param {vector<int64_t>} assignment 记录向量所属分区(可选)
     * @return {*} 向量ids对应的向量
     */    
    std::vector<float> get_with_copy(const std::vector<int64_t>& ids, std::vector<int64_t>* assignment = nullptr);

    /**
     * @brief: 通过向量ids获取指向向量的指针(非拷贝)
     * @param {vector<int64_t>} ids 向量ids
     * @return {*} 向量ids对应的向量指针
     */    
    std::vector<float*> get_wo_copy(std::vector<int64_t> ids);

    /**
     * @brief: 选择指定分区
     * @param {vector<int64_t>} select_ids 分区IDs
     * @param {bool} copy 是否进行拷贝，为True则拷贝数据，否则返回引用
     * @return {*} 指定分区IDs对应的分区
     */    
    std::shared_ptr<Clustering> select_partitions(const std::vector<int64_t>& select_ids, bool copy = false);

    /**
     * @brief: 添加一个新的分区
     * @param {shared_ptr<Clustering>} partitions 要添加的分区对象
     * @return {*} 新添加分区的IDs
     */    
    std::vector<int64_t> add_partitions(std::shared_ptr<Clustering> partitions);

    /**
     * @brief: 删除指定分区
     * @param {vector<int64_t>} delete_ids 要删除的分区IDs
     * @param {bool} reassign 是否进行重分配
     * @return {*}
     */    
    void delete_partitions(const std::vector<int64_t>& delete_ids, bool reassign = false);

    /**
     * @brief: 维护指定分区
     * @param {vector<int64_t>} refine_ids 向量IDs
     * @param {int} refine_iterations 维护迭代次数
     * @return {*}
     */    
    std::shared_ptr<Clustering> reindexing_partitions(const std::vector<int64_t>& reindexing_ids, int k, int niter = 5, std::vector<float> initial_centroids = {});

    /**
     * @brief: 将指定分区分裂为小分区
     * @param {vector<int64_t>} split_ids 需要进行分裂的分区列表
     * @param {int} niter 分裂迭代次数
     * @return {*} 分裂后的分区
     */    
    std::shared_ptr<Clustering> split_partitions(const std::vector<int64_t>& partition_ids, int64_t num_splits, int niter = 5);

    /**
     * @brief: 获取所有分区的向量总数
     * @return {*} 向量数量
     */    
    int64_t ntotal() const;

    /**
     * @brief: 获取当前分区数量
     * @return {*} 分区数量
     */    
    int64_t nlist() const;

    /**
     * @brief: 获取向量维度
     * @return {*} 向量维度
     */    
    int d() const;

    /**
     * @brief: 获取指定分区的大小(多个)
     * @param {vector<int64_t>} partition_ids 分区IDs
     * @return {*} 分区IDs对应的分区大小
     */    
    std::vector<int64_t> get_partitions_sizes(std::vector<int64_t> partition_ids);

    /**
     * @brief: 获取指定分区的大小(单个)
     * @param {int64_t} partition_id 分区ID
     * @return {*} 分区ID对应的分区大小
     */    
    int64_t get_partition_size(int64_t partition_id);

    /**
     * @brief: 获取分区IDs，分区删除时不交换，IDs可能不是连续的[0,1,...]
     * @return {*} 当前分区IDs
     */    
    std::vector<int64_t> get_partitions_ids();

    /**
     * @brief: 获取向量ids
     * @return {*} 向量ids
     */    
    std::vector<int64_t> get_ids();

    /**
     * @brief: 获取指定分区中的向量ids
     * @param {int64_t} partition_id 分区ID
     * @return {*} 分区ID中的向量ids
     */
    std::vector<int64_t> get_ids(int64_t partition_id);
};
#endif // PARTITION_MANAGER_H