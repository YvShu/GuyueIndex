/*
 * @Author: Guyue
 * @Date: 2026-03-24 13:01:41
 * @LastEditTime: 2026-03-24 15:30:12
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/partition_tree.h
 */
#ifndef PARTITION_TREE_H
#define PARTITION_TREE_H

#include <common.h>
#include <clustering.h>

/**
 * @brief: 分区树节点
 */
struct Node
{
    int64_t ID_;
    std::vector<float> centroid_;
    Node* parent_;
    std::vector<Node*> children_;

    ~Node()
    {
        // 递归删除子节点
        for (Node* child : children_)
        {
            delete child;
        }
    }
};

/**
 * @brief: 束搜索时存储结果的结构
 */
struct BeamNode
{
    float dist_;
    Node* node_;

    bool operator>(const BeamNode& other) const
    {
        return dist_ > other.dist_;
    }
};

class PartitionTree {
public:
    Node* root_;
    std::unordered_map<int64_t, Node*> leaves_;
    int64_t curr_partition_id_ = 0;

    /**
     * @brief: 分区树构造函数(初始化一个空树结构)
     * @return {*}
     */    
    PartitionTree();

    /**
     * @brief: 分区树构造函数(初始只有一个分区)
     * @param {std::vector<float>&} centroid 分区中心
     * @return {*}
     */    
    PartitionTree(std::vector<float>& centroid);

    /**
     * @brief: 分区树构造函数(使用多个分区进行初始化)
     * @param {shared_ptr<Clustering>} partitions 分区信息
     * @return {*}
     */    
    PartitionTree(std::shared_ptr<Clustering> partitions);

    ~PartitionTree();

    void build(Node* root, const std::vector<float>& vectors, const std::vector<int64_t>& ids, int n_clusters);

    /**
     * @brief: 对叶子节点进行分裂
     * @param {std::vector<int64_t>&} ids 要分裂的节点IDs
     * @param {std::vector<float>&} centroids 节点中心
     * @param {std::vector<int64_t>&} assignment 分裂后节点的所属情况
     * @return {*}
     */    
    void split(const std::vector<int64_t>& IDs, std::vector<float>& centroids, const std::vector<int64_t>& assignment);

    /**
     * @brief: 对叶子节点进行删除
     * @param {std::vector<int64_t>&} IDs 待删除的节点
     * @return {*}
     */    
    void shrink(const std::vector<int64_t>& IDs);

    /**
     * @brief: 获取根节点
     * @return {*}
     */    
    Node* root() const;
};

#endif // PARTITION_TREE_H