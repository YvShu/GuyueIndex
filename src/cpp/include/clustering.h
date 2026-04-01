/*
 * @Author: Guyue
 * @Date: 2025-11-12 15:17:57
 * @LastEditTime: 2026-04-01 16:25:29
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/clustering.h
 */
#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <common.h>

struct Clustering
{
    int dimension;                                  // 向量维度    
    std::vector<float> centroids;                   // 聚类中心, [nlist*dimension]
    std::vector<int64_t> partition_ids;             // 聚类ID, [nlist]
    std::vector<std::vector<float>> vectors;        // 聚类向量, [nlist*[count*dimension]]
    std::vector<std::vector<int64_t>> vector_ids;   // 向量IDs, [nlist*count]
    // new
    // std::vector<std::vector<float>> dists;          // 向量到其所属分区的距离, [nlist*count]
    // std::vector<float> errors;                      // 聚类误差, [nlist]

    /**
     * @brief: 获取聚类中的向量总数
     * @return {*}
     */    
    int64_t ntotal() const
    {
        int64_t n = 0;
        for (const auto& v : vectors)
        {
            n += v.size() / dimension;
        }
        return n;
    }

    /**
     * @brief: 获取聚类的个数
     * @return {*}
     */    
    int64_t nlist() const
    {
        return vectors.size();
    }

    /**
     * @brief: 获取指定的某个聚类的大小
     * @param {int64_t} i 聚类ID
     * @return {*}
     */    
    int64_t cluster_size_of(int64_t i) const
    {
        return vectors[i].size() / dimension;
    }
};

/**
 * @brief: 使用faiss-kmeans对向量聚类
 * @param {int} n_vector 向量个数
 * @param {vector<float>} vectors 进行聚类的向量
 * @param {vector<int64_t>} ids 向量IDs
 * @param {int} n_clusters 聚类数量
 * @param {MetricType} metric_type 度量类型
 * @param {int} niter 运行k-means的迭代次数
 * @param {vector<float>} initial_centroids 用于k-means的初始中心
 * @return {*}
 */
std::shared_ptr<Clustering> kmeans(
    int n_vectors,
    std::vector<float> vectors,
    std::vector<int64_t> ids,
    int nlist,
    faiss::MetricType metric_type,
    int niter = 10,
    std::vector<float> initial_centroids = {}
);


#endif // CLUSTERING_H