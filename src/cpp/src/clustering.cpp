/*
 * @Author: Guyue
 * @Date: 2026-03-23 11:03:41
 * @LastEditTime: 2026-04-01 16:23:28
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/src/clustering.cpp
 */
#include <clustering.h>
#include <common.h>
#include <faiss/IndexFlat.h>
#include <faiss/Clustering.h>
#include <partition_base.h>

std::shared_ptr<Clustering> kmeans(
    int n_vectors,
    std::vector<float> vectors,
    std::vector<int64_t> ids,
    int nlist,
    faiss::MetricType metric_type,
    int niter,
    std::vector<float>
)
{
    assert(n_vectors == ids.size());
    
    int d = vectors.size() / n_vectors;
    faiss::Index* index_ptr = nullptr;
    if (metric_type == faiss::METRIC_INNER_PRODUCT)
    {
        index_ptr = new faiss::IndexFlatIP(d);
    } else {
        index_ptr = new faiss::IndexFlatL2(d);
    }

    //////////////////////////////////////////
    /// 向量聚类
    //////////////////////////////////////////
    faiss::ClusteringParameters cp;
    cp.niter = niter;
    if (metric_type == faiss::METRIC_INNER_PRODUCT)
    {
        cp.spherical = true;
    }
    faiss::Clustering clus(d, nlist, cp);
    clus.train(n_vectors, vectors.data(), *index_ptr);

    //////////////////////////////////////////
    /// 向量分配(到最近的聚类)
    //////////////////////////////////////////
    std::vector<faiss::idx_t> assign_vec(n_vectors);
    std::vector<float> distance_vec(n_vectors);
    index_ptr->search(n_vectors, vectors.data(), 1, distance_vec.data(), assign_vec.data());

    //////////////////////////////////////////
    /// 填充分区向量和ids
    //////////////////////////////////////////
    std::vector<std::vector<float>> partition_vectors(nlist);   // 每个分区包含的向量
    std::vector<std::vector<int64_t>> partition_ids(nlist);     // 每个分区包含的向量id
    std::vector<std::vector<float>> partition_dists(nlist);     // 每个分区向量距离中心的距离
    // new
    // std::vector<float> partition_errors(nlist, 0);              // 每个分区的误差
    std::vector<int64_t> partition_size(nlist, 0);              // 每个分区的大小
    
    for (int i = 0; i < n_vectors; ++i)
    {
        if (assign_vec[i] >= 0 && assign_vec[i] < nlist)
        {
            partition_size[assign_vec[i]]++;
        }
    }
    for (int i = 0; i < nlist; ++i)
    {
        partition_vectors[i].reserve(partition_size[i] * d);
        partition_ids[i].reserve(partition_size[i]);
        partition_dists[i].reserve(partition_size[i]);
    }
    for (int i = 0; i < n_vectors; ++i)
    {
        const int idx = assign_vec[i];
        if (idx >=0 && idx < nlist)
        {
            for (int j = 0; j < d; ++j)
            {
                partition_vectors[idx].push_back(vectors[i * d + j]);
            }
            partition_ids[idx].push_back(ids[i]);
            partition_dists[idx].push_back(distance_vec[i]);
            // new
            // partition_errors[idx] += distance_vec[i] / partition_size[idx];
        }
    }

    //////////////////////////////////////////
    /// 生成Clustering对象
    //////////////////////////////////////////    
    std::shared_ptr<Clustering> clustering = std::make_shared<Clustering>();
    clustering->dimension = d;
    clustering->centroids = std::move(clus.centroids);
    clustering->partition_ids.resize(nlist);
    for (int i = 0; i < nlist; ++i)
    {
        clustering->partition_ids[i] = i;
    }
    clustering->vectors = std::move(partition_vectors);
    clustering->vector_ids = std::move(partition_ids);
    // new
    // clustering->dists = std::move(partition_dists);
    // clustering->errors = std::move(partition_errors);

    delete index_ptr;
    return clustering;
}
