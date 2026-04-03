/*
 * @Author: Guyue
 * @Date: 2026-03-23 11:01:11
 * @LastEditTime: 2026-04-03 15:17:24
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/src/partition_manager.cpp
 */
#include <partition_manager.h>

PartitionManager::PartitionManager()
{
    partition_store_ = nullptr;
}

PartitionManager::~PartitionManager()
{
    // 底层分区存储自动释放内存
}

void PartitionManager::init_partitions(std::shared_ptr<Clustering> partitions, faiss::MetricType metric, std::shared_ptr<PQParams> pq_params)
{
    metric_ = metric;
    int64_t nlist = partitions->nlist();
    int64_t ntotal = partitions->ntotal();
    int dim = partitions->dimension;

    if (nlist <= 0 && ntotal <=0 )
    {
        throw std::runtime_error("[PartitionManager] init_partitions: nlist and ntotal is <= 0.");
    }

    //////////////////////////////////////////
    /// 初始化分区存储
    ////////////////////////////////////////// 
    size_t code_size_bytes = pq_params ? (dim * pq_params->bytes_per_dim) + pq_params->extra_bytes : static_cast<size_t>(dim * sizeof(float));
    partition_store_ = std::make_shared<faiss::DynamicInvertedLists>(0, code_size_bytes);
    partition_store_->dimension_ = dim;

    //////////////////////////////////////////
    /// 设置分区IDs & 添加空分区列表
    ////////////////////////////////////////// 
    partitions->partition_ids.resize(nlist);
    for (int64_t i = 0; i < nlist; ++i)
    {
        partitions->partition_ids[i]= i;
        partition_store_->add_list(i);
    }
    curr_partition_id_ = nlist;

    //////////////////////////////////////////
    /// 将向量插入各个分区
    ////////////////////////////////////////// 
    for (int64_t i = 0; i < nlist; ++i)
    {
        auto& v = partitions->vectors[i];       // vector<float> 向量数
        auto& id = partitions->vector_ids[i];   // vector<int64_t> 向量ID
        size_t count = id.size();
        if (count == 0) continue;
    
        if (pq_params)
        {
            std::vector<uint8_t> codes(count * code_size_bytes);
            for (size_t j = 0; j < count; ++j)
            {
                guyue::encode(v.data() + j * dim, codes.data() + j * code_size_bytes, dim);
                // guyue::encode_avx2(v.data() + j * dim, codes.data() + j * code_size_bytes, dim);
            }
            partition_store_->add_entries(i, count, id.data(), codes.data());

        } else {
            partition_store_->add_entries(i, count, id.data(), reinterpret_cast<const uint8_t*>(v.data()));
        }
    }
}

void PartitionManager::add(int64_t n_vectors, const std::vector<float>& vectors, const std::vector<int64_t>& vector_ids, const std::vector<int64_t>& assignments)
{
    /* TODO
     * 优化点：汇总插入到同一分区中的向量，然后并行处理(直接并行处理可能会出现两个线程同时修改一个分区的情况)
     */

    if (!partition_store_)
    {
        throw std::runtime_error("[PartitionManager] add: partition_store_ is null. Did you call init_partitions?");
    }
    
    if (vectors.empty() || vector_ids.empty())
    {
        throw std::runtime_error("[PartitionManager] add: vectors or vector_ids is empty.");
    }

    if (n_vectors == 0)
    {
        return;
    }

    //////////////////////////////////////////
    /// 每个向量的分区分配
    //////////////////////////////////////////    
    if (assignments.size() != n_vectors)
    {
        throw std::runtime_error("[PartitionManager] add: assignments size != vectors size.");
    }
    
    //////////////////////////////////////////
    /// 将向量添加到分区中
    //////////////////////////////////////////
    size_t code_size_bytes = partition_store_->code_size;
    const uint8_t* code_ptr = reinterpret_cast<const uint8_t*>(vectors.data());

    for (int64_t i = 0; i < n_vectors; ++i)
    {
        int64_t pid = assignments[i];
        partition_store_->add_entries(pid, 1, vector_ids.data() + i, code_ptr + i * code_size_bytes);
    }
}

void PartitionManager::update_centroids(int64_t n_vectors, const std::vector<float>& vectors, std::unordered_map<int64_t, int64_t>& partition_size, const std::vector<int64_t>& assignments, int delta)
{
    if (nlist() != 1)
    {
        throw std::runtime_error("[PartitionManager] update_centroids: only used for centroids_manager!");
    }

    int dim = vectors.size() / n_vectors;
    std::vector<float*> origin_vectors = partition_store_->get_vector_by_id(assignments);
    for (int i = 0; i < n_vectors; ++i)
    {
        int64_t p_id = assignments[i];
        for (size_t j = 0; j < dim; ++j)
        {
            origin_vectors[i][j] = (origin_vectors[i][j] * partition_size[p_id] + delta * vectors[i * dim + j]) / (partition_size[p_id] + delta);
        }
        partition_size[p_id] += delta;
    }
}

std::vector<float> PartitionManager::remove(const std::vector<int64_t>& ids, std::unordered_map<int64_t, std::vector<int64_t>>* assignments)
{
    /* TODO
     * 可能存在问题
     */

    if (!partition_store_)
    {
        throw std::runtime_error("[PartitionManager] remove: partition_store_ is null.");
    }

    std::unordered_set<faiss::idx_t> to_remove;
    std::unordered_map<faiss::idx_t, int64_t> ids_map;
    for (int64_t i = 0; i < ids.size(); ++i)
    {
        to_remove.insert(ids[i]);
        ids_map[ids[i]] = i;
    }

    return partition_store_->remove_vectors(to_remove, ids_map, assignments);
}

std::vector<float> PartitionManager::get_with_copy(const std::vector<int64_t>& ids, std::vector<int64_t>* assignment)
{
    const int dim = partition_store_->dimension_;
    const size_t n = ids.size();
    
    std::vector<float> vectors(n * dim);

#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < n; ++i)
    {
        int64_t p_id = partition_store_->get_vector_for_id(ids[i], vectors.data() + i * dim);
        if (assignment != nullptr)
        {
            (*assignment)[i] = p_id;
        }
    }

    return vectors;
}

std::vector<float*> PartitionManager::get_wo_copy(std::vector<int64_t> ids)
{
    return partition_store_->get_vector_by_id(ids);
}

std::shared_ptr<Clustering> PartitionManager::select_partitions(const std::vector<int64_t>& select_ids, bool copy)
{
    std::vector<std::vector<float>> partition_vectors(select_ids.size());
    std::vector<std::vector<int64_t>> partition_ids(select_ids.size());
    int dim = partition_store_->dimension_;

    for (int i = 0; i < select_ids.size(); ++i)
    {
        int64_t list_id = select_ids[i];
        int64_t list_size = partition_store_->list_size(list_id);
        if (list_size == 0)
        {
            partition_vectors[i] = {};
            partition_ids[i] = {};
            continue;
        }

        //////////////////////////////////////////
        /// 获取分区编码向量
        //////////////////////////////////////////
        auto codes = partition_store_->get_codes(list_id);
        auto ids = partition_store_->get_ids(list_id);

        //////////////////////////////////////////
        /// 分区向量
        //////////////////////////////////////////        
        if (copy)
        {
            // 优化1：使用memcpy代替copy
            std::vector<float> partition_vectors_i(list_size * dim);
            const float* src_vectors = reinterpret_cast<const float*>(codes);
            std::memcpy(partition_vectors_i.data(), src_vectors, list_size * dim * sizeof(float));

            // 优化2：使用emplace_back或直接赋值
            partition_vectors[i] = std::move(partition_vectors_i);

            // 优化3：预留空间并批量拷贝
            partition_ids[i].assign(ids, ids + list_size);
        } else {
            const float* src_vectors = reinterpret_cast<const float*>(codes);
            partition_vectors[i] = std::vector<float>(src_vectors, src_vectors + list_size * dim);
            partition_ids[i] = std::vector<int64_t>(ids, ids + list_size);
        }
    }

    std::shared_ptr<Clustering> partition = std::make_shared<Clustering>();
    partition->dimension = dim;
    partition->partition_ids = select_ids;
    partition->vectors = partition_vectors;
    partition->vector_ids = partition_ids;

    return partition;
}

std::vector<int64_t> PartitionManager::add_partitions(std::shared_ptr<Clustering> partitions)
{
    int64_t nlist = partitions->nlist();
    std::vector<int64_t> partition_ids(nlist);

    for (int64_t i = 0; i < nlist; ++i)
    {
        int64_t list_id = partitions->partition_ids[i] + curr_partition_id_;
        partition_store_->add_list(list_id);
        partition_store_->add_entries(
            list_id,
            partitions->vector_ids[i].size(),
            partitions->vector_ids[i].data(),
            reinterpret_cast<uint8_t*>(partitions->vectors[i].data())
        );
        partition_ids[i] = list_id;
    }
    curr_partition_id_ += nlist;

    return partition_ids;
}

void PartitionManager::delete_partitions(const std::vector<int64_t>& delete_ids, bool reassign)
{
    for (int i = 0; i < delete_ids.size(); ++i)
    {
        int64_t list_id = delete_ids[i];
        partition_store_->remove_list(list_id);
    }
}

std::shared_ptr<Clustering> PartitionManager::reindexing_partitions(const std::vector<int64_t>& reindexing_ids, int k, int niter, std::vector<float> initial_centroids)
{
    int dim = partition_store_->dimension_;
    int64_t reindexing_size = reindexing_ids.size();

    // 统计截至到第i分区时的向量总数
    std::vector<int64_t> lists_size(reindexing_size + 1);
    lists_size[0] = 0;

#pragma omp parallel for
    for (int i = 0; i < reindexing_size; ++i)
    {
        lists_size[i + 1] = partition_store_->list_size(reindexing_ids[i]);
    }

    for (int i = 1; i <= reindexing_size; ++i)
    {
        lists_size[i] += lists_size[i - 1];
    }
    
    int64_t total_vectors = lists_size[reindexing_size];

    // 汇总IDs包含的分区向量
    std::vector<float> vectors(total_vectors * dim);
    std::vector<int64_t> ids(total_vectors);
    
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < reindexing_ids.size(); ++i)
    {
        int64_t list_no = reindexing_ids[i];
        int64_t list_size = lists_size[i + 1] - lists_size[i];

        if (list_size == 0)
        {
            continue;
        }

        const float* src_codes = reinterpret_cast<const float*>(partition_store_->get_codes(list_no));
        const int64_t* src_ids = partition_store_->get_ids(list_no);

        std::memcpy(vectors.data() + lists_size[i] * dim, src_codes, list_size * dim * sizeof(float));
        std::memcpy(ids.data() + lists_size[i], src_ids, list_size * sizeof(int64_t));
    }

    return kmeans(ids.size(), vectors, ids, k, metric_, niter, initial_centroids);
}

std::shared_ptr<Clustering> PartitionManager::split_partitions(const std::vector<int64_t>& partition_ids, int64_t num_splits, int niter)
{
    int64_t num_partition_split = partition_ids.size();
    int64_t total_new_partitions = num_partition_split * num_splits;
    int dim = partition_store_->dimension_;

    std::vector<float> split_centroids(total_new_partitions * dim, 0);
    std::vector<std::vector<float>> split_vectors;
    std::vector<std::vector<int64_t>> split_ids;
    
    // new
    // std::vector<float> split_error;

    split_vectors.resize(total_new_partitions);
    split_ids.resize(total_new_partitions);
    
    // new
    // split_error.resize(total_new_partitions);
    
    std::shared_ptr<Clustering> split_partitions = select_partitions(partition_ids);

#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < num_partition_split; ++i)
    {
        assert(split_partitions->cluster_size_of(i) >= 4 && "[PartitionManager] split_partition : Partition must have at least 8 vectors to split.");
        std::shared_ptr<Clustering> curr_split_clustering = kmeans(
            split_partitions->cluster_size_of(i),
            split_partitions->vectors[i],
            split_partitions->vector_ids[i],
            num_splits,
            metric_,
            niter
        );

        for (size_t j = 0; j < curr_split_clustering->nlist(); ++j)
        {
            for (size_t k = 0; k < dim; ++k)
            {
                split_centroids[i * num_splits * dim + j * dim + k] = curr_split_clustering->centroids[j * dim + k];
            }
            
            split_vectors[i * num_splits + j] = curr_split_clustering->vectors[j];
            split_ids[i * num_splits + j] = curr_split_clustering->vector_ids[j];
            
            // new 
            // split_error[i * num_splits + j] = curr_split_clustering->errors[j];
        }
    }

    std::shared_ptr<Clustering> partition = std::make_shared<Clustering>();
    partition->dimension = dim;
    partition->centroids = split_centroids;
    for (int i = 0; i < total_new_partitions; ++i)
    {
        partition->partition_ids.push_back(i);
    }
    partition->vectors = split_vectors;
    partition->vector_ids = split_ids;
    
    // new 
    // partition->errors = split_error;

    return partition;
}

int64_t PartitionManager::ntotal() const
{
    if (!partition_store_)
    {
        return 0;
    }
    return partition_store_->ntotal();
}

int64_t PartitionManager::nlist() const
{
    if (!partition_store_)
    {
        return 0;
    }
    return partition_store_->nlist;
}

int PartitionManager::d() const
{
    if (!partition_store_)
    {
        return 0;
    }
    return partition_store_->dimension_;
}

std::vector<int64_t> PartitionManager::get_partitions_sizes(std::vector<int64_t> partition_ids)
{
    std::vector<int64_t> partition_sizes;
    for (int64_t partition_id : partition_ids)
    {
        partition_sizes.push_back(partition_store_->list_size(partition_id));
    }
    return partition_sizes;
}

int64_t PartitionManager::get_partition_size(int64_t partition_id)
{
    return partition_store_->list_size(partition_id);
}

std::vector<int64_t> PartitionManager::get_partitions_ids()
{
    return partition_store_->get_partition_ids();
}


std::vector<int64_t> PartitionManager::get_ids()
{
    std::vector<int64_t> partition_ids = get_partitions_ids();
    std::vector<int64_t> ids;

    for (int i = 0; i < partition_ids.size(); ++i)
    {
        int64_t list_no = partition_ids[i];
        size_t list_size = partition_store_->list_size(list_no);
        if (list_size == 0) continue;

        const int64_t* partition_id_ptr = partition_store_->get_ids(list_no);
        ids.insert(ids.end(), partition_id_ptr, partition_id_ptr + list_size);
    }
    return ids;
}

std::vector<int64_t> PartitionManager::get_ids(int64_t partition_id)
{
    std::vector<int64_t> partition_ids = get_partitions_ids();
    std::vector<int64_t> ids;

    size_t list_size = partition_store_->list_size(partition_id);
    const int64_t* partition_id_ptr = partition_store_->get_ids(partition_id);
    ids.insert(ids.end(), partition_id_ptr, partition_id_ptr + list_size);

    return ids;
}
