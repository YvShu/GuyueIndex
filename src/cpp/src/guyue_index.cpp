/*
 * @Author: Guyue
 * @Date: 2026-03-23 10:00:04
 * @LastEditTime: 2026-04-08 18:12:12
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/src/guyue_index.cpp
 */
#include <guyue_index.h>

GuyueIndex::GuyueIndex()
{
    centroids_manager_ = nullptr;
    partition_manager_ = nullptr;
    searcher_ = nullptr;
    reindexing_params_ = nullptr;
    pq_params_ = nullptr;
}

GuyueIndex::~GuyueIndex()
{
    centroids_manager_ = nullptr;
    partition_manager_ = nullptr;
    searcher_ = nullptr;
    reindexing_params_ = nullptr;
    pq_params_ = nullptr;
}

void GuyueIndex::build(std::vector<float>& vectors, std::vector<int64_t>& ids, std::shared_ptr<IndexBuildParams> build_params, std::shared_ptr<ReindexingParams> reindexing_params, std::shared_ptr<PQParams> pq_params)
{
    metric_ = str_to_metric_type(build_params->metric);
    tree_build_ = build_params->tree_build;
    int64_t n_vectors = ids.size();

    if (pq_params)
    {
        pq_params_ = std::make_shared<PQParams>();
        pq_params_ = pq_params;
    }

    if (n_vectors == 0) // 从空开始构建
    {
        //////////////////////////////////////////
        /// 初始一个空分区
        //////////////////////////////////////////
        std::vector<float> centroid(build_params->dimension, 0.0f);

        std::shared_ptr<Clustering> partitions = std::make_shared<Clustering>();
        partitions->dimension = build_params->dimension;
        partitions->partition_ids = {0};
        partitions->vectors = {vectors};
        partitions->vector_ids = {ids};
        partitions->centroids = centroid;
        partition_manager_ = std::make_shared<PartitionManager>();
        partition_manager_->init_partitions(partitions, metric_, pq_params);

        //////////////////////////////////////////
        /// 初始化中心管理器
        //////////////////////////////////////////
        std::shared_ptr<Clustering> centroids = std::make_shared<Clustering>();
        centroids->dimension = build_params->dimension;
        centroids->partition_ids = {0};
        centroids->vectors = {centroid};
        centroids->vector_ids = {partitions->partition_ids};
        centroids_manager_ = std::make_shared<PartitionManager>();
        centroids_manager_->init_partitions(centroids, metric_);

        //////////////////////////////////////////
        /// 初始化分区树
        //////////////////////////////////////////        
        if (build_params->tree_build)
        {
            partition_tree_ = std::make_shared<PartitionTree>(centroids);
        };
    } else {
        //////////////////////////////////////////
        /// 聚类得到分区信息
        //////////////////////////////////////////
        std::shared_ptr<Clustering> partitions = kmeans(
            n_vectors, 
            vectors, 
            ids, 
            build_params->nlist, 
            metric_, 
            build_params->niter
        );
        partitions->dimension = build_params->dimension;

        //////////////////////////////////////////
        /// 利用聚类结果构建分区管理器
        //////////////////////////////////////////
        partition_manager_ = std::make_shared<PartitionManager>();
        partition_manager_->init_partitions(partitions, metric_, pq_params);

        //////////////////////////////////////////
        /// 利用聚类结果构建中心管理器
        //////////////////////////////////////////
        std::shared_ptr<Clustering> centroids = std::make_shared<Clustering>();
        centroids->dimension = build_params->dimension;
        centroids->partition_ids = {0};
        centroids->vectors = {partitions->centroids};
        centroids->vector_ids = {partitions->partition_ids};
        centroids_manager_ = std::make_shared<PartitionManager>();
        centroids_manager_->init_partitions(centroids, metric_);
        // new
        // for (int i = 0; i < partitions->nlist(); ++i)
        // {
        //     centroids_manager_->errors_[partitions->partition_ids[i]] = partitions->errors[i];
        // }
    }

    //////////////////////////////////////////
    /// 构建查询器
    //////////////////////////////////////////
    searcher_ = std::make_shared<Searcher>(metric_);

    //////////////////////////////////////////
    /// 维护参数
    //////////////////////////////////////////
    if (reindexing_params->reindexing_strategy == "None")
    {

    } else if (reindexing_params->reindexing_strategy == "DeDrift") {

    } else if (reindexing_params->reindexing_strategy == "LIRE") {

    } else if (reindexing_params->reindexing_strategy == "Tree-LIRE") {
    
    } else if (reindexing_params->reindexing_strategy == "Error-LIRE") {

    } else if (reindexing_params->reindexing_strategy == "Ada-IVF") {
        
    } else if (reindexing_params->reindexing_strategy == "CM") {

    } else {
        throw std::runtime_error("[GuyueIndex] build: The maintenance strategy does not exist.");
    }
    reindexing_params_ = std::make_shared<ReindexingParams>();
    reindexing_params_ = reindexing_params;
}

std::shared_ptr<SearchResult> GuyueIndex::search(int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params)
{
    if (!searcher_)
    {
        throw std::runtime_error("[GuyueIndex] search: No searcher. Did you build the index?");
    }

    //////////////////////////////////////////
    /// 一阶段：执行对分区中心的查找
    //////////////////////////////////////////
    auto centroids_search_result = std::make_shared<SearchResult>();
    auto centroids_search_params = std::make_shared<SearchParams>();
    centroids_search_params->k = std::min(search_params->nprobe, (int) nlist());
    centroids_search_result = searcher_->search_centers(centroids_manager_, n_queries, queries, centroids_search_params);

    //////////////////////////////////////////
    /// 二阶段：执行对分区的查找
    //////////////////////////////////////////
    auto search_results = std::make_shared<SearchResult>();
    // search_results = searcher_->search_partitions(partition_manager_, n_queries, queries, centroids_search_result->indices, search_params, pq_params_);
    search_results = searcher_->search_partitions_batch(partition_manager_, n_queries, queries, centroids_search_result->indices, search_params, pq_params_);

    //////////////////////////////////////////
    /// 查询信息统计
    //////////////////////////////////////////
    search_results->c_search_time = centroids_search_result->c_search_time;
    search_results->search_time = search_results->c_search_time + search_results->p_search_time;


    // int d = dim();
    // auto search_results = std::make_shared<SearchResult>();
    // search_results->distances.resize(n_queries);
    // search_results->indices.resize(n_queries);

    // size_t kBatchSize = div_roundup(n_queries, search_params->num_threads);
    // parallel_for((size_t) 0, div_roundup(n_queries, kBatchSize), [&](size_t i)
    // {
    //     size_t begin = kBatchSize * i;
    //     size_t curSize = std::min(n_queries - begin, kBatchSize);
    //     std::vector<float> queryCopy(queries.begin() + begin * d, queries.begin() + (begin + curSize) * d);
    
    //     auto centroids_search_result = std::make_shared<SearchResult>();
    //     auto centroids_search_params = std::make_shared<SearchParams>();
    //     centroids_search_params->k = std::min(search_params->nprobe, (int) nlist());
    //     centroids_search_result = searcher_->search_centers(centroids_manager_, curSize, queryCopy, centroids_search_params);
        
    //     auto local_search_results = std::make_shared<SearchResult>();
    //     local_search_results = searcher_->search_partitions_acc(partition_manager_, curSize, queryCopy, centroids_search_result->indices, search_params, pq_params_);
    //     // local_search_results = searcher_->search_partitions(partition_manager_, curSize, queryCopy, centroids_search_result->indices, search_params, pq_params_);

    //     for (int j = 0; j < curSize; ++j)
    //     {
    //         search_results->distances[begin + j].resize(search_params->k);
    //         search_results->indices[begin + j].resize(search_params->k);
    //         for (int l = 0; l < search_params->k; ++l)
    //         {
    //             search_results->distances[begin + j][l] = local_search_results->distances[j][l];
    //             search_results->indices[begin + j][l] = local_search_results->indices[j][l];
    //         }
    //         // search_results->distances[begin + j] = local_search_results->distances[j];
    //         // search_results->indices[begin + j] = local_search_results->indices[j];
    //     }
    // }, search_params->num_threads);

    return search_results;
}

void GuyueIndex::add(std::vector<float>& vectors, std::vector<int64_t>& ids, bool reassign)
{
    if (!centroids_manager_ || !partition_manager_)
    {
        throw std::runtime_error("[GuyueIndex] add : No centroids_manager or partition_manager.");
    }

    //////////////////////////////////////////
    /// 查找向量要插入分区
    //////////////////////////////////////////
    int d = dim();
    int64_t n_vectors = ids.size();
    auto insert_search_result = std::make_shared<InsertSearchResult>(); // <分区ID，<属于该分区的向量ids>>
    auto search_params = std::make_shared<SearchParams>();
    search_params->k = 1;
    if (!tree_build_)
    {
        insert_search_result = searcher_->search_insert(centroids_manager_, n_vectors, vectors, search_params);
    } else {
        insert_search_result = searcher_->search_tree(partition_tree_, n_vectors, vectors, search_params);
    }

    //////////////////////////////////////////
    /// 插入向量到对应的分区
    //////////////////////////////////////////
    std::vector<int64_t> partitions_ids = partition_manager_->get_partitions_ids();
    size_t code_size_bytes = pq_params_ ? (d * pq_params_->bytes_per_dim) + pq_params_->extra_bytes : static_cast<size_t>(d * sizeof(float));
    const uint8_t* code_ptr = reinterpret_cast<const uint8_t*>(vectors.data());

    if (reindexing_params_->centroids_update && !reassign)
    {
    #pragma omp parallel for schedule(dynamic)
        for (int64_t p = 0; p < partitions_ids.size(); ++p)
        {
            if (insert_search_result->assignment.find(partitions_ids[p]) == insert_search_result->assignment.end())
            {
                continue;
            }
            int64_t p_id = partitions_ids[p];
            int64_t p_size = insert_search_result->assignment[p_id].size();

            for (int64_t i = 0; i < p_size; ++i)
            {
                int64_t idx = insert_search_result->assignment[p_id][i];
                if (pq_params_)
                {
                    std::vector<uint8_t> codes(code_size_bytes);
                    guyue::encode(vectors.data() + idx * d, codes.data(), d);
                    partition_manager_->partition_store_->add_entries(p_id, 1, ids.data() + idx, codes.data());
                } else {
                    partition_manager_->partition_store_->add_entries(p_id, 1, ids.data() + idx, code_ptr + idx * code_size_bytes);
                }
            }
            
            //////////////////////////////////////////
            /// 修改分区中心状态
            //////////////////////////////////////////
            if (reindexing_params_->centroids_update)
            {
                update_centroids(p_size, p_id, vectors, insert_search_result->assignment[p_id]);
            }
        }  
    } else {
    #pragma omp parallel for schedule(dynamic)
        for (int64_t p = 0; p < partitions_ids.size(); ++p)
        {
            if (insert_search_result->assignment.find(partitions_ids[p]) == insert_search_result->assignment.end())
            {
                continue;
            }
            
            int64_t p_id = partitions_ids[p];
            int64_t p_size = insert_search_result->assignment[p_id].size();

            // // 开辟局部连续缓冲区进行拼装
            // std::vector<faiss::idx_t> local_ids(p_size);
            // std::vector<uint8_t> local_codes(p_size * code_size_bytes);
            // for (int64_t i = 0; i < p_size; ++i)
            // {
            //     int64_t idx = insert_search_result->assignment[p_id][i];
            //     local_ids[i] = ids[idx];
            //     std::memcpy(local_codes.data() + i * code_size_bytes, 
            //                 code_ptr + idx * code_size_bytes, 
            //                 code_size_bytes);
            // }
            // partition_manager_->partition_store_->add_entries(p_id, p_size, local_ids.data(), local_codes.data());
            
            // new
            // float delta_error = 0;

            // 向量ids离散，需要逐个插入
            for (int64_t i = 0; i < p_size; ++i)
            {
                int64_t idx = insert_search_result->assignment[p_id][i];
                if (pq_params_)
                {
                    std::vector<uint8_t> codes(code_size_bytes);
                    guyue::encode(vectors.data() + idx * d, codes.data(), d);
                    partition_manager_->partition_store_->add_entries(p_id, 1, ids.data() + idx, codes.data());
                } else {
                    partition_manager_->partition_store_->add_entries(p_id, 1, ids.data() + idx, code_ptr + idx * code_size_bytes);
                }
                // new
                // delta_error += insert_search_result->dists[idx] / p_size;
            }

            // new
            // float old_error = centroids_manager_->errors_[p_id];
            // float n = partition_manager_->get_partition_size(p_id);
            // centroids_manager_->errors_[p_id] = old_error + p_size / n * (delta_error - old_error);
        } 
    }
}

void GuyueIndex::remove(std::vector<int64_t>& ids)
{
    if (!partition_manager_)
    {
        throw std::runtime_error("[GuyueIndex] add : No partition_manager.");
    }

    if (reindexing_params_->centroids_update)
    {
        std::unordered_map<int64_t, std::vector<int64_t>> assignment;
        std::vector<float> vectors = partition_manager_->remove(ids, &assignment);
        
        //////////////////////////////////////////
        /// 修改分区中心状态
        //////////////////////////////////////////
        if (reindexing_params_->centroids_update)
        {
            std::vector<int64_t> partitions_ids = partition_manager_->get_partitions_ids();

        #pragma omp parallel for schedule(dynamic)
            for (int64_t p = 0; p < partitions_ids.size(); ++p)
            {
                if (assignment.find(partitions_ids[p]) == assignment.end())
                {
                    continue;
                }
                int64_t p_id = partitions_ids[p];
                int64_t p_size = assignment[p_id].size();

                update_centroids(p_size, p_id, vectors, assignment[p_id], -1);
            }
        }
    } else {
        partition_manager_->remove(ids);
    }
}

int64_t GuyueIndex::ntotal()
{
    if (partition_manager_)
    {
        return partition_manager_->ntotal();
    }
    return 0;
}

int64_t GuyueIndex::nlist()
{
    if (partition_manager_)
    {
        return partition_manager_->nlist();
    }
    return 0;
}

int GuyueIndex::dim()
{
    if (partition_manager_)
    {
        return partition_manager_->d();
    }
    return 0;
}

void GuyueIndex::update_centroids(int64_t n_vectors, int64_t partition_id, const std::vector<float>& vectors, std::vector<int64_t> vectors_ids, int delta)
{
    int d = dim();
    float n = n_vectors;
    std::vector<float*> origin_centroids = centroids_manager_->get_wo_copy({partition_id});
    std::vector<float> delta_centroids(d, 0.0);
    int64_t partition_size = partition_manager_->get_partition_size(partition_id);

    for (int i = 0; i < d; ++i)
    {
        for (int j = 0; j < n_vectors; ++j)
        {
            delta_centroids[i] += vectors[vectors_ids[j] * d + i];
        }
        delta_centroids[i] /= n;
    }

    for (int i = 0; i < d; ++i)
    {
        origin_centroids[0][i] = origin_centroids[0][i] + delta * n / partition_size * (delta_centroids[i] - origin_centroids[0][i]);
    }
}

std::vector<int64_t> GuyueIndex::add_partitions(std::shared_ptr<Clustering> partitions)
{
    int64_t nlist = partitions->nlist();

    //////////////////////////////////////////
    /// 执行对分区管理器(partition_manager_)的添加
    //////////////////////////////////////////
    std::vector<int64_t> partition_ids = partition_manager_->add_partitions(partitions);

    //////////////////////////////////////////
    /// 执行对中心管理器(centroids_manager_)的添加
    //////////////////////////////////////////
    std::vector<int64_t> assignment(partitions->partition_ids.size(), 0);
    centroids_manager_->add(partitions->partition_ids.size(), partitions->centroids, partition_ids, assignment);
    
    // new
    // for (int i = 0; i < partitions->nlist(); ++i)
    // {
    //     centroids_manager_->errors_[partition_ids[i]] = partitions->errors[i];
    // }

    //////////////////////////////////////////
    /// 修改分区状态信息
    //////////////////////////////////////////

    return partition_ids;
}

void GuyueIndex::delete_partitions(const std::vector<int64_t>& partition_ids)
{
    if (centroids_manager_ == nullptr)
    {
        throw std::runtime_error("[GuyueIndex] delete_partitions: No centroids manager.");
    }

    //////////////////////////////////////////
    /// 删除中心管理器(centroids_manager_)中的对应中心
    //////////////////////////////////////////
    centroids_manager_->remove(partition_ids);

    // new
    // for (int i = 0; i < partition_ids.size(); ++i)
    // {
    //     centroids_manager_->errors_.erase(partition_ids[i]);
    // }

    //////////////////////////////////////////
    /// 删除分区管理器(partition_manager_)中的对应分区
    //////////////////////////////////////////
    partition_manager_->delete_partitions(partition_ids);
}

std::vector<int64_t> GuyueIndex::get_ids_to_split()
{
    std::vector<int64_t> partition_to_split;
    std::vector<int64_t> partition_ids = partition_manager_->get_partitions_ids();

    for (const auto& partition_id : partition_ids)
    {
        int64_t partition_size = partition_manager_->get_partition_size(partition_id);
        if (partition_size > reindexing_params_->max_partition_size)
        {
            partition_to_split.push_back(partition_id);
        }
    }

    return partition_to_split;
}

std::vector<int64_t> GuyueIndex::get_ids_to_shrink()
{
    std::vector<int64_t> partition_to_shrink;
    std::vector<int64_t> partition_ids = partition_manager_->get_partitions_ids();

    for (const auto& partition_id : partition_ids)
    {
        int64_t partition_size = partition_manager_->get_partition_size(partition_id);
        if (partition_size < reindexing_params_->min_partition_size)
        {
            partition_to_shrink.push_back(partition_id);
        }
    }

    return partition_to_shrink;
}

void GuyueIndex::local_reassign(std::vector<int64_t>& partition_ids)
{
    // 获取IDs一定半径范围内的分区
    auto centroids_search_result = std::make_shared<SearchResult>();
    auto centroids_search_params = std::make_shared<SearchParams>();

    int d = dim();
    std::vector<float> queries = centroids_manager_->get_with_copy(partition_ids);
    int64_t n_queries = partition_ids.size();
    centroids_search_params->k = std::min(reindexing_params_->reindexing_radius, (int) nlist());

    centroids_search_result = searcher_->search_centers(
        centroids_manager_,
        n_queries, 
        queries, 
        centroids_search_params
    );

    // 获取去重后的IDs列表
    std::vector<int64_t> reassign_ids;
    if (!centroids_search_result->indices.empty())
    {
        std::unordered_set<int64_t> unique_ids;
        for (int i = 0; i < partition_ids.size(); ++i)
        {
            for (int j = 0; j < centroids_search_result->indices[i].size(); ++j)
            {
                unique_ids.insert(centroids_search_result->indices[i][j]);
            }
        }
        reassign_ids.assign(unique_ids.begin(), unique_ids.end());
    }

    // 获取IDs包含的分区向量
    int64_t reassign_size = reassign_ids.size();
    std::vector<std::vector<float>> list_vectors(reassign_size);
    std::vector<std::vector<int64_t>> list_ids(reassign_size);

#pragma omp parallel for
    for (int i = 0; i < reassign_size; ++i)
    {
        int64_t list_no = reassign_ids[i];
        int64_t list_size = partition_manager_->partition_store_->list_size(list_no);
        if (list_size == 0)
        {
            list_vectors[i] = {};
            list_ids[i] = {};
            continue;
        }
        auto src_codes = partition_manager_->partition_store_->get_codes(list_no);
        auto src_ids = partition_manager_->partition_store_->get_ids(list_no);

        std::vector<float> part_vectors(list_size * d);
        
        if (!pq_params_)
        {
            const float* src_vectors = reinterpret_cast<const float*>(src_codes);
            std::memcpy(part_vectors.data(), src_vectors, list_size * d * sizeof(float));
        } else {
            guyue::decode_batch(src_codes, part_vectors.data(), list_size, d);
        }

        list_vectors[i] = std::move(part_vectors);
        list_ids[i].assign(src_ids, src_ids + list_size);
        partition_manager_->partition_store_->partitions_[list_no]->clear();
    }

    // 对向量进行重分配
    for (int i = 0; i < reassign_ids.size(); ++i)
    {
        if (list_ids[i].empty())
        {
            continue;
        }
        add(list_vectors[i], list_ids[i], true);
    }
}

void GuyueIndex::reassign(const std::vector<int64_t>& partition_ids)
{
    int d = dim();
    // 获取IDs包含的分区向量
    int64_t reassign_size = partition_ids.size();
    std::vector<std::vector<float>> list_vectors(reassign_size);
    std::vector<std::vector<int64_t>> list_ids(reassign_size);

#pragma omp parallel for
    for (int i = 0; i < reassign_size; ++i)
    {
        int64_t list_no = partition_ids[i];
        int64_t list_size = partition_manager_->partition_store_->list_size(list_no);
        if (list_size == 0)
        {
            list_vectors[i] = {};
            list_ids[i] = {};
            continue;
        }
        auto src_codes = partition_manager_->partition_store_->get_codes(list_no);
        auto src_ids = partition_manager_->partition_store_->get_ids(list_no);

        std::vector<float> part_vectors(list_size * d);
        
        if (!pq_params_)
        {
            const float* src_vectors = reinterpret_cast<const float*>(src_codes);
            std::memcpy(part_vectors.data(), src_vectors, list_size * d * sizeof(float));
        } else {
            guyue::decode_batch(src_codes, part_vectors.data(), list_size, d);
        }

        list_vectors[i] = std::move(part_vectors);
        list_ids[i].assign(src_ids, src_ids + list_size);
        partition_manager_->partition_store_->partitions_[list_no]->clear();
    }
    delete_partitions(partition_ids);
    
    // 修改树结构
    if (tree_build_)
    {
        partition_tree_->shrink(partition_ids);
    }

    // 对向量进行重分配
    for (int i = 0; i < partition_ids.size(); ++i)
    {
        if (list_ids[i].empty())
        {
            continue;
        }
        add(list_vectors[i], list_ids[i]);
    }
}

void GuyueIndex::ReindexingPolicy()
{
    if (reindexing_params_->reindexing_strategy == "None")
    {
        return;
    }
    else if (reindexing_params_->reindexing_strategy == "DeDrift") {
        DeDrift();
    } else if (reindexing_params_->reindexing_strategy == "LIRE") {
        LIRE();
    } else if (reindexing_params_->reindexing_strategy == "Tree-LIRE") {
        TreeLIRE();
    } else if (reindexing_params_->reindexing_strategy == "Error-LIRE") {
        ErrorLIRE();
    } else if (reindexing_params_->reindexing_strategy == "Ada-IVF") {
        AdaIVF();
    } else if (reindexing_params_->reindexing_strategy == "CM") {
        CM();
    } else {
        throw std::runtime_error("[GuyueIndex] ReindexingPolicy: The maintenance strategy does not exist.");
    }
}

void GuyueIndex::DeDrift()
{
    int64_t total_lists = nlist();
    std::vector<std::pair<int64_t, int64_t>> list_sizes;
    list_sizes.reserve(total_lists);
    
    //////////////////////////////////////////
    /// Step 1: 统计所有分区的大小 (list_id, size)
    //////////////////////////////////////////
    std::vector<int64_t> partitions_ids = partition_manager_->get_partitions_ids();
    for (const auto& partition_id : partitions_ids)
    {
        int64_t partition_size = partition_manager_->get_partition_size(partition_id);
        list_sizes.emplace_back(partition_id, partition_size);
    }

    //////////////////////////////////////////
    /// Step 2: 按分区 size 降序排序
    //////////////////////////////////////////
    std::sort(list_sizes.begin(), list_sizes.end(), [](const auto& a, const auto& b) {
        return a.second > b.second; 
    });
    
    int64_t mid = total_lists / 2;
    int64_t mid_size = list_sizes[mid].second;

    //////////////////////////////////////////
    /// Step 3: 选取前 top_k 个分区 id
    //////////////////////////////////////////
    std::vector<int64_t> reindexing_ids;
    int64_t topk_largest_size = 0;
    int64_t limit = std::min((int64_t) reindexing_params_->topk_largest_partitions, total_lists);
    for (int64_t i = 0; i < limit; ++i)
    {
        reindexing_ids.push_back(list_sizes[i].first);
        topk_largest_size += list_sizes[i].second;
    }

    //////////////////////////////////////////
    /// Step 4: 选取后 k2 个分区 IDs
    //////////////////////////////////////////
    int k2 = topk_largest_size / mid_size - reindexing_params_->topk_largest_partitions;
    for (int64_t i = 0; i < k2; ++i) 
    {
        reindexing_ids.push_back(list_sizes[total_lists - i - 1].first);
    }

    //////////////////////////////////////////
    /// Step 5: 对收集的分区进行重索引
    //////////////////////////////////////////
    std::shared_ptr<Clustering> reindexing_partitions;
    // reindexing_partitions = partition_manager_->reindexing_partitions(reindexing_ids, k2, reindexing_params_->refinement_iterations);
    reindexing_partitions = partition_manager_->reindexing_partitions(reindexing_ids, topk_largest_size / mid_size , reindexing_params_->refinement_iterations);
    delete_partitions(reindexing_ids);
    add_partitions(reindexing_partitions);
}

void GuyueIndex::LIRE()
{
    //////////////////////////////////////////
    /// Step1: 拆分
    //////////////////////////////////////////
    std::vector<int64_t> ids_to_split = get_ids_to_split();

    while (ids_to_split.size() > 0)
    {
        // 将每个超过阈值的分区拆分为2个小分区
        std::shared_ptr<Clustering> split_partitions;
        split_partitions = partition_manager_->split_partitions(ids_to_split, 2, reindexing_params_->refinement_iterations);
        delete_partitions(ids_to_split);
        std::vector<int64_t> partition_ids = add_partitions(split_partitions);

        // 为拆分分区以及附近范围内的向量触发重分配
        local_reassign(partition_ids);

        // 获取违规分区
        ids_to_split = get_ids_to_split();
    }
    
    //////////////////////////////////////////
    /// Step2: 缩减
    //////////////////////////////////////////
    if (ntotal() / nlist() < reindexing_params_->min_partition_size)
    {
        return;
    }
    
    std::vector<int64_t> ids_to_shrink = get_ids_to_shrink();
    if (ids_to_shrink.size() > 0)
    {
        reassign(ids_to_shrink);
    }

    if (reindexing_params_->target_nlist != -1)
    {
        // 需要缩减分区
        while (nlist() > reindexing_params_->target_nlist)
        {
            int64_t min_size = reindexing_params_->max_partition_size;
            int64_t partition_to_shrink;
            std::vector<int64_t> partition_ids = partition_manager_->get_partitions_ids();

            for (const auto& partition_id : partition_ids)
            {
                int64_t partition_size = partition_manager_->get_partition_size(partition_id);
                if (partition_size < min_size)
                {
                    partition_to_shrink = partition_id;
                    min_size = partition_size;
                }
            }
            reassign({partition_to_shrink});
        }

        // 需要分裂
        while (nlist() < reindexing_params_->target_nlist)
        {
            int64_t max_size = reindexing_params_->min_partition_size;
            int64_t partition_to_split;
            std::vector<int64_t> partition_ids = partition_manager_->get_partitions_ids();

            for (const auto& partition_id : partition_ids)
            {
                int64_t partition_size = partition_manager_->get_partition_size(partition_id);
                if (partition_size > max_size)
                {
                    partition_to_split = partition_id;
                    max_size = partition_size;
                }
            }
            std::shared_ptr<Clustering> split_partitions;
            split_partitions = partition_manager_->split_partitions({partition_to_split}, 2, reindexing_params_->refinement_iterations);
            delete_partitions({partition_to_split});
            std::vector<int64_t> new_ids = add_partitions(split_partitions);
            local_reassign(new_ids);
        }
    }
}

void GuyueIndex::TreeLIRE()
{
    if (!tree_build_ || !partition_tree_)
    {

    }
    
    //////////////////////////////////////////
    /// Step1: 拆分
    //////////////////////////////////////////
    std::vector<int64_t> ids_to_split = get_ids_to_split();

    while (ids_to_split.size() > 0)
    {
        // 将每个超过阈值的分区拆分为2个小分区
        std::shared_ptr<Clustering> split_partitions;
        split_partitions = partition_manager_->split_partitions(ids_to_split, 2, reindexing_params_->refinement_iterations);
        delete_partitions(ids_to_split);
        std::vector<int64_t> partition_ids = add_partitions(split_partitions);

        // 修改树结构
        std::vector<int64_t> assignment(split_partitions->partition_ids.size(), 0);
        int i = 0;
        for (int64_t ID : ids_to_split)
        {
            assignment[i] = ID;
            assignment[++i] = ID;
            ++i;
        }
        partition_tree_->split(ids_to_split, split_partitions->centroids, assignment);

        // 为拆分分区以及附近范围内的向量触发重分配
        local_reassign(partition_ids);

        // 获取违规分区
        ids_to_split = get_ids_to_split();
    }
    
    //////////////////////////////////////////
    /// Step2: 缩减
    //////////////////////////////////////////
    std::vector<int64_t> ids_to_shrink = get_ids_to_shrink();
    
    if (ids_to_shrink.size() > 0)
    {
        reassign(ids_to_shrink);
    }
}

void GuyueIndex::ErrorLIRE()
{
    // //////////////////////////////////////////
    // /// Step1: 缩减
    // //////////////////////////////////////////
    // std::vector<int64_t> ids_to_shrink = get_ids_to_shrink();
    // if (ids_to_shrink.size() > 0)
    // {
    //     reassign(ids_to_shrink);
    // }
    // int64_t n = ids_to_shrink.size();

    // //////////////////////////////////////////
    // /// Step2: 拆分
    // //////////////////////////////////////////
    // int64_t split_pid = -1;
    // float max_error = -1;
    // for (auto kv : centroids_manager_->errors_)
    // {
    //     if (kv.second > max_error)
    //     {
    //         split_pid = kv.first;
    //     }
    // }

    // while (n > 0)
    // {
    //     // 将每个超过阈值的分区拆分为2个小分区
    //     std::shared_ptr<Clustering> split_partitions;
    //     split_partitions = partition_manager_->split_partitions({split_pid}, 2, reindexing_params_->refinement_iterations);
    //     delete_partitions({split_pid});
    //     std::vector<int64_t> partition_ids = add_partitions(split_partitions);
        
    //     max_error = -1;
    //     for (auto kv : centroids_manager_->errors_)
    //     {
    //         if (kv.second > max_error)
    //         {
    //             split_pid = kv.first;
    //         }
    //     }
    //     n--;
    // }
}

void GuyueIndex::AdaIVF()
{
    // TODO
}

void GuyueIndex::CM()
{
    // TODO
}
