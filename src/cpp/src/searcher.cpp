/*
 * @Author: Guyue
 * @Date: 2026-03-23 14:38:34
 * @LastEditTime: 2026-04-03 15:27:04
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/src/searcher.cpp
 */
#include <searcher.h>
#include <omp.h>

Searcher::Searcher(faiss::MetricType metric) : metric_(metric) {}

Searcher::~Searcher() {}

std::shared_ptr<SearchResult> Searcher::search_centers(std::shared_ptr<PartitionManager> centroids_manager, int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params)
{
    // 分区搜索计时开始
    auto start = std::chrono::high_resolution_clock::now();

    if (centroids_manager->nlist() != 1)
    {
        throw std::runtime_error("[Searcher] search_center: The partition must be center!");
    }

    if (queries.empty() || queries.size() == 0)
    {
        auto empty_res = std::make_shared<SearchResult>();
        empty_res->indices = std::vector<std::vector<int64_t>>(0, std::vector<int64_t>(0, 0));
        empty_res->distances = std::vector<std::vector<float>>(0, std::vector<float>(0, 0.0));

        return empty_res;
    }

    using HeapForIP = faiss::CMin<float, int64_t>;
    using HeapForL2 = faiss::CMax<float, int64_t>;

    // 将中心管理器视为一个分区，nprobe视为1，k视为分区搜索时的nprobe
    int dim = queries.size() / n_queries;
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;
    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    std::vector<std::vector<float>> all_topk_dists(n_queries, std::vector<float>(k));
    std::vector<std::vector<int64_t>> all_topk_ids(n_queries, std::vector<int64_t>(k));

    auto init_result = [&](float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_heapify<HeapForIP>(k, simi, idxi);
        } else {
            faiss::heap_heapify<HeapForL2>(k, simi, idxi);
        }
    };

    auto add_local_results = [&](const float* local_dis, const int64_t* local_idx, float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
        } else {
            faiss::heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
        }
    };

    auto reorder_result = [&](float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_reorder<HeapForIP>(k, simi, idxi);
        } else {
            faiss::heap_reorder<HeapForL2>(k, simi, idxi);
        }
    };

    float* partition_vectors = (float*) centroids_manager->partition_store_->get_codes(0);
    int64_t* partition_ids = (int64_t*) centroids_manager->partition_store_->get_ids(0);
    int64_t partition_size = centroids_manager->partition_store_->list_size(0);

#pragma omp parallel for schedule(dynamic)
    for (int64_t q = 0; q < n_queries; ++q)
    {
        const float* query = queries.data() + q * dim;
        float* simi = all_topk_dists[q].data();
        int64_t* idxi = all_topk_ids[q].data();

        init_result(simi, idxi);

        const float* vec = partition_vectors;
        if (metric_ == faiss::METRIC_INNER_PRODUCT)
        {
            for (int l = 0; l < partition_size; ++l)
            {
                float ip = faiss::fvec_inner_product(query, vec, dim);
                if (ip > simi[0])
                {
                    faiss::minheap_replace_top(k, simi, idxi, ip, partition_ids[l]);
                }
                vec += dim;
            }
        } else {
            for (int l = 0; l < partition_size; ++l)
            {
                float dis = faiss::fvec_L2sqr(query, vec, dim);
                if (dis < simi[0])
                {
                    faiss::maxheap_replace_top(k, simi, idxi, dis, partition_ids[l]);
                }
                vec += dim;
            }
        }
        reorder_result(simi, idxi);
    }

    auto centroids_search_result = std::make_shared<SearchResult>();
    centroids_search_result->indices = all_topk_ids;
    centroids_search_result->distances = all_topk_dists;

    // 分区搜索计时开始
    auto end = std::chrono::high_resolution_clock::now();

    centroids_search_result->c_search_time = std::chrono::duration<double, std::milli>(end - start).count();

    return centroids_search_result;
}

std::shared_ptr<InsertSearchResult> Searcher::search_tree(std::shared_ptr<PartitionTree> partition_tree, int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params)
{
    if (queries.empty() || queries.size() == 0)
    {
        auto empty_res = std::make_shared<InsertSearchResult>();
        return empty_res;
    }

    int dim = queries.size() / n_queries;
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;
    int beam_size = (search_params && search_params->beam_size > 0) ? search_params->beam_size : 4;
    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    std::vector<std::vector<float>> all_topk_dists(n_queries, std::vector<float>(k, (is_descending ? 
                                                                                     -std::numeric_limits<float>::infinity() : 
                                                                                     std::numeric_limits<float>::infinity())));
    std::vector<std::vector<int64_t>> all_topk_ids(n_queries, std::vector<int64_t>(k));
    
    // 根据度量选择 beam 的比较器
    auto cmp = [&](const BeamNode& a, const BeamNode& b) 
    {
        if (is_descending) {
            // IP：内积大的优先（最大堆）
            return a.dist_ < b.dist_;
        } else {
            // L2：距离小的优先（最小堆）
            return a.dist_ > b.dist_;
        }
    };

#pragma omp parallel for schedule(dynamic)
    for (int64_t q = 0; q < n_queries; ++q)
    {
        const float* query = queries.data() + q * dim;

        std::priority_queue<BeamNode, std::vector<BeamNode>, decltype(cmp)> beam(cmp);

        beam.push({0.0f, partition_tree->root()});
        bool all_leaves = false;

        while (!beam.empty() && !all_leaves)
        {
            // 取出当前层所有候选节点
            std::vector<BeamNode> current_level;
            while (!beam.empty())
            {
                current_level.push_back(beam.top());
                beam.pop();
            }

            // 检查当前层是否全是叶子节点
            all_leaves = true;
            for (const BeamNode& bn : current_level)
            {
                if (!bn.node_->children_.empty())
                {
                    all_leaves = false;
                    break;
                }
            }

            // 如果全为叶子节点则重新放入beam并跳出循环
            if (all_leaves)
            {
                for (const BeamNode& bn : current_level)
                {
                    beam.push(bn);
                }
                break;
            }

            // 扩展候选节点
            std::vector<BeamNode> next_level;
            for (const BeamNode& bn : current_level)
            {
                if (bn.node_->children_.empty()) // 保留叶子节点
                {
                    next_level.push_back(bn);
                } else { // 扩展非叶子节点
                    for (Node* child : bn.node_->children_)
                    {
                        float child_dist;
                        if (is_descending) {
                            child_dist = faiss::fvec_inner_product(query, child->centroid_.data(), dim);
                        } else {
                            child_dist = faiss::fvec_L2sqr(query, child->centroid_.data(), dim);
                        }
                        next_level.push_back({child_dist, child});
                    }
                }
            }

            // 从所有候选节点中选取 beam_size 个最优节点进入下一层
            size_t num_to_keep = std::min((size_t) beam_size, next_level.size());
            if (next_level.empty()) break;
            if (is_descending) {
                // IP：按内积降序，取前 beam_size 个
                std::partial_sort(next_level.begin(),
                                  next_level.begin() + num_to_keep,
                                  next_level.end(),
                                  [](const BeamNode& a, const BeamNode& b) { return a.dist_ > b.dist_; });
            } else {
                // L2：按距离升序，取前 beam_size 个
                std::partial_sort(next_level.begin(),
                                  next_level.begin() + num_to_keep,
                                  next_level.end(),
                                  [](const BeamNode& a, const BeamNode& b) { return a.dist_ < b.dist_; });
            }

            // 重新填充beam
            for (size_t j = 0; j < num_to_keep; ++j)
            {
                beam.push(next_level[j]);
            }
        }
        
        for (int j = 0; j < k; ++j)
        {
            all_topk_ids[q][j] = beam.top().node_->ID_;
            all_topk_dists[q][j] = beam.top().dist_;
            beam.pop();
        }
    }

    auto search_result = std::make_shared<InsertSearchResult>();
    
    for (int64_t q = 0; q < n_queries; ++q)
    {
        for (int j = 0; j < k; ++j)
        {
            search_result->assignment[all_topk_ids[q][j]].push_back(q);
        }
    }

    return search_result;
}

std::shared_ptr<InsertSearchResult> Searcher::search_greedy(std::shared_ptr<PartitionTree> partition_tree, int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params)
{
    if (queries.empty() || queries.size() == 0)
    {
        auto empty_res = std::make_shared<InsertSearchResult>();
        return empty_res;
    }

    int dim = queries.size() / n_queries;
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;
    int beam_size = (search_params && search_params->beam_size > 0) ? search_params->beam_size : 4;
    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    std::vector<std::vector<int64_t>> all_results(n_queries);

#pragma omp parallel for schedule(dynamic)
    for (int q = 0; q < n_queries; ++q)
    {
        const float* query = queries.data() + q * dim;
     
        // 最小堆：用于贪心选择下一要遍历的节点
        using Candidate = std::pair<float, Node*>;
        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>> frontier;

        // 最大堆：用于维护距离最近的 k 个节点 ID
        std::priority_queue<std::pair<float, int64_t>> top_k_results;

        // 初始化根节点
        frontier.push({0.0f, partition_tree->root()});
        
        int visits = 0;
        while (!frontier.empty() && visits < 1000)
        {
            auto [current_dist, current_node] = frontier.top();
            frontier.pop();
            visits++;

            if (current_node->children_.empty())
            {
                top_k_results.push({current_dist, current_node->ID_});
                if (top_k_results.size() > k)
                {
                    top_k_results.pop();
                }
            } else {
                for (Node* child : current_node->children_)
                {
                    float dist = faiss::fvec_L2sqr(query, child->centroid_.data(), dim);
                    frontier.push({dist, child});
                }
            }
        }

        // 提取结果并反转，使得距离最近的排在前面
        std::vector<int64_t> result;
        result.reserve(top_k_results.size());
        while (!top_k_results.empty()) {
            result.push_back(top_k_results.top().second);
            top_k_results.pop();
        }
        std::reverse(result.begin(), result.end());
        
        all_results[q] = result;
    }
    
    auto search_result = std::make_shared<InsertSearchResult>();
    for (int64_t q = 0; q < n_queries; ++q)
    {
        for (int j = 0; j < k; ++j)
        {
            search_result->assignment[all_results[q][j]].push_back(q);
        }
    }

    return search_result;
}

std::shared_ptr<SearchResult> Searcher::search_partitions(std::shared_ptr<PartitionManager> partition_manager, int64_t n_queries, std::vector<float>& queries, std::vector<std::vector<int64_t>>& scan_lists, std::shared_ptr<SearchParams> search_params, std::shared_ptr<PQParams> pq_params)
{
    // 分区搜索计时开始
    auto start = std::chrono::high_resolution_clock::now();

    if (queries.empty() || queries.size() == 0)
    {
        auto empty_res = std::make_shared<SearchResult>();
        empty_res->indices = std::vector<std::vector<int64_t>>(0, std::vector<int64_t>(0, 0));
        empty_res->distances = std::vector<std::vector<float>>(0, std::vector<float>(0, 0));
        return empty_res;
    }

    using HeapForIP = faiss::CMin<float, int64_t>;
    using HeapForL2 = faiss::CMax<float, int64_t>;

    int dim = queries.size() / n_queries;
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;
    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    std::vector<std::vector<float>> all_topk_dists(n_queries, std::vector<float>(k));
    std::vector<std::vector<int64_t>> all_topk_ids(n_queries, std::vector<int64_t>(k));

    auto init_result = [&](float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_heapify<HeapForIP>(k, simi, idxi);
        } else {
            faiss::heap_heapify<HeapForL2>(k, simi, idxi);
        }
    };

    auto add_local_results = [&](const float* local_dis, const int64_t* local_idx, float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
        } else {
            faiss::heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
        }
    };

    auto reorder_result = [&](float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_reorder<HeapForIP>(k, simi, idxi);
        } else {
            faiss::heap_reorder<HeapForL2>(k, simi, idxi);
        }
    };

#pragma omp parallel for schedule(dynamic)
    for (int64_t q = 0; q < n_queries; ++q)
    {
        const float* query = queries.data() + q * dim;
        int num_parts = scan_lists[q].size();
        float* simi = all_topk_dists[q].data();
        int64_t* idxi = all_topk_ids[q].data();

        init_result(simi, idxi);
        
        for (int p = 0; p < num_parts; ++p)
        {
            int64_t p_id = scan_lists[q][p];
            uint8_t* partition_vectors = (uint8_t*) partition_manager->partition_store_->get_codes(p_id);
            int64_t* partition_ids = (int64_t*) partition_manager->partition_store_->get_ids(p_id);
            int64_t partition_size = partition_manager->partition_store_->list_size(p_id);
            
            scan_one_list(query, partition_vectors, partition_ids, partition_size, dim, simi, idxi, k, metric_, pq_params);

            // if (q == 0 && p == 0)
            // {
            //     for (int i = 0; i<10;++i)
            //     {
            //         std::cout << simi[i] << " ";
            //     }
            //     std::cout << std::endl;
            // }
        }
        reorder_result(simi, idxi);
    }

    auto search_result = std::make_shared<SearchResult>();
    search_result->indices = all_topk_ids;
    search_result->distances = all_topk_dists;

    // 分区搜索计时结束
    auto end = std::chrono::high_resolution_clock::now();

    for (int64_t i = 0; i < n_queries; ++i)
    {
        search_result->search_nprobe += scan_lists[i].size();
        for (int64_t j = 0; j < scan_lists[i].size(); ++j)
        {
            search_result->search_points += partition_manager->get_partition_size(scan_lists[i][j]);
        }
    }
    search_result->search_nprobe /= n_queries;
    search_result->search_points /= n_queries;
    search_result->p_search_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    return search_result;
}

std::shared_ptr<SearchResult> Searcher::search_partitions_acc(std::shared_ptr<PartitionManager> partition_manager, int64_t n_queries, std::vector<float>& queries, std::vector<std::vector<int64_t>>& scan_lists, std::shared_ptr<SearchParams> search_params, std::shared_ptr<PQParams> pq_params)
{
    if (queries.empty() || queries.size() == 0)
    {
        auto empty_res = std::make_shared<SearchResult>();
        empty_res->indices = std::vector<std::vector<int64_t>>(0, std::vector<int64_t>(0, 0));
        empty_res->distances = std::vector<std::vector<float>>(0, std::vector<float>(0, 0));
        return empty_res;
    }

    using HeapForIP = faiss::CMin<float, int64_t>;
    using HeapForL2 = faiss::CMax<float, int64_t>;

    int dim = queries.size() / n_queries;
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;
    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    std::vector<std::vector<float>> all_topk_dists(n_queries, std::vector<float>(k));
    std::vector<std::vector<int64_t>> all_topk_ids(n_queries, std::vector<int64_t>(k));
    
    auto init_result = [&](float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_heapify<HeapForIP>(k, simi, idxi);
        } else {
            faiss::heap_heapify<HeapForL2>(k, simi, idxi);
        }
    };

    auto add_local_results = [&](const float* local_dis, const int64_t* local_idx, float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
        } else {
            faiss::heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
        }
    };

    auto reorder_result = [&](float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_reorder<HeapForIP>(k, simi, idxi);
        } else {
            faiss::heap_reorder<HeapForL2>(k, simi, idxi);
        }
    };

    for (int64_t q = 0; q < n_queries; ++q)
    {
        const float* query = queries.data() + q * dim;
        int num_parts = scan_lists[q].size();
        float* simi = all_topk_dists[q].data();
        int64_t* idxi = all_topk_ids[q].data();
        init_result(simi, idxi);

        for (int p = 0; p < num_parts; ++p)
        {
            int64_t p_id = scan_lists[q][p];
            const float* partition_vectors = (const float*) partition_manager->partition_store_->get_codes(p_id);
            int64_t* partition_ids = (int64_t*) partition_manager->partition_store_->get_ids(p_id);
            int64_t partition_size = partition_manager->partition_store_->list_size(p_id);
            std::vector<float> partition_result(partition_size);

            accumulating_one2manyl2_avx2(query, dim, partition_vectors, partition_result);

            for (int i = 0; i < partition_size; ++i)
            {
                if (partition_result[0] < simi[0])
                {
                    faiss::maxheap_replace_top(k, simi, idxi, partition_result[i], partition_ids[i]);
                }
            }
            // add_local_results(partition_result.data(), partition_ids, simi, idxi);
        }
        reorder_result(simi, idxi);
    }

    auto search_result = std::make_shared<SearchResult>();
    search_result->indices = all_topk_ids;
    search_result->distances = all_topk_dists;

    return search_result;
}

std::shared_ptr<SearchResult> Searcher::search_partitions_batch(std::shared_ptr<PartitionManager> partition_manager, int64_t n_queries, std::vector<float>& queries, std::vector<std::vector<int64_t>>& scan_lists, std::shared_ptr<SearchParams> search_params, std::shared_ptr<PQParams> pq_params)
{
    // 分区搜索计时开始
    auto start = std::chrono::high_resolution_clock::now();

    if (queries.empty() || queries.size() == 0)
    {
        auto empty_res = std::make_shared<SearchResult>();
        empty_res->indices = std::vector<std::vector<int64_t>>(0, std::vector<int64_t>(0, 0));
        empty_res->distances = std::vector<std::vector<float>>(0, std::vector<float>(0, 0));
        return empty_res;
    }
    using HeapForIP = faiss::CMin<float, int64_t>;
    using HeapForL2 = faiss::CMax<float, int64_t>;

    int dim = queries.size() / n_queries;
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;
    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    std::vector<std::vector<float>> all_topk_dists(n_queries, std::vector<float>(k));
    std::vector<std::vector<int64_t>> all_topk_ids(n_queries, std::vector<int64_t>(k));
    int num_parts = scan_lists[0].size();

    auto init_result = [&](float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_heapify<HeapForIP>(k, simi, idxi);
        } else {
            faiss::heap_heapify<HeapForL2>(k, simi, idxi);
        }
    };

    auto add_local_results = [&](const float* local_dis, const int64_t* local_idx, float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
        } else {
            faiss::heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
        }
    };

    auto reorder_result = [&](float* simi, int64_t* idxi)
    {
        if (is_descending)
        {
            faiss::heap_reorder<HeapForIP>(k, simi, idxi);
        } else {
            faiss::heap_reorder<HeapForL2>(k, simi, idxi);
        }
    };

    // 将查询重新组织为: 分区-查询 的形式
    std::unordered_map<int64_t, std::vector<int64_t>> queries_by_partition;
    for (int64_t q = 0; q < n_queries; ++q)
    {
        for (int p = 0; p < num_parts; ++p)
        {
            int64_t pid = scan_lists[q][p];
            queries_by_partition[pid].push_back(q);
        }
        init_result(all_topk_dists[q].data(), all_topk_ids[q].data());
    }

    // 汇总每个分区的查询向量
    std::vector<std::pair<int64_t, std::vector<int64_t>>> queries_vec;
    queries_vec.reserve(queries_by_partition.size());
    for (const auto& entry : queries_by_partition)
    {
        queries_vec.push_back(entry);
    }

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < queries_by_partition.size(); ++i)
    {
        int64_t pid = queries_vec[i].first;
        auto query_indices = queries_vec[i].second;
    
        // 获取分区数据
        uint8_t* partition_src = (uint8_t*) partition_manager->partition_store_->get_codes(pid);
        const int64_t* partition_ids = (int64_t*) partition_manager->partition_store_->get_ids(pid);
        int64_t partition_size = partition_manager->partition_store_->list_size(pid);

        if (pq_params)
        {
            std::vector<float> partition_codes(partition_size * dim);
            guyue::decode_batch(partition_src, partition_codes.data(), partition_size, dim);
            for (int q_idx = 0; q_idx < query_indices.size(); ++q_idx)
            {
                int q = query_indices[q_idx];
                float* simi = all_topk_dists[q].data();
                int64_t* idxi = all_topk_ids[q].data();

                for (int64_t j = 0; j < partition_size; ++j)
                {
                    const float* curr_vec = partition_codes.data() + j * dim;
                    if (metric_ == faiss::METRIC_INNER_PRODUCT)  // IP
                    {
                        float val = faiss::fvec_inner_product(&queries[q*dim], curr_vec, dim);
                        if (val > simi[0])
                        {
                            faiss::minheap_replace_top(k, simi, idxi, val, partition_ids[j]);
                        }
                    } else {    // L2
                        float val = faiss::fvec_L2sqr(&queries[q*dim], curr_vec, dim);
                        if (val < simi[0])
                        {
                            faiss::maxheap_replace_top(k, simi, idxi, val, partition_ids[j]);
                        }
                    }
                }
            }
        } else {
            const float* partition_codes = reinterpret_cast<const float*>(partition_src);
            for (int q_idx = 0; q_idx < query_indices.size(); ++q_idx)
            {
                int q = query_indices[q_idx];
                float* simi = all_topk_dists[q].data();
                int64_t* idxi = all_topk_ids[q].data();

                for (int64_t j = 0; j < partition_size; ++j)
                {
                    const float* curr_vec = partition_codes + j * dim;
                    if (metric_ == faiss::METRIC_INNER_PRODUCT)  // IP
                    {
                        float val = faiss::fvec_inner_product(&queries[q*dim], curr_vec, dim);
                        if (val > simi[0])
                        {
                            faiss::minheap_replace_top(k, simi, idxi, val, partition_ids[j]);
                        }
                    } else {    // L2
                        float val = faiss::fvec_L2sqr(&queries[q*dim], curr_vec, dim);
                        if (val < simi[0])
                        {
                            faiss::maxheap_replace_top(k, simi, idxi, val, partition_ids[j]);
                        }
                    }
                }
            }
        }
    }

    auto search_result = std::make_shared<SearchResult>();
    search_result->indices = all_topk_ids;
    search_result->distances = all_topk_dists;

    // 分区搜索计时结束
    auto end = std::chrono::high_resolution_clock::now();

    for (int64_t i = 0; i < n_queries; ++i)
    {
        search_result->search_nprobe += scan_lists[i].size();
        for (int64_t j = 0; j < scan_lists[i].size(); ++j)
        {
            search_result->search_points += partition_manager->get_partition_size(scan_lists[i][j]);
        }
    }
    search_result->search_nprobe /= n_queries;
    search_result->search_points /= n_queries;
    search_result->p_search_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    return search_result;
}

std::shared_ptr<InsertSearchResult> Searcher::search_insert(std::shared_ptr<PartitionManager> centroids_manager, int64_t n_queries, std::vector<float>& queries, std::shared_ptr<SearchParams> search_params)
{
    if (queries.empty() || queries.size() == 0)
    {
        auto empty_res = std::make_shared<InsertSearchResult>();
        return empty_res;
    }

    int dim = queries.size() / n_queries;
    int k = (search_params && search_params->k > 0) ? search_params->k : 1;
    bool is_descending = (metric_ == faiss::METRIC_INNER_PRODUCT);
    auto search_result = std::make_shared<InsertSearchResult>();

    int64_t p_id = 0;
    float* partition_vectors = (float*) centroids_manager->partition_store_->get_codes(p_id);
    int64_t* partition_ids = (int64_t*) centroids_manager->partition_store_->get_ids(p_id);
    int64_t partition_size = centroids_manager->partition_store_->list_size(p_id);

    std::vector<int64_t> labels(n_queries * k);
    std::vector<float> distances(n_queries * k);

    if (is_descending)
    {
        faiss::knn_inner_product(queries.data(), partition_vectors, dim, n_queries, partition_size, k, distances.data(), labels.data(), nullptr);
    } else {
        faiss::float_maxheap_array_t res = {size_t(n_queries), size_t(k), labels.data(), distances.data()};
        faiss::knn_L2sqr(queries.data(), partition_vectors, dim, n_queries, partition_size, &res, nullptr, nullptr);
    }

    for (int64_t q = 0; q < n_queries; ++q)
    {
        for (int j = 0; j < k; ++j)
        {
            search_result->assignment[partition_ids[labels[q * k + j]]].push_back(q);
        }
        // new
        // search_result->dists.push_back(distances[q * k]);
    }
    
    return search_result;
}

void Searcher::scan_one_list(const float* query_vec, const uint8_t* uint8_vecs, const int64_t* list_ids, int list_size, int d, float* simi, int64_t* idxi, size_t k, faiss::MetricType metric, std::shared_ptr<PQParams> pq_params)
{
    if (!pq_params)
    {
        const float* list_vecs = reinterpret_cast<const float*>(uint8_vecs);
        if (metric == faiss::METRIC_INNER_PRODUCT)  // IP
        {
            for (int l = 0; l < list_size; ++l)
            {
                int64_t current_id = list_ids ? list_ids[l] : l;
                // _mm_prefetch((const char*)(list_vecs + 4 * d), _MM_HINT_T0);
                float val = faiss::fvec_inner_product(query_vec, list_vecs, d);
                if (val > simi[0])
                {
                    faiss::minheap_replace_top(k, simi, idxi, val, current_id);
                }
                list_vecs += d;
            }
        } else {    // L2
            for (int l = 0; l < list_size; ++l)
            {
                int64_t current_id = list_ids ? list_ids[l] : l;
                // _mm_prefetch((const char*)(list_vecs + 4 * d), _MM_HINT_T0);
                float val = faiss::fvec_L2sqr(query_vec, list_vecs, d);
                if (val < simi[0])
                {
                    faiss::maxheap_replace_top(k, simi, idxi, val, current_id);
                }
                list_vecs += d;
            }
        }
    } else {
        size_t code_size = d * pq_params->bytes_per_dim + pq_params->extra_bytes;
        const uint8_t* list_vecs = uint8_vecs;
        if (metric == faiss::METRIC_INNER_PRODUCT)  // IP
        {
            for (int l = 0; l < list_size; ++l)
            {
                int64_t current_id = list_ids ? list_ids[l] : l;
                std::vector<float> fvec(d);
                guyue::decode(list_vecs, fvec.data(), d);
                float val = faiss::fvec_inner_product(query_vec, fvec.data(), d);
                if (val > simi[0])
                {
                    faiss::minheap_replace_top(k, simi, idxi, val, current_id);
                }
                list_vecs += code_size;
            }
        } else {    // L2
            for (int l = 0; l < list_size; ++l)
            {
                int64_t current_id = list_ids ? list_ids[l] : l;
                std::vector<float> fvec(d);
                guyue::decode(list_vecs, fvec.data(), d);
                float val = faiss::fvec_L2sqr(query_vec, fvec.data(), d);
                if (val < simi[0])
                {
                    faiss::maxheap_replace_top(k, simi, idxi, val, current_id);
                }
                list_vecs += code_size;
            }
        }
    }
}

void Searcher::accumulating_one2manyl2_avx2(const float* query, int dim, const float* dataset, std::vector<float>& result)
{
    const size_t num_outer_iters = result.size() / 3;
    const size_t parallel_end = num_outer_iters * 3;

    constexpr size_t kMinPrefetchAheadDims = 512;
    size_t num_prefetch_datapoints = std::max<size_t>(1, kMinPrefetchAheadDims / dim);

    auto get_db_ptr = [&dataset, result, dim](size_t i) -> const float* {
        // const float* result_ptr = result.data() + i;
        // __builtin_prefetch(result_ptr, 1, 3);
        return dataset + i * dim;
    };

    auto sum4 = [](__m128 x) -> float {
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(x), 8)));
        return x[0] + x[1];
    };

    for (size_t i = 0; i < num_outer_iters; i++)
    {
        const float* f0 = get_db_ptr(i);
        const float* f1 = get_db_ptr(i + num_outer_iters);
        const float* f2 = get_db_ptr(i + 2 * num_outer_iters);
        const float* p0 = nullptr, *p1 = nullptr, *p2 = nullptr;

        if (i + num_prefetch_datapoints < num_outer_iters)
        {
            p0 = get_db_ptr(i + num_prefetch_datapoints);
            p1 = get_db_ptr(i + num_outer_iters + num_prefetch_datapoints);
            p2 = get_db_ptr(i + 2 * num_outer_iters + num_prefetch_datapoints);
        }

        __m256 a0_256 = _mm256_setzero_ps();
        __m256 a1_256 = _mm256_setzero_ps();
        __m256 a2_256 = _mm256_setzero_ps();
        size_t j = 0;
        
        for (; j + 8 <= dim; j += 8)
        {
            __m256 q = _mm256_loadu_ps(query + j);
            __m256 v0 = _mm256_loadu_ps(f0 + j);
            __m256 v1 = _mm256_loadu_ps(f1 + j);
            __m256 v2 = _mm256_loadu_ps(f2 + j);

            if (p0)
            {
                _mm_prefetch(reinterpret_cast<const char*>(p0 + j), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(p1 + j), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(p2 + j), _MM_HINT_T0);
            }

            __m256 tmp = _mm256_sub_ps(q, v0);
            a0_256 = _mm256_fmadd_ps(tmp, tmp, a0_256);
            __m256 tmp1 = _mm256_sub_ps(q, v1);
            a1_256 = _mm256_fmadd_ps(tmp1, tmp1, a1_256);
            __m256 tmp2 = _mm256_sub_ps(q, v2);
            a2_256 = _mm256_fmadd_ps(tmp2, tmp2, a2_256);
        }

        const __m128 upper = _mm256_extractf128_ps(a0_256, 1);
        const __m128 lower = _mm256_castps256_ps128(a0_256);
        __m128 a0 = _mm_add_ps(upper, lower);
        const __m128 upper1 = _mm256_extractf128_ps(a1_256, 1);
        const __m128 lower1 = _mm256_castps256_ps128(a1_256);
        __m128 a1 = _mm_add_ps(upper1, lower1);
        const __m128 upper2 = _mm256_extractf128_ps(a2_256, 1);
        const __m128 lower2 = _mm256_castps256_ps128(a2_256);
        __m128 a2 = _mm_add_ps(upper2, lower2);

        if (j + 4 <= dim)
        {
            __m128 q = _mm_loadu_ps(query + j);
            __m128 v0 = _mm_loadu_ps(f0 + j);
            __m128 v1 = _mm_loadu_ps(f1 + j);
            __m128 v2 = _mm_loadu_ps(f2 + j);

            if (p0)
            {
                _mm_prefetch(reinterpret_cast<const char*>(p0 + j), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(p1 + j), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(p2 + j), _MM_HINT_T0);
            }

            __m128 tmp = _mm_sub_ps(q, v0);
            a0 = _mm_fmadd_ps(tmp, tmp, a0);
            __m128 tmp1 = _mm_sub_ps(q, v1);
            a1 = _mm_fmadd_ps(tmp1, tmp1, a1);
            __m128 tmp2 = _mm_sub_ps(q, v2);
            a2 = _mm_fmadd_ps(tmp2, tmp2, a2);
            j += 4;
        }
        
        if (j + 2 <= dim)
        {
            __m128 q = _mm_setzero_ps();
            __m128 v0 = _mm_setzero_ps();
            __m128 v1 = _mm_setzero_ps();
            __m128 v2 = _mm_setzero_ps();

            q = _mm_loadh_pi(q, reinterpret_cast<const __m64*>(query + j));
            v0 = _mm_loadh_pi(v0, reinterpret_cast<const __m64*>(f0 + j));
            v1 = _mm_loadh_pi(v1, reinterpret_cast<const __m64*>(f1 + j));
            v2 = _mm_loadh_pi(v2, reinterpret_cast<const __m64*>(f2 + j));

            __m128 tmp = _mm_sub_ps(q, v0);
            a0 = _mm_fmadd_ps(tmp, tmp, a0);
            __m128 tmp1 = _mm_sub_ps(q, v1);
            a1 = _mm_fmadd_ps(tmp1, tmp1, a1);
            __m128 tmp2 = _mm_sub_ps(q, v2);
            a2 = _mm_fmadd_ps(tmp2, tmp2, a2);
            j += 2;
        }

        float result0 = sum4(a0);
        float result1 = sum4(a1);
        float result2 = sum4(a2);

        if (j < dim)
        {
            float tmp = query[j] - f0[j];
            result0 = result0 + tmp * tmp;
            float tmp1 = query[j] - f1[j];
            result1 = result1 + tmp1 * tmp1;
            float tmp2 = query[j] - f2[j];
            result2 = result2 + tmp2 * tmp2;
        }

        result[i] = result0;
        result[i + num_outer_iters] = result1;
        result[i + 2 * num_outer_iters] = result2;
    }

    size_t i = parallel_end;
    for (; i < result.size(); ++i)
    {
        const float* f0 = dataset + i * dim;
        float result0 = 0.0;
        for (size_t j = 0; j < dim; ++j)
        {
            const float tmp = query[j] - f0[j];
            result0 += tmp * tmp;
        }
        result[i] = result0;
    }
}
