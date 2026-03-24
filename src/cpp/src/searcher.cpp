/*
 * @Author: Guyue
 * @Date: 2026-03-23 14:38:34
 * @LastEditTime: 2026-03-24 16:59:49
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

std::shared_ptr<SearchResult> Searcher::search_partitions(std::shared_ptr<PartitionManager> partition_manager, int64_t n_queries, std::vector<float>& queries, std::vector<std::vector<int64_t>>& scan_lists, std::shared_ptr<SearchParams> search_params)
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
            float* partition_vectors = (float*) partition_manager->partition_store_->get_codes(p_id);
            int64_t* partition_ids = (int64_t*) partition_manager->partition_store_->get_ids(p_id);
            int64_t partition_size = partition_manager->partition_store_->list_size(p_id);

            scan_one_list(query, partition_vectors, partition_ids, partition_size, dim, simi, idxi, k, metric_);
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
    }
    
    return search_result;
}

void Searcher::scan_one_list(const float* query_vec, const float* list_vecs, const int64_t* list_ids, int list_size, int d, float* simi, int64_t* idxi, size_t k, faiss::MetricType metric)
{
    if (metric == faiss::METRIC_INNER_PRODUCT)
    {
        if (list_ids == nullptr)
        {
            const float* vec = list_vecs;
            for (int l = 0; l < list_size; ++l)
            {
                float ip = faiss::fvec_inner_product(query_vec, vec, d);
                if (ip > simi[0])
                {
                    faiss::minheap_replace_top(k, simi, idxi, ip, l);
                }
                vec += d;
            }
        } else {
            const float *vec = list_vecs;
            for (int l = 0; l < list_size; ++l)
            {
                float ip = faiss::fvec_inner_product(query_vec, vec, d);
                if (ip > simi[0])
                {
                    faiss::minheap_replace_top(k, simi, idxi, ip, list_ids[l]);
                }
                vec += d;
            }
        }
    } else {
        if (list_ids == nullptr)
        {
            const float *vec = list_vecs;
            for (int l = 0; l < list_size; ++l)
            {
                float dis = faiss::fvec_L2sqr(query_vec, vec, d);
                if (dis < simi[0])
                {
                    faiss::maxheap_replace_top(k, simi, idxi, dis, l);
                }
                vec += d;
            }
        } else {
            const float *vec = list_vecs;
            for (int l = 0; l < list_size; ++l)
            {
                // if (l + 2 < list_size) {
                //     _mm_prefetch((const char*)(list_vecs + (l + 2) * d), _MM_HINT_T0);
                // }
                float dis = faiss::fvec_L2sqr(query_vec, vec, d);
                if (dis < simi[0])
                {
                    faiss::maxheap_replace_top(k, simi, idxi, dis, list_ids[l]);
                }
                vec += d;
            }
        }
    }
}
