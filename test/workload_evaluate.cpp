/*
 * @Author: Guyue
 * @Date: 2026-03-24 10:56:02
 * @LastEditTime: 2026-04-01 17:03:12
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/test/workload_evaluate.cpp
 */
#include <guyue_index.h>
#include <utils.h>

int main()
{
    std::string dataset = "sift-2M";
    size_t pos = dataset.find('-');
    std::string dataset_name = dataset.substr(0, pos);
    size_t batch_size_ = 1000;
    int k = 10;
    float target_recall = 0.98;
    int nprobe = 20;

    // std::string vectors_file_path = "/mnt/hgfs/DataSet/" + dataset + "/workload2/" + dataset_name + "_base.fvecs";
    std::string vectors_file_path = "/mnt/hgfs/DataSet/" + dataset + "/" + dataset_name + "_base.fvecs";
    std::string queries_file_path = "/mnt/hgfs/DataSet/" + dataset + "/" + dataset_name + "_query.fvecs";
    std::string gt_dir_path = "/mnt/hgfs/DataSet/" + dataset + "/workload2/";
    std::string runbook_file_path = "/mnt/hgfs/DataSet/" + dataset + "/workload2/runbook.json";
    const std::string output_csv_path = "../output/workload2/" + dataset + "_LIRE-Update_b" + std::to_string(batch_size_) + "_" + std::to_string(k) + ".csv";

    //////////////////////////////////////////
    /// 向量读取
    //////////////////////////////////////////
    int dim = 0;
    int64_t n_vectors = 0;
    std::vector<float> vectors = read_fvecs(vectors_file_path, dim, n_vectors);
    std::cout << "vectors dim: " << dim << std::endl;
    std::cout << "vectors size: " << n_vectors << std::endl;
    std::cout << "========================================================================"<< std::endl;

    //////////////////////////////////////////
    /// runbook读取
    //////////////////////////////////////////
    nlohmann::json runbook = read_json(runbook_file_path);
    auto& datainfo = runbook["datainfo"];
    std::cout << "data: " << datainfo["data"] << std::endl;
    std::cout << "total_vectors: " << datainfo["total_vectors"] << std::endl;
    std::cout << "dimension: " << datainfo["dimension"] << std::endl;
    std::cout << "========================================================================"<< std::endl;
    
    // auto& parameters = runbook["parameters"];
    // std::cout << "s_0: " << parameters["s_0"] << std::endl;
    // std::cout << "r_id: " << parameters["r_id"] << std::endl;
    // std::cout << "CSF_u: " << parameters["CSF_u"] << std::endl;
    // std::cout << "r_rw: " << parameters["r_rw"] << std::endl;
    // std::cout << "CSF_q: " << parameters["CSF_q"] << std::endl;
    // std::cout << "num_clusters: " << parameters["num_clusters"] << std::endl;
    // std::cout << "num_insert_ops: " << parameters["num_insert_ops"] << std::endl;
    // std::cout << "num_delete_ops: " << parameters["num_delete_ops"] << std::endl;
    // std::cout << "num_search_ops: " << parameters["num_search_ops"] << std::endl;
    // std::cout << "========================================================================"<< std::endl;

    //////////////////////////////////////////
    /// 索引构建配置
    //////////////////////////////////////////
    auto& operations = runbook["operations"];
    auto index = std::make_shared<GuyueIndex>();
    auto build_params = std::make_shared<IndexBuildParams>();
    build_params->dimension = dim;
    build_params->nlist = 1;
    build_params->niter = 15;
    build_params->metric = "l2";

    auto reindexing_params = std::make_shared<ReindexingParams>();
    // reindexing_params->reindexing_strategy = "DeDrift";
    reindexing_params->reindexing_strategy = "LIRE";
    // reindexing_params->reindexing_strategy = "Tree-LIRE";
    // build_params->tree_build = true;
    // reindexing_params->reindexing_strategy = "Ada-IVF";
    // reindexing_params->reindexing_strategy = "CM";
    // reindexing_params->reindexing_strategy = "None";
    reindexing_params->reindexing_radius = 15;
    reindexing_params->refinement_iterations = 10;
    reindexing_params->max_partition_size = 450;    // 470
    reindexing_params->min_partition_size = 200;    // 100
    // reindexing_params->topk_largest_partitions = 64;
    reindexing_params->centroids_update = true;

    int search_step = 0;
    bool is_build = false;
    for (size_t op_i = 0; op_i < operations.size(); ++op_i)
    {
        auto& operation = operations[op_i];
        std::string op_type = operation["operation"];
        if (op_type == "build")
        {
            int start = operation["start"];
            int end = operation["end"];
            std::cout << std::endl;
            std::cout << op_type << " start: " << start << " end: " << end << std::endl;

            int64_t build_size = end - start + 1;
            std::vector<int64_t> build_ids(build_size, 0);
            for (int64_t i = 0, id = start; id <= end; ++id, ++i)
            {
                build_ids[i] = id;
            }

            std::vector<float> build_data(build_size * dim, 0);
            std::copy(vectors.begin() + start * dim,
                      vectors.begin() + start * dim + build_size * dim,
                      build_data.begin());

            auto s = std::chrono::high_resolution_clock::now();
            index->build(build_data, build_ids, build_params, reindexing_params);
            auto e = std::chrono::high_resolution_clock::now();
            double build_time = std::chrono::duration<double, std::milli>(e - s).count();

            std::cout << "Step " << op_i << ": " << op_type << " time: " << build_time << " ms" << std::endl;
        }
        else if (op_type == "insert")
        {
            int start = operation["start"];
            int end = operation["end"];
            std::cout << std::endl;
            std::cout << op_type << " start: " << start << " end: " << end << std::endl;

            int64_t insert_size = end - start + 1;
            double insert_time = 0.0f;
            double reindexing_time = 0.0f;
            if (!is_build)
            {
                for (size_t b = 0; b < div_roundup(insert_size, 100); ++b)
                {
                    size_t begin = b * 100;
                    size_t bs = std::min(insert_size - begin, (size_t)100);
                    std::vector<int64_t> batch_ids_(bs);
                    for (size_t i = 0; i < bs; ++i)
                    {
                        batch_ids_[i] = i + begin + start;
                    }
                    
                    std::vector<float> batch_data_(bs * dim, 0);
                    std::copy(vectors.begin() + start * dim + begin * dim,
                            vectors.begin() + start * dim + begin * dim + bs * dim,
                            batch_data_.begin());

                    auto s1 = std::chrono::high_resolution_clock::now();
                    index->add(batch_data_, batch_ids_);
                    auto e1 = std::chrono::high_resolution_clock::now();
                    insert_time += std::chrono::duration<double, std::milli>(e1 - s1).count();

                    auto s2 = std::chrono::high_resolution_clock::now();
                    index->ReindexingPolicy();
                    auto e2 = std::chrono::high_resolution_clock::now();
                    reindexing_time += std::chrono::duration<double, std::milli>(e2 - s2).count();
                }
                is_build = true;
            } else {
                for (size_t b = 0; b < div_roundup(insert_size, batch_size_); ++b)
                {
                    size_t begin = b * batch_size_;
                    size_t bs = std::min(insert_size - begin, batch_size_);
                    std::vector<int64_t> batch_ids_(bs);
                    for (size_t i = 0; i < bs; ++i)
                    {
                        batch_ids_[i] = i + begin + start;
                    }
                    
                    std::vector<float> batch_data_(bs * dim, 0);
                    std::copy(vectors.begin() + start * dim + begin * dim,
                            vectors.begin() + start * dim + begin * dim + bs * dim,
                            batch_data_.begin());

                    auto s1 = std::chrono::high_resolution_clock::now();
                    index->add(batch_data_, batch_ids_);
                    auto e1 = std::chrono::high_resolution_clock::now();
                    insert_time += std::chrono::duration<double, std::milli>(e1 - s1).count();

                    auto s2 = std::chrono::high_resolution_clock::now();
                    index->ReindexingPolicy();
                    auto e2 = std::chrono::high_resolution_clock::now();
                    reindexing_time += std::chrono::duration<double, std::milli>(e2 - s2).count();
                }
            }
            
            std::cout << "Step " << op_i << ": " << op_type << " time: " << insert_time << " ms" << std::endl;
            std::cout << "reindexing time: " << reindexing_time << std::endl;
            std::cout << "nlist: " << index->nlist() << " ntotal: " << index->ntotal() << std::endl;

            std::vector<std::vector<std::string>> insert_time_result= {{std::to_string(op_i), "insert time", std::to_string(insert_time)}};
            writeCSVApp(output_csv_path, insert_time_result);
            if (reindexing_params->reindexing_strategy == "None")
            {
                std::vector<std::vector<std::string>>  reindexing_time_result= {{std::to_string(op_i), "reindexing time", "0"}};
                writeCSVApp(output_csv_path, reindexing_time_result);
            } else {
                std::vector<std::vector<std::string>> reindexing_time_result= {{std::to_string(op_i), "reindexing time", std::to_string(reindexing_time)}};
                writeCSVApp(output_csv_path, reindexing_time_result);
            }
            std::vector<std::vector<std::string>> index_nlist_result= {{std::to_string(op_i), "nlist", std::to_string(index->nlist())}};
            writeCSVApp(output_csv_path, index_nlist_result);

            // reindexing_params->reindexing_strategy = "None";
        } 
        else if (op_type == "search")
        {
            int num_queries = operation["num_queries"];
            std::string query_file = operation["query_file"];
            std::cout << std::endl;
            std::cout << op_type << " num_queries: " << num_queries << " query_file: " << query_file << std::endl;

            int query_dim = 0;
            int64_t n_queries = 0;
            std::vector<float> queries = read_fvecs(query_file, query_dim, n_queries);
            std::string gt_file_path = gt_dir_path + "gt_step_" + std::to_string(search_step) + ".ivecs";
            std::vector<std::vector<int64_t>> gt_ids = read_ivecs(gt_file_path);
            std::cout << "gt_file: " << gt_file_path << std::endl;

            n_queries = 1000;
            queries.resize(n_queries * dim);
            gt_ids.resize(n_queries);

            auto search_params = std::make_shared<SearchParams>();
            search_params->k = k;
            search_params->nprobe = nprobe;
            search_params->num_threads = 16;
            auto search_results = std::make_shared<SearchResult>();
            double recalls = 0;

            while (search_params->nprobe < index->nlist())
            {
                search_results = index->search(n_queries, queries, search_params);
                
                std::vector<double> recall = compute_recall(search_results->indices, gt_ids, search_params->k);
                recalls = 0;
                for (int64_t i = 0; i < n_queries; ++i)
                {
                    recalls += recall[i];
                }
                recalls = recalls / n_queries;
                
                // if (std::abs(recalls - target_recall) <= 0.005 || search_params->nprobe == 1)
                if (recalls >= target_recall || search_params->nprobe == 1)
                {
                    search_results = index->search(n_queries, queries, search_params);

                    std::cout << "Step " << op_i << ": " << op_type << " time: " << search_results->search_time << " ms" << std::endl;
                    std::cout << "search time: " << search_results->search_time << " ms" << std::endl;
                    std::cout << "center search time: " << search_results->c_search_time << " ms" << std::endl;
                    std::cout << "partition search time: " << search_results->p_search_time << " ms" << std::endl;
                    std::cout << "nprobe: " << search_results->search_nprobe << std::endl;
                    std::cout << "points: " << search_results->search_points << std::endl;
                    std::cout << "qps: " << n_queries * 1000 / search_results->search_time << std::endl;
                    std::cout << "recall: " << recalls << std::endl;

                    std::vector<std::vector<std::string>> search_time_result= {{std::to_string(op_i), "search time", std::to_string(search_results->search_time)}};
                    writeCSVApp(output_csv_path, search_time_result);
                    std::vector<std::vector<std::string>> c_search_time_result= {{std::to_string(op_i), "center search time", std::to_string(search_results->c_search_time)}};
                    writeCSVApp(output_csv_path, c_search_time_result);
                    std::vector<std::vector<std::string>> p_search_time_result= {{std::to_string(op_i), "partition search time", std::to_string(search_results->p_search_time)}};
                    writeCSVApp(output_csv_path, p_search_time_result);
                    std::vector<std::vector<std::string>> nprobe_result= {{std::to_string(op_i), "search nprobe", std::to_string(search_results->search_nprobe)}};
                    writeCSVApp(output_csv_path, nprobe_result);
                    std::vector<std::vector<std::string>> points_result= {{std::to_string(op_i), "search points", std::to_string(search_results->search_points)}};
                    writeCSVApp(output_csv_path, points_result);
                    std::vector<std::vector<std::string>> qps_result= {{std::to_string(op_i), "qps", std::to_string(n_queries * 1000 / search_results->search_time)}};
                    writeCSVApp(output_csv_path, qps_result);
                    std::vector<std::vector<std::string>> recall_result= {{std::to_string(op_i), "recall@" + std::to_string(search_params->k), std::to_string(recalls)}};
                    writeCSVApp(output_csv_path, recall_result);

                    // nprobe = search_results->search_nprobe - 20;
                    break;
                }
                // std::cout << recalls << std::endl;
                if (recalls > target_recall)
                {
                    search_params->nprobe--;
                } else {
                    search_params->nprobe++;
                }
            }
            nprobe = search_params->nprobe - 20;
            search_step++;
        }
        else if (op_type == "delete")
        {
            int start = operation["start"];
            int end = operation["end"];
            std::cout << std::endl;
            std::cout << op_type << " start: " << start << " end: " << end << std::endl;
            
            int64_t delete_size = end - start + 1;
            double delete_time = 0.0f;
            double reindexing_time = 0.0f;
            
            std::vector<int64_t> delete_ids(delete_size, 0);
            for (int64_t i = 0, id = start; id <= end; ++id, ++i)
            {
                delete_ids[i] = id;
            }

            auto s1 = std::chrono::high_resolution_clock::now();
            index->remove(delete_ids);
            auto e1 = std::chrono::high_resolution_clock::now();
            delete_time += std::chrono::duration<double, std::milli>(e1 - s1).count();

            auto s2 = std::chrono::high_resolution_clock::now();
            index->ReindexingPolicy();
            auto e2 = std::chrono::high_resolution_clock::now();
            reindexing_time += std::chrono::duration<double, std::milli>(e2 - s2).count();

            std::cout << "Step " << op_i << ": " << op_type << " time: " << delete_time << " ms" << std::endl;
            std::cout << "reindexing time: " << reindexing_time << std::endl;
            std::cout << "nlist: " << index->nlist() << " ntotal: " << index->ntotal() << std::endl;

            std::vector<std::vector<std::string>> delete_time_result= {{std::to_string(op_i), "delete time", std::to_string(delete_time)}};
            writeCSVApp(output_csv_path, delete_time_result);
            if (reindexing_params->reindexing_strategy == "None")
            {
                std::vector<std::vector<std::string>>  reindexing_time_result= {{std::to_string(op_i), "reindexing time", "0"}};
                writeCSVApp(output_csv_path, reindexing_time_result);
            } else {
                std::vector<std::vector<std::string>> reindexing_time_result= {{std::to_string(op_i), "reindexing time", std::to_string(reindexing_time)}};
                writeCSVApp(output_csv_path, reindexing_time_result);
            }
            std::vector<std::vector<std::string>> index_nlist_result= {{std::to_string(op_i), "nlist", std::to_string(index->nlist())}};
            writeCSVApp(output_csv_path, index_nlist_result);
        }
    }

    return 0;
}