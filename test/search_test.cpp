// /*
//  * @Author: Guyue
//  * @Date: 2026-03-23 17:02:39
//  * @LastEditTime: 2026-04-03 16:12:25
//  * @LastEditors: Guyue
//  * @FilePath: /GuyueIndex/test/search_test.cpp
//  */
// #include <utils.h>
// #include <guyue_index.h>

// int main()
// {
//     std::string dataset = "sift-1M";
//     size_t pos = dataset.find('-');
//     std::string dataset_name = dataset.substr(0, pos);
//     int k = 10;
//     std::vector<float> target_recalls = {0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.9875, 1.0};
//     // std::vector<int64_t> search_nprobes = {9, 11, 12, 15, 18, 23, 32, 51, 75, 114}; // deep

//     std::string vectors_file_path = "/mnt/hgfs/DataSet/" + dataset + "/"+ dataset_name +"_base.fvecs";
//     std::string queries_file_path = "/mnt/hgfs/DataSet/" + dataset + "/"+ dataset_name +"_query.fvecs";
//     std::string gt_file_path = "/mnt/hgfs/DataSet/" + dataset + "/gt_step_0.ivecs";
//     const std::string output_csv_path = "../output/search_test/" + dataset + "_TEST_" + std::to_string(k) + ".csv";

//     //////////////////////////////////////////
//     /// 向量读取
//     //////////////////////////////////////////
//     int dim = 0;
//     int64_t n_vectors = 0;
//     int64_t n_queries = 0;
//     std::vector<float> vectors = read_fvecs(vectors_file_path, dim, n_vectors);
//     std::vector<float> queries = read_fvecs(queries_file_path, dim, n_queries);
//     std::vector<std::vector<int64_t>> gt_ids = read_ivecs(gt_file_path);
//     std::cout << "vectors dim: " << dim << std::endl;
//     std::cout << "vectors size: " << n_vectors << std::endl;
//     std::cout << "queries size: " << n_queries << std::endl;
//     std::cout << "========================================================================"<< std::endl;

//     //////////////////////////////////////////
//     /// 索引构建
//     //////////////////////////////////////////
//     auto index = std::make_shared<GuyueIndex>();
//     auto build_params = std::make_shared<IndexBuildParams>();
//     build_params->dimension = dim;
//     build_params->nlist = 2600;
//     build_params->niter = 20;
//     build_params->metric = "l2";

//     auto reindexing_params = std::make_shared<ReindexingParams>();
//     reindexing_params->reindexing_strategy = "None";

//     auto pq_params = std::make_shared<PQParams>();

//     std::vector<int64_t> vector_ids(n_vectors, 0);
//     for (int64_t i = 0; i < n_vectors; ++i)
//     {
//         vector_ids[i] = i;
//     }

//     auto s = std::chrono::high_resolution_clock::now();
//     index->build(vectors, vector_ids, build_params, reindexing_params, pq_params);
//     // index->build(vectors, vector_ids, build_params, reindexing_params);
//     auto e = std::chrono::high_resolution_clock::now();
//     double build_time = std::chrono::duration<double>(e - s).count();

//     std::cout << "Index Build Finish!" << "" << std::endl;
//     std::cout << "build time: " << build_time << std::endl;
//     std::cout << "========================================================================"<< std::endl;

//     //////////////////////////////////////////
//     /// 查询统计
//     //////////////////////////////////////////
//     n_queries = 100;
//     queries.resize(n_queries * dim);
//     gt_ids.resize(n_queries);
    
//     int64_t nlist = index->nlist();
//     auto centroids_search_result = std::make_shared<SearchResult>();
//     auto centroids_search_params = std::make_shared<SearchParams>();
//     centroids_search_params->k = nlist;
//     centroids_search_result = index->searcher_->search_centers(
//         index->centroids_manager_,
//         n_queries,
//         queries,
//         centroids_search_params
//     );
//     double c_search_time = centroids_search_result->c_search_time;
//     std::cout << "center search finish!" << std::endl;
//     std::cout << "========================================================================"<< std::endl;

//     //////////////////////////////////////////
//     /// 查询执行
//     //////////////////////////////////////////
//     auto search_params = std::make_shared<SearchParams>();
//     search_params->k = k;
//     search_params->nprobe = 5;
//     search_params->num_threads = 16;
//     auto search_results = std::make_shared<SearchResult>();
    
//     double recall = 0;
//     float target_recall = 0.0;

//     for (int i = 0; i < target_recalls.size(); ++i)
//     {
//         target_recall = target_recalls[i];

//         while (search_params->nprobe <= index->nlist())
//         {
//             auto s = std::chrono::high_resolution_clock::now();
//             search_results = index->search(n_queries, queries, search_params);
//             auto e = std::chrono::high_resolution_clock::now();
//             search_results->search_time = std::chrono::duration<double, std::milli>(e - s).count();
//             search_results->search_nprobe = search_params->nprobe;

//             std::vector<double> recalls = compute_recall(search_results->indices, gt_ids, k);
            
//             recall = 0;
//             for (int64_t i = 0; i < n_queries; ++i)
//             {
//                 recall += recalls[i];
//             }
//             recall = recall / n_queries;

//             std::cout << "search time: " << search_results->search_time << " ms" << std::endl;
//             std::cout << "center search time: " << search_results->c_search_time << " ms" << std::endl;
//             std::cout << "partition search time: " << search_results->p_search_time << " ms" << std::endl;
//             std::cout << "nprobe: " << search_results->search_nprobe << std::endl;
//             std::cout << "points: " << search_results->search_points << std::endl;
//             std::cout << "qps: " << n_queries * 1000 / search_results->search_time << std::endl;
//             std::cout << "recall: " << recall << std::endl;

//             if (recall >= target_recall || (target_recall == 1.0 && abs(recall - target_recall) <= 0.005))
//             {
//                 std::cout << "search time: " << search_results->search_time << " ms" << std::endl;
//                 std::cout << "center search time: " << search_results->c_search_time << " ms" << std::endl;
//                 std::cout << "partition search time: " << search_results->p_search_time << " ms" << std::endl;
//                 std::cout << "nprobe: " << search_results->search_nprobe << std::endl;
//                 std::cout << "points: " << search_results->search_points << std::endl;
//                 std::cout << "qps: " << n_queries * 1000 / search_results->search_time << std::endl;
//                 std::cout << "recall: " << recall << std::endl;

//                 std::vector<std::vector<std::string>> search_time_result= {{std::to_string(i+1), "search time", std::to_string(search_results->search_time)}};
//                 writeCSVApp(output_csv_path, search_time_result);
//                 std::vector<std::vector<std::string>> c_search_time_result= {{std::to_string(i+1), "center search time", std::to_string(search_results->c_search_time)}};
//                 writeCSVApp(output_csv_path, c_search_time_result);
//                 std::vector<std::vector<std::string>> p_search_time_result= {{std::to_string(i+1), "partition search time", std::to_string(search_results->p_search_time)}};
//                 writeCSVApp(output_csv_path, p_search_time_result);
//                 std::vector<std::vector<std::string>> nprobe_result= {{std::to_string(i+1), "search nprobe", std::to_string(search_results->search_nprobe)}};
//                 writeCSVApp(output_csv_path, nprobe_result);
//                 std::vector<std::vector<std::string>> points_result= {{std::to_string(i+1), "search points", std::to_string(search_results->search_points)}};
//                 writeCSVApp(output_csv_path, points_result);
//                 std::vector<std::vector<std::string>> qps_result= {{std::to_string(i+1), "qps", std::to_string(n_queries * 1000 / search_results->search_time)}};
//                 writeCSVApp(output_csv_path, qps_result);
//                 std::vector<std::vector<std::string>> recall_result= {{std::to_string(i+1), "recall@" + std::to_string(search_params->k), std::to_string(recall)}};
//                 writeCSVApp(output_csv_path, recall_result);
//                 std::cout << "----------------------------------------------------Save Success!" << std::endl;

//                 break;
//             }

//             if (recall > target_recall)
//             {
//                 search_params->nprobe--;
//             } else {
//                 search_params->nprobe++;
//             }
//         }
//     }

//     return 0;
// }