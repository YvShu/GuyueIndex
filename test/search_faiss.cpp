// /*
//  * @Author: Guyue
//  * @Date: 2026-03-23 17:30:55
//  * @LastEditTime: 2026-03-24 10:59:11
//  * @LastEditors: Guyue
//  * @FilePath: /GuyueIndex/test/search_faiss.cpp
//  */
// #include <guyue_index.h>
// #include <faiss/IndexFlat.h>
// #include <faiss/IndexIVFFlat.h>
// #include <omp.h>
// #include <utils.h>

// int main()
// {
//     std::string dataset = "sift-1M";
//     size_t pos = dataset.find('-');
//     std::string dataset_name = dataset.substr(0, pos);
//     int k = 100;
//     std::vector<float> target_recalls = {0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 0.9875, 1.0};
//     // std::vector<int64_t> search_nprobes = {9, 11, 12, 15, 18, 23, 32, 51, 75, 114}; // deep

//     std::string vectors_file_path = "/mnt/hgfs/DataSet/" + dataset + "/"+ dataset_name +"_base.fvecs";
//     std::string queries_file_path = "/mnt/hgfs/DataSet/" + dataset + "/"+ dataset_name +"_query.fvecs";
//     std::string gt_file_path = "/mnt/hgfs/DataSet/" + dataset + "/gt_step_0.ivecs";
//     const std::string output_csv_path = "../output/search_test/" + dataset + "_Faiss_" + std::to_string(k) + ".csv";

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
//     faiss::IndexFlatL2 quantizer = faiss::IndexFlatL2(dim);
//     faiss::IndexIVFFlat index(&quantizer, dim, 1000, faiss::METRIC_L2);
    
//     std::vector<int64_t> vector_ids(n_vectors, 0);
//     for (int64_t i = 0; i < n_vectors; ++i)
//     {
//         vector_ids[i] = i;
//     }

//     auto s = std::chrono::high_resolution_clock::now();
//     index.train(n_vectors, vectors.data());
//     index.add_with_ids(n_vectors, vectors.data(), vector_ids.data());
//     auto e = std::chrono::high_resolution_clock::now();
//     double build_time = std::chrono::duration<double>(e - s).count();

//     std::cout << "Index Build Finish!" << "" << std::endl;
//     std::cout << "build time: " << build_time << std::endl;
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
//     std::vector<faiss::idx_t> search_ids(n_queries * search_params->k);
//     std::vector<float> search_dists(n_queries * search_params->k);

//     for (int i = 0; i < target_recalls.size(); ++i)
//     {
//         target_recall = target_recalls[i];

//         while (search_params->nprobe < index.nlist)
//         {
//             index.nprobe = search_params->nprobe;
//             auto s = std::chrono::high_resolution_clock::now();
//             index.search(n_queries, queries.data(), search_params->k, search_dists.data(), search_ids.data());
//             auto e = std::chrono::high_resolution_clock::now();
//             double search_time = std::chrono::duration<double, std::milli>(e - s).count();

//             std::vector<std::vector<int64_t>> ids(n_queries, std::vector<int64_t>(search_params->k, 0));
//             for (int64_t i = 0; i < n_queries; ++i)
//             {
//                 for (int64_t j = 0; j < search_params->k; ++j)
//                 {
//                     ids[i][j] = search_ids[i * search_params->k + j];
//                 }
//             }
//             std::vector<double> recalls = compute_recall(ids, gt_ids, search_params->k);
//             recall = 0;
//             for (int64_t i = 0; i < n_queries; ++i)
//             {
//                 recall += recalls[i];
//             }
//             recall = recall / n_queries;

//             if (recall >= target_recall || (target_recall == 1.0 && abs(recall - target_recall) <= 0.005))
//             {
//                 std::cout << "search time: " << search_time << " ms" << std::endl;
//                 // std::cout << "center search time: " << search_results->c_search_time << " ms" << std::endl;
//                 // std::cout << "partition search time: " << search_results->p_search_time << " ms" << std::endl;
//                 std::cout << "nprobe: " << search_params->nprobe << std::endl;
//                 // std::cout << "points: " << search_results->search_points << std::endl;
//                 std::cout << "qps: " << n_queries * 1000 / search_time << std::endl;
//                 std::cout << "recall: " << recall << std::endl;

//                 std::vector<std::vector<std::string>> search_time_result= {{std::to_string(i+1), "search time", std::to_string(search_time)}};
//                 writeCSVApp(output_csv_path, search_time_result);
//                 // std::vector<std::vector<std::string>> c_search_time_result= {{std::to_string(i+1), "center search time", std::to_string(search_results->c_search_time)}};
//                 // writeCSVApp(output_csv_path, c_search_time_result);
//                 // std::vector<std::vector<std::string>> p_search_time_result= {{std::to_string(i+1), "partition search time", std::to_string(search_results->p_search_time)}};
//                 // writeCSVApp(output_csv_path, p_search_time_result);
//                 std::vector<std::vector<std::string>> nprobe_result= {{std::to_string(i+1), "search nprobe", std::to_string(search_params->nprobe)}};
//                 writeCSVApp(output_csv_path, nprobe_result);
//                 // std::vector<std::vector<std::string>> points_result= {{std::to_string(i+1), "search points", std::to_string(search_results->search_points)}};
//                 // writeCSVApp(output_csv_path, points_result);
//                 std::vector<std::vector<std::string>> qps_result= {{std::to_string(i+1), "qps", std::to_string(n_queries * 1000 / search_time)}};
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