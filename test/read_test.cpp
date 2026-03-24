// /*
//  * @Author: Guyue
//  * @Date: 2026-03-23 09:52:37
//  * @LastEditTime: 2026-03-23 16:59:52
//  * @LastEditors: Guyue
//  * @FilePath: /GuyueIndex/test/read_test.cpp
//  */
// #include <utils.h>

// int main()
// {
//     std::string vectors_file_path = "/mnt/hgfs/DataSet/sift-1M/sift_base.fvecs";
//     std::string queries_file_path = "/mnt/hgfs/DataSet/sift-1M/sift_query.fvecs";
//     std::string gt_file_path = "/mnt/hgfs/DataSet/sift-1M/gt_step_0.ivecs";

//     int dim = 0;
//     int64_t n_vectors = 0;
//     int64_t n_queries = 0;
//     std::vector<float> vectors = read_fvecs(vectors_file_path, dim, n_vectors);
//     std::vector<float> queries = read_fvecs(queries_file_path, dim, n_queries);
//     std::vector<std::vector<int64_t>> gt_ids = read_ivecs(gt_file_path);

//     std::cout << "Dimension: " << dim << std::endl;
//     std::cout << "Data Size: " << n_vectors << std::endl;
//     std::cout << "Query Size: " << n_queries << std::endl;
//     std::cout << "GT Size: " << gt_ids.size() << std::endl;

//     return 0;
// }