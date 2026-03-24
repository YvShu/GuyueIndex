/*
 * @Author: Guyue
 * @Date: 2026-03-23 09:44:44
 * @LastEditTime: 2026-03-23 09:50:45
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/utils.h
 */
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>
#include <filesystem>
#include <nlohmann/json.hpp>

/**
 * @brief: 读取.fvecs格式的文件
 * @param {string&} filename 文件名
 * @param {int&} dim 向量维度
 * @param {int64_t&} n_vectors 向量数量
 * @return {*} 平铺的向量
 */
std::vector<float> read_fvecs(const std::string& filename, int& dim, int64_t& n_vectors)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input)
    {
        std::cerr << "Can't open the file: " << filename << std::endl;
        return {};
    }

    input.read(reinterpret_cast<char*>(&dim), sizeof(int));
    input.seekg(0, std::ios::end);
    size_t file_size = input.tellg();
    n_vectors = file_size / ((dim + 1) * sizeof(float));
    input.seekg(0, std::ios::beg);

    const size_t total_float = n_vectors * dim;
    std::vector<float> data;
    data.reserve(total_float);
    
    constexpr size_t BLOCK_SIZE = 1000000;
    std::vector<float> buffer(dim * BLOCK_SIZE);
    
    for (size_t offset = 0; offset < n_vectors;)
    {
        const size_t block = std::min(BLOCK_SIZE, n_vectors - offset);
        const size_t block_size_floats = block * dim;

        if (buffer.size() < block_size_floats) 
        {
            buffer.resize(block_size_floats);
        }

        float* block_ptr = buffer.data();
        for (size_t i = 0; i < block; ++i)
        {
            input.ignore(sizeof(int));
            input.read(reinterpret_cast<char*>(block_ptr), dim * sizeof(float));
            block_ptr += dim;
        }

        data.insert(data.end(), buffer.begin(), buffer.begin() + block_size_floats);
        offset += block;
    }

    input.close();
    return data;
}

/**
 * @brief: 读取.ivecs格式的文件
 * @param {string&} filename 文件名
 * @return {*} 平铺的向量
 */
std::vector<std::vector<int64_t>> read_ivecs(const std::string& filename)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input)
    {
        std::cerr << "Can't open the file: " << filename << std::endl;
        return {};
    }

    std::vector<std::vector<int64_t>> indices;
    while (input.peek() != EOF)
    {
        int k;
        input.read(reinterpret_cast<char*>(&k), sizeof(int));

        std::vector<int> vec(k);
        input.read(reinterpret_cast<char*>(vec.data()), k * sizeof(int));
        indices.emplace_back(vec.begin(), vec.end());
    }
    input.close();
    
    return indices;
}

/**
 * @brief: 读取.json格式的向量
 * @param {string&} filename 文件名
 * @return {*} 解析的json文件
 */
nlohmann::json read_json(const std::string& filename)
{
    nlohmann::json output_file;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
    }
    
    try {
        file >> output_file;
        std::cout << "Runbook loaded successfully!" << std::endl;
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "JSON parse error: " << e.what() << std::endl;
    }
    return output_file;
}

/**
 * @brief: 计算批次数
 * @param {size_t} num 数据量
 * @param {size_t} denom 批次大小
 * @return {*} 批次数
 */
size_t div_roundup(size_t num, size_t denom) 
{
    return (num + static_cast<size_t>(denom) - static_cast<size_t>(1)) / static_cast<size_t>(denom);
}

/**
 * @brief: 计算召回率
 * @param {std::vector<std::vector<int64_t>>&} ids 查询结果ids
 * @param {std::vector<std::vector<int64_t>>&} gt_ids 真实结果ids
 * @param {int} k TopK
 * @return {*} recall@k
 */
std::vector<double> compute_recall(std::vector<std::vector<int64_t>>& ids, std::vector<std::vector<int64_t>>& gt_ids, int k)
{
    if (ids.empty() || gt_ids.empty()) {
        throw std::invalid_argument("[utils] compute_recall : Input arrays cannot be empty.");
    }
    if (ids.size() != gt_ids.size()) {
        throw std::invalid_argument("[utils] compute_recall : Number of queries must be the same for ids and gt_ids.");
    }

    int64_t n_queries = ids.size();
    std::vector<double> recall(n_queries, 0.0);

    for (int64_t i = 0; i < n_queries; ++i)
    {
        if (ids[i].size() < k || gt_ids[i].size() < k)
        {
            throw std::invalid_argument("[utils] compute_recall : Each query must have at least k results.");
        }

        std::unordered_set<int64_t> gt_set;
        for (int j = 0; j < k; ++j)
        {
            gt_set.insert(gt_ids[i][j]);
        }
        int interesction_count = 0;
        for (int j = 0; j < k; ++j)
        {
            if (gt_set.find(ids[i][j]) != gt_set.end())
            {
                interesction_count++;
            }
        }

        recall[i] = static_cast<double>(interesction_count) / k;
    }

    return recall;
}

/**
 * @brief: 将结果追加写入到.csv文件
 * @param {string&} filename 文件名
 * @param {std::vector<std::vector<std::string>>&} data 写入内容
 * @return {*}
 */
void writeCSVApp(const std::string& filename, const std::vector<std::vector<std::string>>& data) 
{
    std::filesystem::path file_path(filename);
    std::filesystem::path dir_path = file_path.parent_path();

    // Check if directory exists, if not, create it
    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }

    std::ofstream file;
    file.open(filename, std::ios_base::app);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    for (const auto& row : data) {
        std::stringstream ss;
        for (size_t i = 0; i < row.size(); ++i) {
            ss << row[i];
            if (i < row.size() - 1) {
                ss << ",";
            }
        }
        file << ss.str() << "\n";
    }

    file.close();
}

#endif // UTILS_H