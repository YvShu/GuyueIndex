/*
 * @Author: Guyue
 * @Date: 2026-03-31 09:40:43
 * @LastEditTime: 2026-04-08 16:35:07
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/quantization.h
 */
#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <common.h>
#include <immintrin.h>

namespace guyue {
/**
 * @brief: 将单条float向量压缩为uint8格式
 * @param {float*} fvec 原始float向量指针
 * @param {uint8_t*} code 目标写入地址，要求预分配大小为(d+8)字节
 * @param {int} dim 向量维度
 * @return {*}
 */
inline void encode(const float* fvec, uint8_t* code, int dim)
{
    float min_val = fvec[0], max_val = fvec[0];
    for (int i = 0; i < dim; ++i)
    {
        if (fvec[i] < min_val) min_val = fvec[i];
        if (fvec[i] > max_val) max_val = fvec[i];
    }

    float step = 1;
    // if (max_val <= 255)
    // {
    //     min_val = std::min(0.0f, min_val);
    // } else {
        step = (max_val - min_val) / 255.0f;
    // }
    for (int i = 0; i < dim; ++i)
    {
        if (step == 0.0f)
        {
            code[i] = 0;
        } else {
            int val = std::round((fvec[i] - min_val) / step);
            code[i] = static_cast<uint8_t>(std::max(0, std::min(255, val)));
        }
    }
    // 将 min_val 和 step(各4字节)附在向量尾部
    std::memcpy(code + dim, &min_val, sizeof(float));
    std::memcpy(code + dim + sizeof(float), &step, sizeof(float));
}

/**
 * @brief: 将8比特格式的向量解压恢复为float32类型
 * @param {uint8_t*} code 量化向量
 * @param {float*} fvec 目标写入地址，要求预分配大小d*32字节
 * @param {int} dim 向量维度
 * @return {*}
 */
inline void decode(const uint8_t* code, float* fvec, int dim)
{
    float min_val, step;
    std::memcpy(&min_val, code + dim, sizeof(float));
    std::memcpy(&step, code + dim + sizeof(float), sizeof(float));
    for (int i = 0; i < dim; ++i)
    {
        fvec[i] = code[i] * step + min_val;
    }
}

/**
 * @brief 批量编码浮点向量
 * @param fvecs 输入的浮点向量矩阵，大小为 num_vectors * dim
 * @param codes 输出的量化字节矩阵，预分配大小需为 num_vectors * (dim + 2 * sizeof(float))
 * @param num_vectors 向量的数量
 * @param dim 每个向量的维度
 */
inline void encode_batch(const float* fvecs, uint8_t* codes, int num_vectors, int dim) 
{
    // 编码后单条向量的字节长度：原始维度 + 2个float的元数据大小
    size_t code_size_per_vec = dim + 2 * sizeof(float);
    
    for (int i = 0; i < num_vectors; ++i) {
        // 计算当前向量在输入和输出数组中的起始指针位置并调用单条编码
        encode(fvecs + i * dim, codes + i * code_size_per_vec, dim);
    }
}

/**
 * @brief 批量解码浮点向量
 * @param codes 输入的量化字节矩阵，大小为 num_vectors * (dim + 2 * sizeof(float))
 * @param fvecs 输出的浮点向量矩阵，预分配大小需为 num_vectors * dim
 * @param num_vectors 向量的数量
 * @param dim 每个向量的维度
 */
inline void decode_batch(const uint8_t* codes, float* fvecs, int num_vectors, int dim) 
{
    size_t code_size_per_vec = dim + 2 * sizeof(float); // 8 + 2 * 4
    
    for (int i = 0; i < num_vectors; ++i) {
        decode(codes + i * code_size_per_vec, fvecs + i * dim, dim);
    }
}

// AVX2
inline void encode_avx2(const float* fvec, uint8_t* code, int dim)
{
    __m256 v_min = _mm256_set1_ps(fvec[0]);
    __m256 v_max = _mm256_set1_ps(fvec[0]);

    // 找极值
    for (int i = 0; i < dim; i += 8)
    {
        __m256 v = _mm256_loadu_ps(fvec + i);
        v_min = _mm256_min_ps(v_min, v);
        v_max = _mm256_max_ps(v_max, v);
    }

    // 水平提取极值
    float res_min[8], res_max[8];
    _mm256_storeu_ps(res_min, v_min);
    _mm256_storeu_ps(res_max, v_max);
    float min_val = res_min[0], max_val = res_max[0];
    for(int i=1; i<8; ++i) {
        min_val = std::min(min_val, res_min[i]);
        max_val = std::max(max_val, res_max[i]);
    }

    float step = (max_val - min_val) / 255.0f;
    float inv_step = (step == 0) ? 0 : 1.0f / step;

    __m256 v_min_val = _mm256_set1_ps(min_val);
    __m256 v_inv_step = _mm256_set1_ps(inv_step);

    // 向量化量化
    for (int i = 0; i < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(fvec + i);
        // (v - min) * inv_step
        __m256 v_norm = _mm256_mul_ps(_mm256_sub_ps(v, v_min_val), v_inv_step);
        // 转换为 32位整数并截断
        __m256i v_int = _mm256_cvtps_epi32(_mm256_round_ps(v_norm, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC));
        
        // 打包成 uint8 (通过两次剪裁打包)
        // 这里简化演示，实际需注意存储顺序
        for(int j=0; j<8; ++j) {
            code[i + j] = (uint8_t)((int*) &v_int)[j];
        }
    }
    // 尾部存储元数据
    memcpy(code + dim, &min_val, 4);
    memcpy(code + dim + 4, &step, 4);
}

inline float query_uint8_ip_avx2(const float* q, const uint8_t* code, int dim) {
    float min_val, step;
    memcpy(&min_val, code + dim, 4);
    memcpy(&step, code + dim + 4, 4);

    __m256 sum = _mm256_setzero_ps();
    __m256 v_step = _mm256_set1_ps(step);
    __m256 v_min = _mm256_set1_ps(min_val);
    
    // 预计算 Query 的和，用于处理 min_val 部分
    // Score = sum(q * (c*step + min)) = step * sum(q*c) + min * sum(q)
    // 这里的实现采用直接计算，以适应复杂的 SQ 情况
    for (int i = 0; i < dim; i += 8) {
        __m256 v_q = _mm256_loadu_ps(q + i);
        
        // 实时解码：将 8 个 uint8 转换为 8 个 float
        __m128i v_8int = _mm_loadl_epi64((__m128i*)(code + i));
        __m256i v_32int = _mm256_cvtepu8_epi32(v_8int);
        __m256 v_c = _mm256_cvtepi32_ps(v_32int);
        
        // v_db = v_c * step + min
        __m256 v_db = _mm256_fmadd_ps(v_c, v_step, v_min);
        sum = _mm256_fmadd_ps(v_q, v_db, sum);
    }

    // 水平求和
    float res[8];
    _mm256_storeu_ps(res, sum);
    return res[0]+res[1]+res[2]+res[3]+res[4]+res[5]+res[6]+res[7];
}

inline float query_uint8_l2_avx2(const float* q, const uint8_t* code, int dim) {
    float min_val, step;
    memcpy(&min_val, code + dim, 4);
    memcpy(&step, code + dim + 4, 4);

    __m256 sum_sqr = _mm256_setzero_ps();
    __m256 v_step = _mm256_set1_ps(step);
    __m256 v_min = _mm256_set1_ps(min_val);

    for (int i = 0; i < dim; i += 8) {
        __m256 v_q = _mm256_loadu_ps(q + i);
        
        __m128i v_8int = _mm_loadl_epi64((__m128i*)(code + i));
        __m256 v_c = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(v_8int));
        
        // 解码并直接算差值：diff = q - (c * step + min)
        __m256 v_db = _mm256_fmadd_ps(v_c, v_step, v_min);
        __m256 v_diff = _mm256_sub_ps(v_q, v_db);
        sum_sqr = _mm256_fmadd_ps(v_diff, v_diff, sum_sqr);
    }

    // 水平求和
    float res[8];
    _mm256_storeu_ps(res, sum_sqr);
    return res[0]+res[1]+res[2]+res[3]+res[4]+res[5]+res[6]+res[7];
}

} // guyue

#endif // QUANTIZATION_H