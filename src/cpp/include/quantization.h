/*
 * @Author: Guyue
 * @Date: 2026-03-31 09:40:43
 * @LastEditTime: 2026-03-31 09:57:18
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/quantization.h
 */
#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <common.h>

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

    float step = (max_val - min_val) / 255.0f;
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

} // guyue

#endif // QUANTIZATION_H