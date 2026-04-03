/*
 * @Author: Guyue
 * @Date: 2026-03-23 10:03:48
 * @LastEditTime: 2026-04-03 14:16:11
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/partition_base.h
 */
#ifndef PARTITION_BASE_H
#define PARTITION_BASE_H

#include <common.h>
#include <immintrin.h>

/**
 * @brief: 一个已编码的向量分区(单个)
 */
class PartitionBase{
public:
    int64_t buffer_size_ = 0;       // 已分配的容量(能够存储的向量数量)
    int64_t num_vectors_ = 0;       // 当前存储向量数量
    int64_t code_size_ = 0;         // 以字节为单位时每个向量编码的大小

    uint8_t* codes_ = nullptr;      // 已编码向量的指针
    faiss::idx_t* ids_ = nullptr;   // 向量IDs

    /**
     * @brief: 默认构造函数
     * @return {*}
     */    
    PartitionBase() = default;

    /**
     * @brief: 使用给定的向量和IDs初始化分区
     * @param {int64_t} num_vectors 向量数量
     * @param {uint8_t*} codes 指向已编码向量缓冲区的指针
     * @param {idx_t*} ids 向量IDs
     * @param {int64_t} code_size 每个向量编码的字节数
     * @return {*}
     */    
    PartitionBase(int64_t num_vectors, uint8_t* codes, faiss::idx_t* ids, int64_t code_size);

    /**
     * @brief: 利用现有的PartitionBase进行初始化
     * @param {PartitionBase&&} other 现有PartitionBase
     * @return {*}
     */    
    PartitionBase(PartitionBase&& other) noexcept;

    /**
     * @brief: 从另一个分区传输信息，清除现有数据
     * @return {*}
     */    
    PartitionBase& operator=(PartitionBase&& other) noexcept;

    /**
     * @brief: 析构函数，释放已分配的内存
     * @return {*}
     */    
    ~PartitionBase();

    /**
     * @brief: 设置向量的编码大小(一个向量占多少个字节)
     * @return {*}
     */    
    void set_code_size(int64_t code_size);

    /**
     * @brief: 在当前分区末尾追加n个新向量
     * @param {int64_t} n_entries 追加的向量数量
     * @param {idx_t*} ids 追加的向量IDs
     * @param {uint8_t*} codes 指向要追加的已编码向量的缓存指针
     * @return {*}
     */    
    void append(int64_t n_entries, const faiss::idx_t* ids, const uint8_t* codes);

    /**
     * @brief: 对当前分区向量编码进行覆写
     * @param {int64_t} offset 开始覆写的起始位置
     * @param {int64_t} n_entries 要覆写的向量个数
     * @param {idx_t*} ids 新向量IDs
     * @param {uint8_t*} codes 指向要覆写的已编码向量的缓存指针
     * @return {*}
     */    
    void update(int64_t offset, int64_t n_entries, const faiss::idx_t* ids, const uint8_t* codes);

    /**
     * @brief: 将指定id的向量从当前分区中移除
     * @param {int64_t} id 向量id
     * @return {*}
     */    
    void remove(int64_t id);

    /**
     * @brief: 修改分区的容量大小
     * @param {int64_t} new_capacity 修改后的分区容量(能容纳的向量数量)
     * @return {*}
     */    
    void resize(int64_t new_capacity);

    /**
     * @brief: 释放所有已分配的内存，重置分区状态
     * @return {*}
     */    
    void clear();

    /**
     * @brief: 根据向量id找到该向量在分区中的位置
     * @param {idx_t} id 向量id
     * @return {*}
     */    
    int64_t find_pos_of(faiss::idx_t id) const;

    /**
     * @brief: 为分区重分配新的容量并拷贝过去
     * @param {int64_t} new_capacity 新的容量大小(能容纳的向量数量)
     * @return {*}
     */    
    void reallocate_memory(int64_t new_capacity);

private:
    /**
     * @brief: 从另一个分区移动数据
     * @param {IndexPartition&&} other 要移动的分区
     * @return {*}
     */
    void move_from(PartitionBase&& other);

    /**
     * @brief: 释放已分配的内存(编码和IDs)
     * @return {*}
     */    
    void free_memory();

    /**
     * @brief: 检查缓冲区是否可以容纳所需数量的向量(必要时进行大小调整)
     * @param {int64_t} required 最小所需向量数量
     * @return {*}
     */    
    void ensure_capacity(int64_t required);
    
    template <typename T>
    T* allocate_memory(size_t num_elements); 
};

#endif // PARTITION_BASE_H