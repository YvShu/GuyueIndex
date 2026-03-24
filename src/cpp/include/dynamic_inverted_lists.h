/*
 * @Author: Guyue
 * @Date: 2026-03-23 10:24:50
 * @LastEditTime: 2026-03-23 10:25:10
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/include/dynamic_inverted_lists.h
 */
#ifndef DYNAMIC_INVERTED_LIST_H
#define DYNAMIC_INVERTED_LIST_H

#include <common.h>
#include <faiss/invlists/InvertedLists.h>
#include <partition_base.h>

namespace faiss {
    class DynamicInvertedLists : public InvertedLists {
    public:
        int curr_list_id_ = 0;  // 下一可用的分区ID
        int dimension_;         // 向量维度
        int code_size_;         // 每个向量的字节数量
        std::unordered_map<size_t, std::shared_ptr<PartitionBase>> partitions_;

        /**
         * @brief: 以给定的分区数量和向量编码字节数初始化
         * @param {size_t} nlist 初始化分区数量
         * @param {size_t} code_size 每个向量编码字节数
         * @return {*}
         */        
        DynamicInvertedLists(size_t nlist, size_t code_size);

        /**
         * @brief: 析构函数，通过每个PartitionBase的析构函数释放内存
         * @return {*}
         */        
        ~DynamicInvertedLists() override;

        /**
         * @brief: 通过每个分区的向量数返回总向量数
         * @return {*} 总向量数
         */        
        size_t ntotal() const;

        /**
         * @brief: 获取指定分区中的向量数
         * @param {size_t} list_id 分区ID
         * @return {*} 分区ID中的向量数量
         */        
        size_t list_size(size_t list_id) const override;

        /**
         * @brief: 获取指定分区的向量编码指针
         * @param {size_t} list_id 分区ID
         * @return {*} 分区ID对应的向量编码指针
         */        
        const uint8_t* get_codes(size_t list_id) const override;

        /**
         * @brief: 获取指定分区的向量ids指针
         * @param {size_t} list_id 分区ID
         * @return {*} 分区ID对应的向量ids指针
         */        
        const faiss::idx_t* get_ids(size_t list_id) const override;

        /**
         * @brief: 释放向量编码指针
         * @param {size_t} list_id 分区ID
         * @param {uint8_t*} codes
         * @return {*}
         */        
        void release_codes(size_t list_id, const uint8_t* codes) const override;

        /**
         * @brief: 释放向量ids指针
         * @param {size_t} list_id 分区ID
         * @param {uint8_t*} codes
         * @return {*}
         */        
        void release_ids(size_t list_id, const faiss::idx_t* ids) const override;

        /**
         * @brief: 从指定分区中移除指定id的向量
         * @param {size_t} list_id 分区ID
         * @param {idx_t} id 要移除的向量id
         * @return {*}
         */        
        void remove_entry(size_t list_id, faiss::idx_t id);

        /**
         * @brief: 从指定分区中移除指定ids的向量
         * @param {size_t} list_id 分区ID
         * @param {vector<faiss::idx_t>} ids 要移除的向量ids
         * @return {*}
         */        
        void remove_entries_from_partition(size_t list_id, std::vector<faiss::idx_t> ids);

        /**
         * @brief: 从所有分区中移除指定向量
         * @param {unordered_set<faiss::idx_t>} vectors_to_remove 要移除的向量ids
         * @param {unordered_map<faiss::idx_t, int64_t>} ids_map 向量ids到位置的映射
         * @param {unordered_map<int64_t, std::vector<int64_t>>} assignments 包含n_vectors个向量所分配到的分区ID【可选】
         * @return {vector<float>} 要删除的向量拷贝【如果assignment不为nullptr】
         */        
        std::vector<float> remove_vectors(std::unordered_set<faiss::idx_t> vectors_to_remove, 
                                          std::unordered_map<faiss::idx_t, int64_t> ids_map, 
                                          std::unordered_map<int64_t, std::vector<int64_t>>* assignment = nullptr);

        /**
         * @brief: 向指定分区中添加新的向量
         * @param {size_t} list_id 分区ID
         * @param {size_t} n_entries 要添加的向量数量
         * @param {idx_t*} ids 指向新添加向量ids的指针
         * @param {uint8_t*} codes 指向新添加向量编码的指针
         * @return {*} 被添加向量的数量
         */        
        size_t add_entries(size_t list_id, size_t n_entries, const faiss::idx_t* ids, const uint8_t* codes) override;

        /**
         * @brief: 对指定分区从给定偏移位置开始覆写
         * @param {size_t} list_id 分区ID
         * @param {size_t} offset 开始更新的位置
         * @param {size_t} n_entries 要更新的向量数量
         * @param {idx_t*} ids 指向新向量ids的指针
         * @param {uint8_t*} codes 指向新向量编码的指针
         * @return {*}
         */        
        void update_entries(size_t list_id, size_t offset, size_t n_entries, const faiss::idx_t* ids, const uint8_t* codes) override;

        /**
         * @brief: 从旧分区中移动已改变的向量到新的分区
         * @param {size_t} old_partition_id 源分区ID
         * @param {size_t*} new_partition_ids 每个向量所属的新分区IDs
         * @param {uint8_t*} new_codes 指向新向量编码的指针
         * @param {int64_t*} new_ids 指向新向量ids的指针
         * @param {int} n_vectors 要处理的向量数量
         * @return {*}
         */        
        void batch_update_entries(size_t old_partition_id, size_t* new_partition_ids, uint8_t* new_codes, int64_t* new_ids, int n_vectors);

        /**
         * @brief: 移除一整个分区
         * @param {size_t} list_id 分区ID
         * @return {*}
         */        
        void remove_list(size_t list_id);

        /**
         * @brief: 添加一个新的空分区
         * @param {size_t} list_id 分区ID
         * @return {*}
         */        
        void add_list(size_t list_id);

        /**
         * @brief: 检查给定的向量id是否存在与该分区中
         * @param {size_t} list_id 分区ID
         * @param {idx_t*} id 要检查的向量id
         * @return {*} 如果找到返回True，否则返回False
         */        
        bool id_in_list(size_t list_id, faiss::idx_t id) const;

        /**
         * @brief: 通过id检索一向量(将该向量拷贝到指定缓存中)
         * @param {idx_t} id 要检索向量的id
         * @param {float*} vector_values 存储向量的缓存
         * @return {*} 如果找到返回该向量所属分区id，否则返回0
         */        
        int64_t get_vector_for_id(faiss::idx_t id, float* vector_values);

        // void get_vector_for_id_from_partition()

        /**
         * @brief: 通过向量ids检索向量编码指针(无拷贝)
         * @param {vector<int64_t>} ids 要检索的向量ids
         * @return {*} 指向这些向量编码的指针
         */        
        std::vector<float*> get_vector_by_id(std::vector<int64_t> ids);

        /**
         * @brief: 生成并返回一个新分区
         * @return {*} 新分区ID
         */        
        size_t get_new_list_id();

        /**
         * @brief: 重置整个DynamicInvertedList
         * @return {*}
         */        
        void reset() override;

        /**
         * @brief: 重新设置倒排列表大小
         * @param {size_t} nlist 新分区数量
         * @param {size_t} code_size 新向量编码大小
         * @return {*}
         */        
        void resize(size_t nlist, size_t code_size) override;

        /**
         * @brief: 检索分区IDs
         * @return {*} 分区IDs
         */        
        std::vector<int64_t> get_partition_ids();
    };

    /**
     * @brief: 将DynamicInvertedLists对象转换为ArrayInvertedLists对象
     * @param {DynamicInvertedLists*} invlists 指向DynamicInvertedLists对象的指针
     * @param {unordered_map<size_t, size_t>&} remap_ids 输出旧列表号到新列表号的映射
     * @return {*} 指向ArrayInvertedLists对象的指针
     */    
    ArrayInvertedLists* convert_to_array_invlists(DynamicInvertedLists* invlists, std::unordered_map<size_t, size_t>& remap_ids);

    /**
     * @brief: 将ArrayInvertedLists对象转换为DynamicInvertedLists对象。
     * @param {ArrayInvertedLists} *invlists 指向ArrayInvertedLists对象的指针
     * @return {*} 指向DynamicInvertedLists对象的指针
     */
    DynamicInvertedLists *convert_from_array_invlists(ArrayInvertedLists *invlists);
}

#endif // DYNAMIC_INVERTED_LIST_H

