/*
 * @Author: Guyue
 * @Date: 2026-03-23 10:36:54
 * @LastEditTime: 2026-04-03 16:42:51
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/src/dynamic_inverted_lists.cpp
 */
#include <dynamic_inverted_lists.h>

namespace faiss {
    ArrayInvertedLists* convert_to_array_invlists(DynamicInvertedLists* invlists, std::unordered_map<size_t, size_t>& remap_ids)
    {
        auto ret = new ArrayInvertedLists(invlists->nlist, invlists->code_size);
        
        // 迭代遍历所有分区
        size_t new_list_no = 0;
        for (auto &p : invlists->partitions_)
        {
            size_t old_list_no = p.first;
            std::shared_ptr<PartitionBase> part = p.second;

            if (part->num_vectors_ > 0)
            {
                ret->add_entries(new_list_no, part->num_vectors_, part->ids_, part->codes_);
            }
            remap_ids[old_list_no] = new_list_no;
            new_list_no += 1;
        }
        return ret;
    }

    DynamicInvertedLists* convert_from_array_invlists(ArrayInvertedLists* invlist)
    {
        auto ret = new DynamicInvertedLists(invlist->nlist, invlist->code_size);
        for (size_t list_no = 0; list_no < invlist->nlist; list_no++)
        {
            size_t list_size = invlist->list_size(list_no);
            if (list_size > 0)
            {
                ret->add_entries(list_no, list_size, invlist->get_ids(list_no), invlist->get_codes(list_no));
            } else {
                ret->add_list(list_no);
            }
        }
        return ret;
    }

    DynamicInvertedLists::DynamicInvertedLists(size_t nlist, size_t code_size)
        : InvertedLists(nlist, code_size)
    {
        dimension_ = code_size / sizeof(float);
        code_size_ = code_size;
        // --- 初始化空分区 ---
        for (size_t i = 0; i < nlist; ++i)
        {
            std::shared_ptr<PartitionBase> part = std::make_shared<PartitionBase>();
            part->set_code_size(code_size);
            partitions_[i] = part;
        }
        curr_list_id_ = nlist;
    }

    DynamicInvertedLists::~DynamicInvertedLists()
    {
        // partitions_会自动清理当PartitionBase的析构函数释放内存时
    }

    size_t DynamicInvertedLists::ntotal() const
    {
        size_t ntotal = 0;
        for (auto& kv : partitions_)
        {
            ntotal += kv.second->num_vectors_;
        }
        return ntotal;
    }

    size_t DynamicInvertedLists::list_size(size_t list_id) const
    {
        auto it = partitions_.find(list_id);
        if (it == partitions_.end())
        {
            throw std::runtime_error("[DynamicInvertedLists] list_size : List does not exist.");
        }
        return static_cast<size_t>(it->second->num_vectors_);
    }

    const uint8_t* DynamicInvertedLists::get_codes(size_t list_id) const
    {
        auto it = partitions_.find(list_id);
        if (it == partitions_.end())
        {
            throw std::runtime_error("[DynamicInvertedLists] get_codes : List does not exist.");
        }
        return it->second->codes_;
    }

    const faiss::idx_t* DynamicInvertedLists::get_ids(size_t list_id) const
    {
        auto it = partitions_.find(list_id);
        if (it == partitions_.end())
        {
            throw std::runtime_error("[DynamicInvertedList] get_ids : List does not exist.");
        }
        return it->second->ids_;
    }

    void DynamicInvertedLists::release_codes(size_t list_id, const uint8_t* codes) const
    {
        // 无需操作，get_codes不分配新内存
    }

    void DynamicInvertedLists::release_ids(size_t list_id, const faiss::idx_t* ids) const
    {
        // 无需操作，get_ids不分配新内存
    }

    void DynamicInvertedLists::remove_entry(size_t list_id, faiss::idx_t id)
    {
        auto it = partitions_.find(list_id);
        if (it == partitions_.end())
        {
            throw std::runtime_error("[DynamicInvertedLists] remove_entry : List does not exist.");
        }
        
        std::shared_ptr<PartitionBase> part = it->second;
        if (part->num_vectors_ == 0) return;

        int64_t pos_to_remove = part->find_pos_of(id);
        if (pos_to_remove != -1)
        {
            part->remove(pos_to_remove);
        }
    }

    void DynamicInvertedLists::remove_entries_from_partition(size_t list_id, std::vector<faiss::idx_t> ids)
    {
        auto it = partitions_.find(list_id);
        if (it == partitions_.end())
        {
            throw std::runtime_error("[DynamicInvertedLists] remove_entries_from_partition : List does not exist.");
        }

        std::shared_ptr<PartitionBase> part = it->second;
        std::set<faiss::idx_t> vectors_to_remove_set(ids.begin(), ids.end());

        for (int64_t i = 0; i < part->num_vectors_;)
        {
            if (vectors_to_remove_set.find(part->ids_[i]) != vectors_to_remove_set.end())
            {
                part->remove(i);
            } else {
                i++;
            }
        }
    }

    std::vector<float> DynamicInvertedLists::remove_vectors(std::unordered_set<faiss::idx_t> vectors_to_remove, 
                                                            std::unordered_map<faiss::idx_t, int64_t> ids_map, 
                                                            std::unordered_map<int64_t, std::vector<int64_t>>* assignment)
    {
        // for (auto& kv : partitions_)
        // {
        //     std::shared_ptr<PartitionBase> part = kv.second;
        //     for (int64_t i = 0; i < part->num_vectors_;)
        //     {
        //         if (vectors_to_remove.find(part->ids_[i]) != vectors_to_remove.end())
        //         {
        //             part->remove(i);
        //         } else {
        //             i++;
        //         }
        //     }
        // }
        
        std::vector<float> vectors;
        std::vector<int64_t> vectors_pid;
        if (assignment)
        {
            vectors.resize(vectors_to_remove.size() * dimension_);
            vectors_pid.resize(vectors_to_remove.size());
        }

        std::vector<int64_t> partitions_ids = get_partition_ids();
    #pragma omp parallel for schedule(dynamic)
        for (int64_t p = 0; p < partitions_ids.size(); ++p)
        {
            int64_t p_id = partitions_ids[p];
            int64_t i = 0;
            std::shared_ptr<PartitionBase> part = partitions_[p_id];
            std::vector<int64_t> vectors_ids;
            while (i < part->num_vectors_)
            {
                if (vectors_to_remove.find(part->ids_[i]) != vectors_to_remove.end())
                {
                    if (assignment)
                    {
                        vectors_pid[ids_map[part->ids_[i]]] = p_id;
                        if (part->code_size_ == dimension_ * sizeof(float))
                        {
                            std::memcpy(vectors.data() + ids_map[part->ids_[i]] * dimension_,
                                        part->codes_ + i * part->code_size_,
                                        part->code_size_);
                        } else {
                            guyue::decode(part->codes_ + i * part->code_size_, 
                                          vectors.data() + ids_map[part->ids_[i]] * dimension_, 
                                          dimension_);
                        }
                    }
                    part->remove(i);
                } else {
                    i++;
                }
            }
        }

        if (assignment)
        {
            for (int64_t i = 0; i < vectors_pid.size(); ++i)
            {
                (*assignment)[vectors_pid[i]].push_back(i);
            }
        }

        return vectors;
    }

    size_t DynamicInvertedLists::add_entries(size_t list_id, size_t n_entries, const faiss::idx_t* ids, const uint8_t* codes)
    {
        if (n_entries == 0) 
        {
            return 0;
        }

        auto it = partitions_.find(list_id);
        if (it == partitions_.end())
        {
            throw std::runtime_error("[DynamicInvertedLists] add_entries_from_partition : List does not exist.");
        }

        std::shared_ptr<PartitionBase> part = it->second;
        if (part->code_size_ != static_cast<int64_t>(code_size))
        {
            part->set_code_size(static_cast<int64_t>(code_size));
        }

        part->append((int64_t) n_entries, ids, codes);
        return n_entries;
    }

    void DynamicInvertedLists::update_entries(size_t list_id, size_t offset, size_t n_entries, const faiss::idx_t* ids, const uint8_t* codes)
    {
        auto it = partitions_.find(list_id);
        if (it == partitions_.end())
        {
            throw std::runtime_error("[DynamicInvertedLists] update_entries_from_partition : List does not exist.");
        }
        std::shared_ptr<PartitionBase> part = it->second;

        part->update((int64_t) offset, (int64_t) n_entries, ids, codes);
    }

    void DynamicInvertedLists::batch_update_entries(size_t old_partition_id, size_t* new_partition_ids, uint8_t* new_codes, int64_t* new_ids, int n_vectors)
    {
        // --- 汇总移动向量的目标分区 ---
        std::unordered_map<size_t, std::vector<int>> vectors_for_new_partition;
        for (int i = 0; i < n_vectors; ++i)
        {
            size_t new_part_id = static_cast<size_t>(new_partition_ids[i]);
            if (new_part_id != old_partition_id)
            {
                vectors_for_new_partition[new_part_id].push_back(i);
            }
        }

        // --- 添加向量到新的分区 ---
        for (auto& kv : vectors_for_new_partition)
        {
            size_t new_part_id = kv.first;
            auto it = partitions_.find(new_part_id);
            if (it == partitions_.end())
            {
                add_list(new_part_id);
                it = partitions_.find(new_part_id);
            }
            std::shared_ptr<PartitionBase> new_part = it->second;
            if (new_part->code_size_ != static_cast<int64_t>(code_size))
            {
                new_part->set_code_size((int64_t) code_size);
            }

            std::vector<faiss::idx_t> tmp_ids;
            tmp_ids.reserve(kv.second.size());
            std::vector<uint8_t> tmp_codes;
            tmp_codes.reserve(kv.second.size() * code_size);
            for (int id : kv.second)
            {
                tmp_ids.push_back((faiss::idx_t) new_ids[id]);
                tmp_codes.insert(tmp_codes.end(), 
                                 new_codes + id * code_size,
                                 new_codes + (id + 1) * code_size);
            }
            new_part->append((int64_t) kv.second.size(), tmp_ids.data(), tmp_codes.data());
        }

        // --- 从旧分区中移除 ---
        auto old_it = partitions_.find(old_partition_id);
        if (old_it != partitions_.end())
        {
            std::shared_ptr<PartitionBase> old_part = old_it->second;
            for (auto& kv : vectors_for_new_partition)
            {
                for (int id : kv.second)
                {
                    faiss::idx_t old_id = (faiss::idx_t) new_ids[id];
                    int64_t pos = old_part->find_pos_of(old_id);
                    if (pos != -1)
                    {
                        old_part->remove(pos);
                    }
                }
            }
        }
    }

    void DynamicInvertedLists::remove_list(size_t list_id)
    {
        auto it = partitions_.find(list_id);
        if (it == partitions_.end())
        {
            return;
        }
        partitions_.erase(it);
        nlist--;
    }

    void DynamicInvertedLists::add_list(size_t list_id)
    {
        if (partitions_.find(list_id) != partitions_.end())
        {
            throw std::runtime_error("[DynamicInvertedListd] add_list : List already exist.");
        }
        std::shared_ptr<PartitionBase> part = std::make_shared<PartitionBase>();
        part->set_code_size((int64_t) code_size);
        partitions_[list_id] = part;
        nlist++;
    }

    bool DynamicInvertedLists::id_in_list(size_t list_id, faiss::idx_t id) const
    {
        auto it = partitions_.find(list_id);
        if (it == partitions_.end())
        {
            return false;
        }
        std::shared_ptr<PartitionBase> part = it->second;
        return part->find_pos_of(id) != -1;
    }

    int64_t DynamicInvertedLists::get_vector_for_id(faiss::idx_t id, float* vector_values)
    {
        for (auto& kv : partitions_)
        {
            std::shared_ptr<PartitionBase> part = kv.second;
            int64_t pos = part->find_pos_of(id);
            if (pos != -1)
            {
                std::memcpy(vector_values, 
                            part->codes_ + pos * part->code_size_,
                            part->code_size_);
                return kv.first;
            }
        }
        return 0;
    }

    std::vector<float*> DynamicInvertedLists::get_vector_by_id(std::vector<int64_t> ids)
    {
        std::vector<float*> ret;
        for (int64_t id : ids)
        {
            bool found = false;
            for (auto& kv : partitions_)
            {
                std::shared_ptr<PartitionBase> part = kv.second;
                int64_t pos = part->find_pos_of(id);
                if (pos != -1)
                {
                    ret.push_back(reinterpret_cast<float*>(part->codes_ + pos * part->code_size_));
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                throw std::runtime_error("[DynamicInvertedLists] get_vector_by_id : ID not found in any partition.");
            }
        }
        return ret;
    }

    size_t DynamicInvertedLists::get_new_list_id()
    {
        return curr_list_id_++;
    }

    void DynamicInvertedLists::reset()
    {
        partitions_.clear();
        nlist = 0;
        curr_list_id_ = 0;
    }

    void DynamicInvertedLists::resize(size_t nlist, size_t code_size)
    {
        // 使用map进行管理，可以直接添加或删除
    }

    std::vector<int64_t> DynamicInvertedLists::get_partition_ids()
    {
        std::vector<int64_t> result(partitions_.size(), 0);
        size_t i = 0;
        for (auto& kv : partitions_)
        {
            result[i] = static_cast<int64_t>(kv.first);
            i++;
        }
        return result;
    }
} // namespace faiss