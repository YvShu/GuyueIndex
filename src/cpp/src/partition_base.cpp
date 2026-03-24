/*
 * @Author: Guyue
 * @Date: 2025-11-11 14:57:54
 * @LastEditTime: 2026-03-23 10:23:38
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/src/partition_base.cpp
 */
#include <partition_base.h>

PartitionBase::PartitionBase(int64_t num_vectors, uint8_t* codes, faiss::idx_t* ids, int64_t code_size)
{
    buffer_size_ = 0;       // 容量大小
    num_vectors_ = 0;       // 向量数量
    code_size_ = code_size; // 编码大小(一个向量占几个字节)    
    codes_ = nullptr;       // 向量编码指针
    ids_ = nullptr;         // 向量IDs指针
    ensure_capacity(num_vectors);
    append(num_vectors, ids, codes);
}

PartitionBase::PartitionBase(PartitionBase&& other) noexcept
{
    move_from(std::move(other));
}

PartitionBase& PartitionBase::operator=(PartitionBase&& other) noexcept
{
    if (this != &other)
    {
        clear();
        move_from(std::move(other));
    }
    return *this;
}

PartitionBase::~PartitionBase()
{
    clear();
}

void PartitionBase::set_code_size(int64_t code_size)
{
    if (code_size <= 0)
    {
        throw std::runtime_error("[PartitionBase] set_code_size: Invalid code_size.");
    }
    if (num_vectors_ > 0)
    {
        throw std::runtime_error("[PartitionBase] set_code_size: Cannot change code_size when partition has vectors.");
    }
    code_size_ = code_size;
}

void PartitionBase::append(int64_t n_entries, const faiss::idx_t* ids, const uint8_t* codes)
{
    if (n_entries <= 0) return;
    ensure_capacity(num_vectors_ + n_entries);
    const size_t code_bytes = static_cast<size_t>(code_size_);
    /*memcpy(目标地址，起始地址，字节数量)*/
    std::memcpy(codes_ + num_vectors_ * code_bytes, codes, n_entries * code_bytes);
    std::memcpy(ids_ + num_vectors_, ids, n_entries * sizeof(faiss::idx_t));
    num_vectors_ += n_entries;
}

void PartitionBase::update(int64_t offset, int64_t n_entry, const faiss::idx_t* new_ids, const uint8_t* new_codes)
{
    if (n_entry <= 0)
    {
        throw std::runtime_error("[PartitionBase] update: n_entry must be positive in update.");
    }
    if (offset < 0 || offset + n_entry > num_vectors_)
    {
        throw std::runtime_error("[PartitionBase] update: Offset + n_entry out of range in update.");
    }
    const size_t code_bytes = static_cast<size_t>(code_size_);
    std::memcpy(codes_ + offset * code_bytes, new_codes, n_entry * code_bytes);
    std::memcpy(ids_ + offset, new_ids, n_entry * sizeof(faiss::idx_t));
}

void PartitionBase::remove(int64_t index)
{
    if (index < 0 || index >= num_vectors_)
    {
        throw std::runtime_error("[PartitionBase] remove: Index out of range in remove.");
    }
    if (index == num_vectors_ - 1)
    {
        num_vectors_--;
        return;
    }

    int64_t last_idx = num_vectors_ - 1;
    const size_t code_bytes = static_cast<size_t>(code_size_);

    // 将被删移除向量与末尾向量交换位置
    std::memcpy(codes_ + index * code_bytes, codes_ + last_idx * code_bytes, code_bytes);
    ids_[index] = ids_[last_idx];

    num_vectors_--;
}

void PartitionBase::resize(int64_t new_capacity)
{
    if (new_capacity < 0)
    {
        throw std::runtime_error("[PartitionBase] resize: Invalid new_capacity in resize.");
    }
    if (new_capacity < num_vectors_)
    {
        num_vectors_ = new_capacity;
    }
    if (new_capacity != buffer_size_)
    {
        reallocate_memory(new_capacity);
    }
}

void PartitionBase::clear()
{
    free_memory();
    buffer_size_ = 0;
    num_vectors_ = 0;
    code_size_ = 0;
    codes_ = nullptr;
    ids_ = nullptr;
}

int64_t PartitionBase::find_pos_of(faiss::idx_t id) const
{
    for (int64_t i = 0; i < num_vectors_; ++i)
    {
        if (ids_[i] == id)
        {
            return i;
        }
    }
    return -1;
}

void PartitionBase::reallocate_memory(int64_t new_capacity)
{
    if (new_capacity < num_vectors_)
    {
        num_vectors_ = new_capacity;
    }
    const size_t code_bytes = static_cast<size_t>(code_size_);
    int64_t curr_count = num_vectors_;

    uint8_t* new_codes = allocate_memory<uint8_t>(new_capacity * code_bytes);
    faiss::idx_t* new_ids = allocate_memory<faiss::idx_t>(new_capacity);
    if (codes_ && ids_)
    {
        /*memcpy(目标地址，起始地址，字节数量)*/
        std::memcpy(new_codes, codes_, curr_count * code_bytes);
        std::memcpy(new_ids, ids_, curr_count * sizeof(faiss::idx_t));
    }

    free_memory();

    codes_ = new_codes;
    ids_ = new_ids;
    buffer_size_ = new_capacity;  
}

void PartitionBase::move_from(PartitionBase&& other)
{
    buffer_size_ = other.buffer_size_;
    num_vectors_ = other.num_vectors_;
    code_size_ = other.code_size_;
    codes_ = other.codes_;
    ids_ = other.ids_;

    other.codes_ = nullptr;
    other.ids_ = nullptr;
    other.buffer_size_ = 0;
    other.num_vectors_ = 0;
    other.code_size_ = 0;
}

void PartitionBase::free_memory()
{
    if (codes_ == nullptr && ids_ == nullptr)
    {
        return;
    }
    std::free(codes_);
    std::free(ids_);
    codes_ = nullptr;
    ids_ = nullptr;
}

void PartitionBase::ensure_capacity(int64_t required)
{
    if (required > buffer_size_)
    {
        int64_t new_capacity = std::max<int64_t>(1024, buffer_size_);
        while (new_capacity < required)
        {
            new_capacity *= 2;
        }
        reallocate_memory(new_capacity);
    }
}

template<typename T>
T* PartitionBase::allocate_memory(size_t num_elements)
{
    size_t total_bytes = num_elements * sizeof(T);
    T* ptr = nullptr;
    ptr = reinterpret_cast<T*>(std::malloc(total_bytes));
    if (!ptr)
    {
        throw std::bad_alloc();
    }
    return ptr;
}