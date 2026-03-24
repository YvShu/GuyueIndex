/*
 * @Author: Guyue
 * @Date: 2026-03-24 13:37:15
 * @LastEditTime: 2026-03-24 14:14:33
 * @LastEditors: Guyue
 * @FilePath: /GuyueIndex/src/cpp/src/partition_tree.cpp
 */
#include <partition_tree.h>

PartitionTree::PartitionTree() : root_(nullptr)
{
    // nothing to do
}

PartitionTree::PartitionTree(std::vector<float>& centroid)
{
    root_ = new Node;
    root_->centroid_ = centroid;
    root_->parent_ = nullptr;
    root_->ID_ = curr_partition_id_;
    leaves_[curr_partition_id_] = root_;
    curr_partition_id_++;
}

PartitionTree::PartitionTree(std::shared_ptr<Clustering> clustering)
{
    int64_t nlist = clustering->nlist();
    
    if (nlist != 1)
    {
        throw std::runtime_error("[PartitionTree] PartitionTree: nlist!=1, Init must with 1 partition!");
    }

    // 初始化根节点
    root_ = new Node;
    root_->centroid_ = clustering->centroids;
    root_->parent_ = nullptr;

    // 初始化子节点[0,1,2,...nlist-1]
    auto& centroids = clustering->vectors[0];
    auto& IDs = clustering->vector_ids[0];
    int dim = clustering->dimension;
    int64_t c_size = IDs.size();
    for (int64_t i = 0; i < c_size; ++i)
    {
        Node* temp = new Node;
        for (int64_t j = 0; j < dim; ++j)
        {
            temp->centroid_.push_back(centroids[i * dim + j]);
        }
        temp->parent_ = root_;
        temp->ID_ = curr_partition_id_;
        root_->children_.push_back(temp);
        leaves_[curr_partition_id_] = temp;
        curr_partition_id_++;
    }
}

PartitionTree::~PartitionTree()
{
    delete root_;
    leaves_.clear();
}

void PartitionTree::build(Node* root, const std::vector<float>& vectors, const std::vector<int64_t>& ids, int n_clusters)
{
    // TODO
}

void PartitionTree::split(const std::vector<int64_t>& IDs, std::vector<float>& centroids, const std::vector<int64_t>& assignment)
{
    int dim = centroids.size() / assignment.size();
    for (int64_t i = 0; i < assignment.size(); ++i)
    {
        auto it = leaves_.find(assignment[i]);
        if (it == leaves_.end())
        {
            throw std::runtime_error("[PartitionTree] split: List does not exist in leaves.");
        }

        Node* temp = new Node;
        for (int64_t j = 0; j < dim; ++j)
        {
            temp->centroid_.push_back(centroids[i * dim + j]);
        }
        temp->parent_ = leaves_[assignment[i]];
        temp->ID_ = curr_partition_id_;
        leaves_[assignment[i]]->children_.push_back(temp);
        leaves_[curr_partition_id_] = temp;
        curr_partition_id_++;
    }

    // 从leaves_中移除节点
    for (int64_t ID : IDs)
    {
        auto it = leaves_.find(ID);
        if (it == leaves_.end())
        {
            throw std::runtime_error("[PartitionTree] split : List does not exist in leaves.");
        }
        leaves_.erase(it);
    }
}

void PartitionTree::shrink(const std::vector<int64_t>& IDs)
{
    for (auto ID : IDs)
    {
        if (!leaves_[ID]->parent_)
        {
            continue;
        }

        Node* node = leaves_[ID];
        Node* parent = node->parent_;

        auto it = std::find(parent->children_.begin(), parent->children_.end(), node);
        if (it != parent->children_.end())
        {
            parent->children_.erase(it);
            node->parent_ = nullptr;
            delete node;
        }
        leaves_.erase(ID);
    }

    for (const auto& pair : leaves_)
    {
        Node* parent = pair.second->parent_;
        if (parent != nullptr && parent->children_.size() == 1)
        {
            Node* grand_parent = parent->parent_;
            if (grand_parent != nullptr)
            {
                auto it = std::find(grand_parent->children_.begin(), grand_parent->children_.end(), parent);
                if (it != grand_parent->children_.end())
                {
                    grand_parent->children_.erase(it);
                    grand_parent->children_.push_back(pair.second);
                    pair.second->parent_ = grand_parent;
                }
            }
            parent->children_.clear();
            delete parent;
        }
    }
}

Node* PartitionTree::root() const
{
    return root_;
}
