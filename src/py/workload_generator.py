import numpy as np
import argparse
import random
import utils
import faiss
import json
import os
from typing import List, Tuple, Dict, Set


class OptimizedWorkloadGenerator:

    def __init__(
            self,
            input_data_file,
            input_query_file,
            output_data_file,
            output_runbook_file,
            s_0,
            s_u,
            num_queries,
            r_id,
            CSF_u,
            r_rw,
            CSF_q,
            num_clusters,
    ):
        self.input_data_file = input_data_file
        self.input_query_file = input_query_file
        self.output_data_file = output_data_file
        self.output_runbook_file = output_runbook_file
        self.s_0 = s_0
        self.s_u = s_u
        self.r_id = r_id
        self.CSF_u = CSF_u
        self.r_rw = r_rw
        self.CSF_q = CSF_q
        self.num_clusters = num_clusters

        # 使用numpy数组替代Python列表
        self.available_flags = None  # 布尔数组，标记向量是否可用
        self.vector_indices = None  # 所有向量的索引
        self.cluster_vectors = []  # 每个聚类的向量索引列表
        self.cluster_sizes = None  # 每个聚类的向量数量
        self.cluster_available_counts = None  # 每个聚类中可用的向量数量

        self.num_queries = 0
        if input_query_file:
            self.num_queries = utils.read_fvecs(input_query_file).shape[0]
        else:
            self.num_queries = num_queries

        self.current_id = 0
        self.inserted_idx = []  # 仍然需要，但只用于最终排序输出

    def generate_workload(self):
        """生成工作负载"""
        vectors = utils.read_fvecs(self.input_data_file)
        n_vectors = vectors.shape[0]
        print(f"vectors: {vectors.shape}")

        # 初始化数据结构
        self.vector_indices = np.arange(n_vectors, dtype=np.int32)
        self.available_flags = np.ones(n_vectors, dtype=bool)

        # --- 聚类向量 ---
        km = faiss.Kmeans(
            d=vectors.shape[1],
            k=self.num_clusters,
            niter=25,
            seed=42
        )
        km.train(vectors)
        _, labels = km.assign(vectors)
        print("clustering done")

        # --- 构建聚类信息（使用numpy数组）---
        self.cluster_vectors = [np.array([], dtype=np.int32) for _ in range(self.num_clusters)]
        cluster_lists = [[] for _ in range(self.num_clusters)]

        # 收集每个聚类的索引
        for i in range(n_vectors):
            cluster_lists[labels[i]].append(i)

        # 转换为numpy数组
        for cluster_id in range(self.num_clusters):
            self.cluster_vectors[cluster_id] = np.array(cluster_lists[cluster_id], dtype=np.int32)

        self.cluster_sizes = np.array([len(arr) for arr in self.cluster_vectors], dtype=np.int32)
        self.cluster_available_counts = self.cluster_sizes.copy()
        print("build clusters info done")

        # --- 选择初始向量 ---
        initial_indices = self.select_initial_vectors()

        # 更新可用标记
        self.available_flags[initial_indices] = False
        # 更新聚类可用计数
        for idx in initial_indices:
            cluster_id = labels[idx]
            self.cluster_available_counts[cluster_id] -= 1

        self.current_id = len(initial_indices)
        self.inserted_idx.extend(initial_indices)

        # --- 计算操作数量 ---
        total_vectors_to_insert = n_vectors - len(initial_indices)
        num_insert_ops = (total_vectors_to_insert + self.s_u - 1) // self.s_u
        num_delete_ops = int(num_insert_ops * (1 - self.r_id) // self.r_id) if self.r_id >= 0.5 else 0
        total_update = num_insert_ops + num_delete_ops
        num_search_ops = int(total_update * (1 - self.r_rw) // self.r_rw) if self.r_id >= 0 else 0
        print("select initial vectors done")

        # --- 生成操作 ---
        operations = self.generate_operations(
            vectors, labels, initial_indices,
            num_insert_ops, num_delete_ops, num_search_ops
        )
        print("generate operations done")

        # --- 保存排序后的向量 ---
        sorted_vectors = vectors[self.inserted_idx]
        utils.write_fvecs(sorted_vectors, self.output_data_file)
        print("save done")

        # --- 生成runbook ---
        self.create_runbook(operations, vectors, num_insert_ops, num_delete_ops, num_search_ops)
        print("create runbook done")

    def select_initial_vectors(self) -> np.ndarray:
        """选择初始向量"""
        n_vectors = len(self.vector_indices)
        initial_indices = np.array([], dtype=np.int32)

        # 每个聚类的目标数量
        vectors_per_cluster = max(1, self.s_0 // self.num_clusters)

        for cluster_id in range(self.num_clusters):
            cluster_vecs = self.cluster_vectors[cluster_id]
            n_cluster = len(cluster_vecs)

            if n_cluster <= vectors_per_cluster:
                selected = cluster_vecs
            else:
                # 随机选择
                selected = np.random.choice(cluster_vecs, vectors_per_cluster, replace=False)

            initial_indices = np.concatenate([initial_indices, selected])

        # 如果数量不足，从剩余向量中补充
        if len(initial_indices) < self.s_0:
            remaining = self.s_0 - len(initial_indices)
            # 获取所有未选中的向量
            all_indices_set = set(self.vector_indices)
            selected_set = set(initial_indices)
            available_set = all_indices_set - selected_set

            if available_set:
                additional = np.random.choice(list(available_set),
                                              min(remaining, len(available_set)),
                                              replace=False)
                initial_indices = np.concatenate([initial_indices, additional])

        # 如果数量超过，随机减少
        if len(initial_indices) > self.s_0:
            initial_indices = np.random.choice(initial_indices, self.s_0, replace=False)

        return initial_indices

    def generate_operations(self, vectors, labels, initial_indices,
                            num_insert_ops, num_delete_ops, num_search_ops):
        """生成操作序列"""
        operations = []

        # 构建初始操作
        operation = {
            "operation": "build",
            "start": 0,
            "end": len(initial_indices) - 1,
        }
        operations.append(operation)
        print(operation)

        # 使用优化后的操作序列生成算法，确保操作均匀分散
        operation_types = self.generate_balanced_operation_sequence(
            num_insert_ops, num_delete_ops, num_search_ops
        )

        # 用于跟踪删除区间
        deleted_intervals = []  # 存储(start, end)元组

        # 处理每个操作
        for op_type in operation_types:
            if op_type == "insert":
                batch = self.generate_insert_batch(labels)
                if batch is not None and len(batch) > 0:
                    operation = {
                        "operation": "insert",
                        "start": self.current_id,
                        "end": self.current_id + len(batch) - 1,
                    }
                    operations.append(operation)
                    print(operation)

                    # 更新状态
                    self.current_id += len(batch)
                    self.inserted_idx.extend(batch)

                    # 更新可用标记
                    self.available_flags[batch] = False

                    # 更新聚类可用计数
                    for idx in batch:
                        cluster_id = labels[idx]
                        self.cluster_available_counts[cluster_id] -= 1

            elif op_type == "delete":
                if len(self.inserted_idx) == 0:
                    continue

                delete_start, delete_end = self.generate_delete_batch(deleted_intervals)
                if delete_start is not None:
                    operation = {
                        "operation": "delete",
                        "start": delete_start,
                        "end": delete_end,
                    }
                    operations.append(operation)
                    print(operation)
                    deleted_intervals.append((delete_start, delete_end))

            elif op_type == "search":
                if self.input_query_file is not None:
                    operation = {
                        "operation": "search",
                        "num_queries": self.num_queries,
                        "query_file": self.input_query_file,
                    }
                    operations.append(operation)
                    print(operation)

        return operations

    def generate_balanced_operation_sequence(self, num_insert_ops, num_delete_ops, num_search_ops):
        """生成平衡的操作序列，确保三种操作均匀分散"""
        operation_types = []

        # 计算总操作数和每种操作的比例
        total_ops = num_insert_ops + num_delete_ops + num_search_ops
        if total_ops == 0:
            return []

        # 计算目标频率：我们希望操作尽可能均匀分布
        insert_freq = total_ops / num_insert_ops if num_insert_ops > 0 else float('inf')
        delete_freq = total_ops / num_delete_ops if num_delete_ops > 0 else float('inf')
        search_freq = total_ops / num_search_ops if num_search_ops > 0 else float('inf')

        # 使用计数器来跟踪每种操作的"理想"出现位置
        insert_counter = 0
        delete_counter = 0
        search_counter = 0

        # 记录剩余操作数量
        insert_remaining = num_insert_ops
        delete_remaining = num_delete_ops
        search_remaining = num_search_ops

        # 生成操作序列
        while insert_remaining > 0 or delete_remaining > 0 or search_remaining > 0:
            # 计算每种操作的"优先级"（距离理想位置的差距）
            insert_priority = insert_counter
            delete_priority = delete_counter
            search_priority = search_counter

            # 创建候选操作列表
            candidates = []
            if insert_remaining > 0:
                candidates.append(("insert", insert_priority))
            if delete_remaining > 0:
                candidates.append(("delete", delete_priority))
            if search_remaining > 0:
                candidates.append(("search", search_priority))

            # 选择优先级最高（最应该出现）的操作
            candidates.sort(key=lambda x: x[1])
            selected_op = candidates[0][0]

            # 添加选中的操作
            operation_types.append(selected_op)

            # 更新计数器和剩余数量
            if selected_op == "insert":
                insert_counter += insert_freq
                insert_remaining -= 1
            elif selected_op == "delete":
                delete_counter += delete_freq
                delete_remaining -= 1
            elif selected_op == "search":
                search_counter += search_freq
                search_remaining -= 1

        return operation_types

    def generate_insert_batch(self, labels) -> np.ndarray:
        """生成插入批次"""
        # 计算批次大小
        actual_batch_size = min(self.s_u, self.available_flags.sum())
        if actual_batch_size == 0:
            return np.array([], dtype=np.int32)

        # 计算从主聚类抽取的数量
        num_from_main = int(actual_batch_size * self.CSF_u)
        num_from_other = actual_batch_size - num_from_main

        selected_indices = []

        # 随机打乱聚类顺序
        cluster_order = list(range(self.num_clusters))
        random.shuffle(cluster_order)

        # 从主聚类抽取
        for cluster_id in cluster_order:
            if num_from_main <= 0:
                break

            # 获取该聚类中可用的向量
            cluster_vecs = self.cluster_vectors[cluster_id]
            available_in_cluster = cluster_vecs[self.available_flags[cluster_vecs]]

            if len(available_in_cluster) > 0:
                n_to_take = min(num_from_main, len(available_in_cluster))
                selected = np.random.choice(available_in_cluster, n_to_take, replace=False)
                selected_indices.extend(selected)
                num_from_main -= n_to_take

        # 从其他向量抽取（如果还需要）
        if num_from_other > 0:
            # 获取所有可用的向量
            all_available = self.vector_indices[self.available_flags]
            # 排除已选择的
            if selected_indices:
                selected_set = set(selected_indices)
                all_available = all_available[~np.isin(all_available, list(selected_set))]

            if len(all_available) > 0:
                n_to_take = min(num_from_other, len(all_available))
                additional = np.random.choice(all_available, n_to_take, replace=False)
                selected_indices.extend(additional)

        return np.array(selected_indices, dtype=np.int32)

    def generate_delete_batch(self, deleted_intervals) -> Tuple[int, int]:
        """生成删除批次"""
        if len(self.inserted_idx) == 0:
            return None, None

        actual_delete_size = min(self.s_u, len(self.inserted_idx))

        # 获取可用的连续区间
        available_intervals = self.get_available_intervals(deleted_intervals)
        if not available_intervals:
            return None, None

        # 选择足够大的区间
        valid_intervals = [(s, e) for s, e in available_intervals
                           if (e - s + 1) >= actual_delete_size]

        if not valid_intervals:
            return None, None

        # 随机选择一个区间
        selected_interval = random.choice(valid_intervals)
        start, end = selected_interval

        # 在区间内随机选择起始点
        max_start = end - actual_delete_size + 1
        delete_start = random.randint(start, max_start)
        delete_end = delete_start + actual_delete_size - 1

        return delete_start, delete_end

    def get_available_intervals(self, deleted_intervals):
        """获取可用的连续区间"""
        if not deleted_intervals:
            return [(0, self.current_id - 1)]

        intervals = []
        current_start = 0

        # 对删除区间排序
        sorted_deleted = sorted(deleted_intervals, key=lambda x: x[0])

        for del_start, del_end in sorted_deleted:
            if current_start < del_start:
                intervals.append((current_start, del_start - 1))
            current_start = del_end + 1

        if current_start < self.current_id:
            intervals.append((current_start, self.current_id - 1))

        return intervals

    def create_runbook(self, operations, vectors, num_insert_ops, num_delete_ops, num_search_ops):
        """创建runbook文件"""
        datainfo = {
            "data": self.input_data_file,
            "total_vectors": vectors.shape[0],
            "dimension": vectors.shape[1],
        }
        parameters = {
            "s_0": self.s_0,
            "s_u": self.s_u,
            "r_id": self.r_id,
            "CSF_u": self.CSF_u,
            "r_rw": self.r_rw,
            "CSF_q": self.CSF_q,
            "num_clusters": self.num_clusters,
            "num_insert_ops": num_insert_ops,
            "num_delete_ops": num_delete_ops,
            "num_search_ops": num_search_ops,
        }

        runbook = {
            "datainfo": datainfo,
            "parameters": parameters,
            "operations": operations,
        }

        with open(self.output_runbook_file, 'w', encoding='utf-8') as f:
            json.dump(runbook, f, indent=2, ensure_ascii=False)


def main():
    dataset = "sift-1M"
    workload = "workload1"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_data_file", type=str, help="input database file(.fevcs)",
                        default=fr"/mnt/hgfs/DataSet/{dataset}/{dataset.split('-')[0]}_base.fvecs")
    parser.add_argument("--input_query_file", type=str, help="input query file(.fevcs)",
                        default=fr"/mnt/hgfs/DataSet/{dataset}/{dataset.split('-')[0]}_query.fvecs")
    parser.add_argument("--output_data_file", type=str, help="output database file(.fevcs)",
                        default=fr"/mnt/hgfs/DataSet/{dataset}/{workload}/{dataset.split('-')[0]}_base.fvecs")
    parser.add_argument("--output_query_file", type=str, help="output query file(.fevcs)")
    parser.add_argument("--output_runbook_file", type=str, help="output runbook file(.json)",
                        default=fr"/mnt/hgfs/DataSet/{dataset}/{workload}/runbook.json")
    parser.add_argument("--s_0", type=int, help="initial vectors size", default=0)
    parser.add_argument("--s_u", type=int, help="update vectors size", default=50000)
    parser.add_argument("--num_queries", type=int, help="queries size")
    parser.add_argument("--r_id", type=float, help="insert/delete ratio", default=1.0)
    parser.add_argument("--CSF_u", type=float, help="cluster sampling ratio of update", default=0.5)
    parser.add_argument("--r_rw", type=float, help="read/write ratio", default=0.5)
    parser.add_argument("--CSF_q", type=float, help="cluster sampling ratio of query")
    parser.add_argument("--num_clusters", type=int, help="the number of clusters", default=25)
    args = parser.parse_args()

    directory = os.path.dirname(args.output_data_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    directory = os.path.dirname(args.output_runbook_file)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    workload_generator = OptimizedWorkloadGenerator(
        args.input_data_file,
        args.input_query_file,
        args.output_data_file,
        args.output_runbook_file,
        args.s_0,
        args.s_u,
        args.num_queries,
        args.r_id,
        args.CSF_u,
        args.r_rw,
        args.CSF_q,
        args.num_clusters)
    workload_generator.generate_workload()

if __name__ == '__main__':
    main()