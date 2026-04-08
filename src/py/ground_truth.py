import faiss
import json
import utils
import os
import numpy as np


def compute_ground_truth(base_vectors, query_vectors, id_mapping, metric, k=100):
    d = base_vectors.shape[1]
    index: faiss.Index = None

    if metric == 'l2':
        index = faiss.IndexFlatL2(d)
    elif metric == 'ip':
        index = faiss.IndexFlatIP(d)

    index.add(base_vectors)
    D, I = index.search(query_vectors, k)

    # 将 Faiss 索引位置映射回原始向量 ID
    ground_truth = np.zeros_like(I)
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            if I[i, j] >= 0:  # 有效的索引
                ground_truth[i, j] = id_mapping[I[i, j]]
            else:  # 无效索引（通常为 -1）
                ground_truth[i, j] = -1

    return ground_truth

# def brute_force_topk_l2(base_vectors, query_vectors, id_mapping, metric, k=100):
#     N = base_vectors.shape[0]
#     Q = query_vectors.shape[0]
#     # 输出
#     out_ids = np.zeros((Q, k), dtype=np.int64)
#
#     for qi in range(Q):
#         q = query_vectors[qi]
#         if metric == "l2":
#             # 计算 L2 距离: ||x - q||^2
#             dist = np.sum((base_vectors - q) ** 2, axis=1)
#         elif metric == "ip":
#             # 内积越大越相似 → 取负数转成“距离”
#             dist = -np.dot(base_vectors, q)
#         else:
#             raise ValueError("Unsupported metric: use 'l2' or 'ip'")
#         # ==== partial sort: 取出前 k 小 ====
#         topk_idx = np.argpartition(dist, k)[:k]  # 无序 top-k 子集
#         # 重排序
#         sorted_k = topk_idx[np.argsort(dist[topk_idx])]
#         # 映射成原始 id
#         out_ids[qi] = id_mapping[sorted_k]
#     return out_ids

def process_workload(input_data_file, input_runbook_file, output_dir, metric):
    vectors = utils.read_fvecs(input_data_file)
    runbook = json.load(open(input_runbook_file, "r"))

    # 当前数据库状态：已插入但未删除的向量
    current_ids = set()
    search_count = 0

    # --- Run Operations ---
    for op in runbook["operations"]:
        if op["operation"] == "build":
            print("build")
            start = op["start"]
            end = op["end"]
            for id in range(start, end+1):
                current_ids.add(id)
        elif op["operation"] == "insert":
            print("insert")
            start = op["start"]
            end = op["end"]
            for id in range(start, end+1):
                current_ids.add(id)
        elif op["operation"] == "delete":
            print("delete")
            start = op["start"]
            end = op["end"]
            for id in range(start, end+1):
                current_ids.remove(id)
        elif op["operation"] == "search":
            print(f"search {len(current_ids)}")
            queries = utils.read_fvecs(op["query_file"])
            print(queries.shape)
            print(vectors.shape)
            if len(current_ids) > 0:
                sorted_ids = sorted(current_ids)
                current_database = vectors[sorted_ids]
                id_mapping = {i:sorted_ids[i] for i in range(len(sorted_ids))}
                ground_truth = compute_ground_truth(current_database, queries, id_mapping, metric)
                # id_mapping = np.array(sorted_ids, dtype=np.int64)
                # ground_truth = brute_force_topk_l2(current_database, queries, id_mapping, metric)
                output_file = os.path.join(output_dir, f"gt_step_{search_count}.ivecs")
                utils.write_ivecs(ground_truth, output_file)
                print(f"Written ivecs to {output_file}")
            else:
                print(f"No ivecs found for {op}")
            search_count += 1

if __name__ == '__main__':
    dataset = "sift-2M"
    workload = "workload2"
    metric = "l2"
    input_data_file = f"/mnt/hgfs/DataSet/{dataset}/{dataset.split('-')[0]}_base.fvecs"
    # input_data_file = fr"/mnt/hgfs/DataSet/{dataset}/{workload}/{dataset.split('-')[0]}_base.fvecs"
    # input_runbook_file = fr"/mnt/hgfs/DataSet/{dataset}/runbook.json"
    input_runbook_file = fr"/mnt/hgfs/DataSet/{dataset}/{workload}/runbook.json"
    output_dir = os.path.dirname(input_runbook_file)

    process_workload(input_data_file, input_runbook_file, output_dir, metric)