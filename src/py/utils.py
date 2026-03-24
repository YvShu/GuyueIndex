import numpy as np
from pathlib import Path
from typing import List
import csv
import random

def read_ivecs(filename) -> np.ndarray:
    """
    读取一个.ivecs格式的向量文件
    :param filename: 输入文件路径
    :return: numpy.ndarray格式的向量
    """
    buffer = np.fromfile(filename, dtype='int32')
    # a = np.fromfile(filename, dtype='float32')
    d = buffer[0]
    return buffer.reshape(-1, int(d) + 1)[:, 1:].copy()

def read_fvecs(filename) -> np.ndarray:
    """
    读取一个.fvecs格式的向量文件
    :param filename: 输入文件路径
    :return: numpy.ndarray格式的向量
    """
    return read_ivecs(filename).view('float32')

def read_fbin(filename, start_idx=0, chunk_size=None):
    """
    读取一个.fbin格式的向量文件
    :param filename (str): path to *.fbin file
    :param start_idx (int): start reading vectors from this index
    :param chunk_size (int): number of vectors to read. If None, read all vectors
    :return Array of float32 vectors (numpy.ndarray)
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
        nvecs = (nvecs - start_idx) if chunk_size is None else chunk_size
        arr = np.fromfile(f, count=nvecs * dim, dtype=np.float32,
                          offset=start_idx * 4 * dim)
    return arr.reshape(nvecs, dim)

def random_sample_vectors(vectors, num_samples, seed=None):
    """从向量集中随机采样指定数量的向量"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_vectors = vectors.shape[0]
    if num_samples >= n_vectors:
        print(f"Warning: Requested {num_samples} samples but only {n_vectors} available. Using all vectors.")
        return vectors

    # 生成随机索引
    indices = random.sample(range(n_vectors), num_samples)
    return vectors[indices]

def write_fvecs(buffer: np.ndarray, filename):
    """
    写入一个.fvecs格式的向量文件
    :param buffer: 输入numpy文件
    :param filename: 保存文件路径
    """
    num_vectors, d = buffer.shape
    with open(filename, "wb") as f:
        for i in range(num_vectors):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(buffer[i].astype(np.float32).tobytes())

def write_ivecs(buffer: np.ndarray, filename):
    """
    写入一个.ivecs格式的向量文件
    :param buffer: 输入numpy文件
    :param filename: 保存文件路径
    """
    num_vectors, d = buffer.shape
    with open(filename, "wb") as f:
        for i in range(num_vectors):
            f.write(np.array([d], dtype=np.int32).tobytes())
            f.write(buffer[i].astype(np.int32).tobytes())

def write_csv_app(filename: str, data: List[List[str]]) -> None:
    """
    简化的版本，功能相同但更Pythonic
    """
    # 创建目录（如果不存在）
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    # 以追加模式写入CSV
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        csv.writer(file).writerows(data)

def norm_vectors(vectors):
    """对向量进行L2归一化"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # 避免除以零，将零范数的向量保持不变（通常已经是零向量）
    norms[norms == 0] = 1
    return vectors / norms

def compute_recall(ids: np.ndarray, gt_ids: np.ndarray, k: int) -> np.ndarray:
    """
    计算召回率
    :param ids: 搜索返回的ids数组，[num_queries, num_results]
    :param gt_ids: 真实ids数组，[num_queries, num_gt]
    :param k: 考虑的前k个结果
    :return: 每个查询的召回率数组，[num_queries]
    """
    ids = ids[:, :k]
    gt_ids = gt_ids[:, :k]

    num_queries = ids.shape[0]
    recall = np.zeros(num_queries)

    for i in range(num_queries):
        intersection = np.intersect1d(ids[i], gt_ids[i])
        recall[i] = len(intersection) / k

    return recall