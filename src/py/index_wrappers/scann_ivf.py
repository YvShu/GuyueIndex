import scann
import numpy as np
import re
import time

from src.py.index_wrappers.wrapper import IndexWrapper


class Scann(IndexWrapper):
    """
    scann索引
    """
    index: scann.scann_ops_pybind.ScannSearcher

    def __init__(self):
        self.index = None

    def n_total(self) -> int:
        return self.index.size()

    def d(self) -> int:
        config = self.index.config()
        match = re.search(r"input_dim:\s*(\d+)", config)
        return int(match.group(1))

    def build(
        self,
        vectors: np.ndarray,
        ids: np.ndarray = None,
        metric: str = "squared_l2",
        num_neighbors: int = 100,
        num_leaves: int = 3000,
        num_leaves_to_search: int = 10,
        training_samples_size: int = -1,
        num_threads: int = 2,
    ):
        """
        利用给定的向量和参数构建scann倒排索引
        :param vectors: 构建索引所用的向量
        :param ids: 构建向量对应的ids
        :param metric: 距离度量
        :param num_neighbors: 返回的最近邻数量
        :param num_leaves: 搜索树的叶节点个数
        :param num_leaves_to_search: 搜索的叶子个数
        :param training_samples_size: 训练采样数量
        :param num_threads: 线程数量
        """
        start = time.time()
        if training_samples_size == -1:
            training_samples_size = vectors.shape[0]

        if ids is None:
            ids = np.arange(vectors.shape[0]).tolist()
        else:
            ids = ids.tolist()

        searcher = (
            scann.scann_ops_pybind.builder(vectors, num_neighbors, metric)
            .tree(
                num_leaves=num_leaves,
                num_leaves_to_search=num_leaves_to_search,
                training_sample_size=training_samples_size,
                incremental_threshold=0.3,
            )
            .score_brute_force()
            .build(docids=ids)
        )
        searcher.set_num_threads(num_threads)
        self.index = searcher

        end = time.time()
        total_time = end - start
        return total_time

    def search(self, query: np.ndarray, k: int, leaves_to_search: int = 100, num_threads: int = 16):
        """
        查找k近邻
        :param query: 查询向量
        :param k: 要查找的近邻数量
        :param leaves_to_search: 搜索的叶子个数
        :param num_threads: 查询线程数量
        :return:
        """
        start = time.time()
        indices, distances = self.index.search_batched_parallel(
            query,
            final_num_neighbors=k,
            leaves_to_search=leaves_to_search,
            batch_size=query.shape[0]//num_threads,
            # batch_size=1
        )
        end = time.time()
        total_time = end - start
        indices = np.array(indices)

        return indices, distances, total_time

    def add(self, vectors: np.ndarray, ids: np.ndarray = None, num_threads: int = 2):
        """
        向当前索引中添加向量
        :param vectors: 要添加的向量
        :param ids: 添加向量的ids
        :param num_threads: 线程数量
        """
        assert self.index is not None

        if ids is None:
            curr_id = self.n_total()
            ids = np.arange(curr_id, curr_id + vectors.shape[0], dtype=np.int64)

        ids = ids.tolist()
        start = time.time()
        self.index.upsert(ids, vectors, vectors.shape[0]//num_threads)
        end = time.time()
        total_time = end - start

        return total_time

    def remove(self, ids: np.ndarray = None):
        """
        从当前索引中删除向量
        :param ids: 要删除的向量ids
        """
        assert self.index is not None

        ids = ids.tolist()
        start = time.time()
        self.index.delete(ids)
        end = time.time()
        total_time = end - start

        return total_time

    def maintenance(self):
        """
        执行索引中的维护操作
        """
        return