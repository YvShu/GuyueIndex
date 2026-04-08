import diskannpy
import numpy as np
import time
from src.py.index_wrappers.wrapper import IndexWrapper


class DiskANNDynamic(IndexWrapper):
    """
    diskann动态索引
    """

    def __init__(self):
        self.index = None

    def n_total(self) -> int:
        return self.index.get_current_count()

    def d(self) -> int:
        return self.index.get_dimensions()

    def build(
        self,
        vectors: np.ndarray,
        ids: np.ndarray = None,
        metric: str = "l2",
        max_vectors: int = 2_000_000,
        complexity: int = 100,
        graph_degree: int = 32,
        num_threads: int = 2,
    ):
        """
        使用给定的向量和参数构建索引
        :param vectors: 构建索引所使用的向量
        :param ids: 向量对应的索引id
        :param metric: 使用的距离度量，默认“l2”
        :param max_vectors: 索引能够接受的最大向量数量，默认1_000_000
        :param complexity: 索引的复杂度，默认32
        :param graph_degree: 图的度，默认16
        :param num_threads: 使用的线程数量，默认0
        """
        start = time.time()
        d = vectors.shape[1]
        self.index = diskannpy.DynamicMemoryIndex(
            distance_metric=metric,
            vector_dtype=vectors.dtype,
            dimensions=d,
            max_vectors=max_vectors,
            complexity=complexity,
            graph_degree=graph_degree,
            num_threads=num_threads,
        )

        if ids is None:
            ids = np.arange(vectors.shape[0]).astype(np.uint32)

        ids = ids + 1   # diskann中0为无效id
        self.index.batch_insert(vectors, ids)
        end = time.time()
        return end - start

    def search(self, query: np.ndarray, k: int, complexity: int = 16, num_threads: int = 16):
        """
        查询k近邻
        :param query: 查询向量
        :param k: 要查找的近邻数量
        :param complexity: 查询复杂度，默认16
        :param num_threads: 查询线程数量，默认0
        :return: k近邻对应的id和距离以及查询耗时
        """
        assert self.index is not None

        start = time.time()
        indices, distances = self.index.batch_search(
            query, k_neighbors=k, complexity=complexity, num_threads=num_threads
        )
        end = time.time()
        total_time = end - start

        return indices-1, distances, total_time

    def add(self, vectors: np.ndarray, ids: np.ndarray = None, num_threads: int = 2):
        """
        向当前索引中添加向量
        :param vectors: 要添加的向量
        :param ids: 添加向量的id
        :param num_threads: 线程数量
        """
        assert self.index is not None
        assert vectors.shape[0] == ids.shape[0]

        ids = ids + 1
        start = time.time()
        self.index.batch_insert(vectors, ids, num_threads=num_threads)
        end = time.time()
        total_time = end - start

        return total_time

    def remove(self, ids: np.ndarray, lazy: bool = True):
        """
        从当前索引中删除向量
        :param ids: 要删除的向量ids
        :param lazy: 是否惰性删除向量，默认True
        """
        assert self.index is not None

        ids = ids + 1
        start = time.time()
        for id in ids:
            self.index.mark_deleted(id)
        if not lazy:
            self.index.consolidate_delete()
        end = time.time()
        total_time = end - start

        return total_time

    def maintenance(self):
        """
        执行索引中的维护操作
        """
        return