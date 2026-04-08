import faiss
import numpy as np
import time
from src.py.index_wrappers.wrapper import IndexWrapper


class FaissIVF(IndexWrapper):
    """
    faiss倒排索引
    """

    index: faiss.Index

    def __init__(self):
        self.index = None

    def n_total(self) -> int:
        return self.index.ntotal

    def d(self) -> int:
        return self.index.d

    def build(
        self,
        vectors: np.ndarray,
        ids: np.ndarray = None,
        metric: str = "l2",
        nc: int = 3000,
        num_threads: int = 2,
    ):
        """
        构建FaissIVF索引
        :param vectors: 构建索引所用的向量
        :param ids: 构建向量对应的ids
        :param metric: 距离度量
        :param nc: 分区中心数量
        :param num_threads: 线程数量
        """
        start = time.time()
        assert nc >= 0

        d = vectors.shape[1]
        if metric == "l2":
            quantizer = faiss.IndexFlatL2(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nc, faiss.METRIC_L2)
        elif metric == "ip":
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(quantizer, d, nc, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"Invalid metric: {metric}")

        if not self.index.is_trained:
            print("Training the index...")
            self.index.train(vectors)

        print("Adding vectors to the index...")
        if ids is None:
            self.index.add(vectors)
        else:
            self.index.add_with_ids(vectors, ids.astype(np.int64))
        print("Index built.")
        end = time.time()
        return end-start

    def search(self, query: np.ndarray, k: int, nprobe: int = 1, num_threads: int = 2):
        """
        查找k近邻
        :param query: 查询向量
        :param k: 要查找的近邻数量
        :param nprobe: 搜索过程中的分区查询数量
        :param num_threads: 查询线程数量，默认0
        :return: k近邻对应的id和距离以及查询耗时
        """
        self.index.nprobe = nprobe
        self.index.num_threads = num_threads

        start = time.time()
        distances, indices = self.index.search(query, k)
        end = time.time()
        total_time = end - start

        return indices, distances, total_time

    def add(self, vectors: np.ndarray, ids: np.ndarray = None, num_threads: int = 2):
        """
        向当前索引中添加向量
        :param vectors: 要添加的向量
        :param ids: 添加向量的ids
        :param num_threads: 线程数量
        """
        start = time.time()
        if ids is None:
            self.index.add(vectors)
        else:
            self.index.add_with_ids(vectors, ids.astype(np.int64))
        end = time.time()
        total_time = end - start

        return total_time

    def remove(self, ids: np.ndarray):
        """
        从当前索引中删除向量
        :param ids: 要删除的向量ids
        """
        start = time.time()
        self.index.remove_ids(ids)
        end = time.time()
        total_time = end - start

        return total_time

    def maintenance(self):
        """
        执行索引中的维护操作
        """
        return