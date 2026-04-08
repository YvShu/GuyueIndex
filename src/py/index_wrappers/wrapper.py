'''
Author: Guyue
Date: 2025-11-10 15:57:05
LastEditTime: 2025-11-13 11:55:28
LastEditors: Guyue
FilePath: /StreamIndex/src/python/index_wrappers/wrapper.py
'''
import abc
from abc import abstractmethod
from typing import Optional, Tuple, Union
import numpy as np


def get_index_class(index_name):
    """
    工厂函数
    :param index_name: 索引名称
    :return: 根据索引名称返回索引
    """
    if index_name == "FaissIVF":
        from src.py.index_wrappers.faiss_ivf import FaissIVF as IndexClass
    # elif index_name == "Quake":
    #     from src.py.index_wrappers.quake import Quake as IndexClass
    elif index_name == "FaissHNSW":
        from src.py.index_wrappers.faiss_hnsw import FaissHNSW as IndexClass
    elif index_name == "DiskANN":
        from src.py.index_wrappers.diskann import DiskANNDynamic as IndexClass
    elif index_name == "ScannIVF":
        from src.py.index_wrappers.scann_ivf import Scann as IndexClass
    return IndexClass

class IndexWrapper(abc.ABC):
    """
    抽象基类
    """
    @abstractmethod
    def n_total(self) -> int:
        """返回索引中的向量数量"""
        raise NotImplementedError("Subclasses must implement n_total method")

    @abstractmethod
    def d(self) -> int:
        """返回索引中维度大小"""
        raise NotImplementedError("Subclasses must implement d method")

    @abstractmethod
    def build(self, vectors: np.ndarray, *args):
        """利用提供的参数构建索引"""
        raise NotImplementedError("Subclasses must implement build method")

    @abstractmethod
    def search(self, vectors: np.ndarray, k: int, nprobe: int,*args) -> Tuple[np.ndarray, np.ndarray, float]:
        """查找最近邻TopK"""
        raise NotImplementedError("Subclasses must implement search method")

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: np.ndarray = None, *args) -> float:
        """向当前索引中添加向量"""
        raise NotImplementedError("Subclasses must implement add method")

    @abstractmethod
    def remove(self, ids: np.ndarray) -> float:
        """从当前索引中删除向量"""
        raise NotImplementedError("Subclasses must implement remove method")

    @abstractmethod
    def maintenance(self):
        """执行索引中的维护操作"""
        return None