import faiss
from utils import *

if __name__ == '__main__':
    data_path = f"/mnt/hgfs/DataSet/sift-1M/sift_base.fvecs"
    query_path = f"/mnt/hgfs/DataSet/sift-1M/sift_query.fvecs"

    query_output = f"/mnt/hgfs/DataSet/sift-1M/sift_query_100.fvecs"

    data_base = read_fvecs(data_path)
    data_query = read_fvecs(query_path)
    # print(len(data_base), len(data_query))

    # data_query = random_sample_vectors(data_query, 10000, seed=42)
    # print(len(data_base), len(data_query))

    # write_fvecs(data_base, data_output)
    # write_fvecs(data_query, query_output)

    for i in range(10):
        print(data_base[i])

    for i in range(10):
        print(data_query[i])