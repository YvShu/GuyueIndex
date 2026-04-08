import json
import utils
import os
import numpy as np
import src.py.index_wrappers.wrapper as index_wrapper


def process_workload(input_data_file, input_runbook_file, gt_dir, output_file, index_name, metric, k, target_recall):
    IndexClass = index_wrapper.get_index_class(index_name)
    index = IndexClass()
    vectors = utils.read_fvecs(input_data_file)
    runbook = json.load(open(input_runbook_file, "r"))
    search_count = 0
    step = 1
    is_build = False

    # --- 执行操作 ---
    for op in runbook["operations"]:
        if op["operation"] == "build":
            print("build")
            start = op["start"]
            end = op["end"]
            # index.build(vectors[start:end+1])
        elif op["operation"] == "insert":
            print("insert")
            start = op["start"]
            end = op["end"]
            ids = np.arange(start, end + 1).astype(np.uint32)
            if is_build == False:
                T = index.build(vectors[start:end+1]) * 1000
                is_build = True
            else:
                T = index.add(vectors[start:end+1], ids) * 1000
            print("step:", step, "insert time:", T, "ms")
            insert_info = [
                [f"{step}", "insert time", f"{T}"],
                [f"{step}", "reindexing time", f"0"],
            ]
            utils.write_csv_app(output_file, insert_info)
            step += 1
        elif op["operation"] == "delete":
            print("delete")
            start = op["start"]
            end = op["end"]
            ids = np.arange(start, end + 1).astype(np.uint32)
            T = index.remove(ids) * 1000
            print("step:", step, "delete time:", T, "ms")
            delete_info = [
                [f"{step}", "delete time", f"{T}"],
                [f"{step}", "reindexing time", f"0"],
            ]
            utils.write_csv_app(output_file, delete_info)
            step += 1
        elif op["operation"] == "search":
            print("search")
            queries = utils.read_fvecs(op["query_file"])[:4]
            gt_file = os.path.join(gt_dir, f"gt_step_{search_count}.ivecs")
            gt = utils.read_ivecs(gt_file)[:4]

            nprobe = 50
            # I, D, T = index.search(queries, k, nprobe, 16)
            # recall = utils.compute_recall(I, gt, k)
            # recall_mean = round(np.mean(recall), 4)
            # recall_mean = np.mean(recall)
            # print("nprobe:", nprobe)
            # print("step:", step, "search time:", T * 1000, "ms")
            # print("recall:", recall_mean)
            while nprobe < 3000:
                I, D, T = index.search(queries, k, nprobe, 2)
                recall = utils.compute_recall(I, gt, k)
                # recall_mean = round(np.mean(recall), 4)
                recall_mean = np.mean(recall)
                # print("nprobe:", nprobe)
                # print("step:", step, "search time:", T * 1000, "ms")
                # print("recall:", recall_mean)

                if recall_mean > target_recall or nprobe == 1:
                    print("nprobe:", nprobe)
                    print("step:", step, "search time:", T * 1000, "ms")
                    print("recall:", recall_mean)
                    search_info = [
                        [f"{step}", "search time", f"{T*1000}"],
                        [f"{step}", "search nprobe", f"{nprobe}"],
                        [f"{step}", "qps", f"{queries.shape[0] / T}"],
                        [f"{step}", f"recall@{k}", f"{recall_mean}"],
                    ]
                    utils.write_csv_app(output_file, search_info)
                    step += 1
                    # nprobe -= 20
                    break
                if recall_mean > target_recall:
                    nprobe -= 1
                else:
                    nprobe += 1
            search_count += 1
        # index.maintenance()

if __name__ == '__main__':
    index_name = "ScannIVF"
    workload = "workload2"
    dataset = "sift-2M"
    metric = "l2"
    k = 100
    target_recall = 0.98
    # input_data_file = r"/mnt/hgfs/DataSet/sift/sift_base.fvecs"
    # input_runbook_file = r"/mnt/hgfs/DataSet/sift/runbook.json"
    # input_data_file = f"/mnt/hgfs/DataSet/{dataset}/{workload}/{dataset.split('-')[0]}_base.fvecs"
    input_runbook_file = f"/mnt/hgfs/DataSet/{dataset}/{workload}/runbook.json"
    output_file = f"/home/guyue/GuyueIndex/output/{workload}/{dataset}_{index_name}_{k}.csv"
    input_data_file = f"/mnt/hgfs/DataSet/{dataset}/{dataset.split('-')[0]}_base.fvecs"
    # input_runbook_file = f"/mnt/hgfs/DataSet/{dataset}/runbook.json"
    # output_file = f"/home/guyue/StreamIndex/output/{workload}/{dataset}_{index_name}_{k}.csv"
    gt_dir = os.path.dirname(input_runbook_file)

    process_workload(
        input_data_file,
        input_runbook_file,
        gt_dir,
        output_file,
        index_name,
        metric,
        k,
        target_recall)
