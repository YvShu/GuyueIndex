import re
import os
import pandas as pd

dataset = "sift-2M"
workload = "workload2"
base_path = fr"/home/guyue/GuyueIndex/output/{workload}/"
save_name = dataset + ""
k = 100
q = 4
p = 3000

# 实验结果配置
group_config = {
    # f"{dataset}_Full_q{q}_p{p}_{k}": {'marker': '^', 'label': "Full"},
    # f"{dataset}_None_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None"},
    # f"{dataset}_None-Update_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None+Update"},
    # f"{dataset}_None-LIRE_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None+LIRE"},
    # f"{dataset}_LIRE_q{q}_200-400_b1000_{k}": {'marker': 's', 'label': "LIRE 200-400"},
    # f"{dataset}_LIRE_q{q}_200-410_b1000_{k}": {'marker': 's', 'label': "LIRE 200-410"},
    # f"{dataset}_LIRE_q{q}_200-420_b1000_{k}": {'marker': 's', 'label': "LIRE 200-420"},
    # f"{dataset}_LIRE_q{q}_p{p}_b1000_{k}": {'marker': 's', 'label': "LIRE"},
    # f"{dataset}_LIRE-Hit_q{q}_b1000_{k}": {'marker': 's', 'label': "LIRE Hit"},
    # f"{dataset}_FaissIVF_q{q}_p{p}_{k}": {'marker': 's', 'label': "FaissIVF"},
    # f"{dataset}_ScannIVF_q{q}_p{p}_{k}": {'marker': 's', 'label': "ScannIVF"},
    # f"{dataset}_ScannIVF-.2_q{q}_p{p}_{k}": {'marker': 's', 'label': "ScannIVF .2"},
    # f"{dataset}_ScannIVF-.3_q{q}_p{p}_{k}": {'marker': 's', 'label': "ScannIVF .3"},
    # f"{dataset}_DiskANN_q{q}_{k}": {'marker': 's', 'label': "DiskANN"},

    # f"{dataset}_Full-LVQ_q{q}_p{p}_{k}": {'marker': '^', 'label': "Full LVQ"},
    # f"{dataset}_None-LVQ_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None LVQ"},
    # f"{dataset}_None-LVQ-Update_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None LVQ+Update"},
    # f"{dataset}_None-LVQ-LIRE_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None LVQ+LIRE"},
    # f"{dataset}_LIRE-LVQ_q{q}_200-400_b1000_{k}": {'marker': 's', 'label': "LIRE LVQ 200-400"},
    # f"{dataset}_LIRE-LVQ_q{q}_200-410_b1000_{k}": {'marker': 's', 'label': "LIRE LVQ 200-410"},
    # f"{dataset}_LIRE-LVQ_q{q}_200-420_b1000_{k}": {'marker': 's', 'label': "LIRE LVQ 200-420"},
    # f"{dataset}_LIRE-LVQ_q{q}_p{p}_b1000_{k}": {'marker': 's', 'label': "LIRE LVQ"},

    # f"{dataset}_Full-LVQ_q{q}_p{p}_{k}": {'marker': '^', 'label': "Full LVQ"},
    # f"{dataset}_None-LVQ_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None LVQ"},
    # f"{dataset}_None-LVQ-Update_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None LVQ+Update"},
    # f"{dataset}_None-LVQ-LIRE_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None LVQ+LIRE"},
    # f"{dataset}_LIRE-LVQ_q{q}_p{p}_b1000_{k}": {'marker': 's', 'label': "LIRE LVQ"},

    f"{dataset}_Full-Common-LVQ_q{q}_p{p}_{k}": {'marker': '^', 'label': "Full LVQ"},
    f"{dataset}_None-Common-LVQ_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None LVQ"},
    f"{dataset}_None-Common-LVQ-Update_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None LVQ+Update"},
    f"{dataset}_None-Common-LVQ-LIRE_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None LVQ+LIRE"},
    f"{dataset}_LIRE-Common-LVQ_q{q}_p{p}_b1000_{k}": {'marker': 's', 'label': "LIRE LVQ"},

    # f"{dataset}_Full_q{q}_p{p}_{k}": {'marker': '^', 'label': "Full"},
    # f"{dataset}_None_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None"},
    # f"{dataset}_None-Update_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None+Update"},
    # f"{dataset}_None-LIRE_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None+LIRE"},
    # f"{dataset}_LIRE_q{q}_p{p}_b1000_{k}": {'marker': 's', 'label': "LIRE"},
    # f"{dataset}_FaissIVF_q{q}_p{p}_{k}": {'marker': 's', 'label': "FaissIVF"},
    # f"{dataset}_ScannIVF_q{q}_p{p}_{k}": {'marker': 's', 'label': "ScannIVF"},
    # f"{dataset}_ScannIVF-.2_q{q}_p{p}_{k}": {'marker': 's', 'label': "ScannIVF .2"},
    # f"{dataset}_ScannIVF-.3_q{q}_p{p}_{k}": {'marker': 's', 'label': "ScannIVF .3"},
    # f"{dataset}_DiskANN_q{q}_{k}": {'marker': 's', 'label': "DiskANN"},

    # f"{dataset}_Full-Common_q{q}_p{p}_{k}": {'marker': '^', 'label': "Full"},
    # f"{dataset}_None-Common_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None"},
    # f"{dataset}_None-Common-Update_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None+Update"},
    # f"{dataset}_None-Common-LIRE_q{q}_p{p}_b1000_{k}": {'marker': 'o', 'label': "None+LIRE"},
    # f"{dataset}_LIRE-Common_q{q}_p{p}_b1000_{k}": {'marker': 's', 'label': "LIRE"},
    # f"{dataset}_FaissIVF_q{q}_p{p}_{k}": {'marker': 's', 'label': "FaissIVF"},
    # f"{dataset}_ScannIVF_q{q}_p{p}_{k}": {'marker': 's', 'label': "ScannIVF"},
    # f"{dataset}_ScannIVF-.2_q{q}_p{p}_{k}": {'marker': 's', 'label': "ScannIVF .2"},
    # f"{dataset}_ScannIVF-.3_q{q}_p{p}_{k}": {'marker': 's', 'label': "ScannIVF .3"},
    # f"{dataset}_DiskANN_q{q}_{k}": {'marker': 's', 'label': "DiskANN"},
}

# 用于存储结果的列表
results = []
for _, config in enumerate(group_config):
    file_path = os.path.join(base_path, config + '.csv')
    with open(file_path, 'r') as f:
        content = f.read()

    search_time = 0
    insert_time = 0
    delete_time = 0
    reindexing_time = 0
    is_skip = False

    # 使用正则表达式提取所有步骤数和对应值(插入时间)
    pattern = r'^(\d+),insert\s+time,([\d.]+)'
    matches = re.findall(pattern, content, re.MULTILINE)
    for match in matches:
        if is_skip:
            insert_time += float(match[1])
        else:
            is_skip = True

    # 使用正则表达式提取所有步骤数和对应值(查询时间)
    pattern = r'^(\d+),search\s+time,([\d.]+)'
    matches = re.findall(pattern, content, re.MULTILINE)
    for match in matches:
        search_time += float(match[1])

    # 使用正则表达式提取所有步骤数和对应值(删除时间)
    pattern = r'^(\d+),delete\s+time,([\d.]+)'
    matches = re.findall(pattern, content, re.MULTILINE)
    for match in matches:
        delete_time += float(match[1])

    is_skip = False
    # 使用正则表达式提取所有步骤数和对应值(维护时间)
    pattern = r'^(\d+),reindexing\s+time,([\d.]+)'
    matches = re.findall(pattern, content, re.MULTILINE)
    for match in matches:
        if is_skip:
            reindexing_time += float(match[1])
        else:
            is_skip = True


    # 计算总时间
    total_time = search_time + insert_time + delete_time + reindexing_time

    # 添加到结果列表
    results.append({
        'Method': group_config[config]['label'],
        'Search': search_time,
        'Insert': insert_time,
        'Delete': delete_time,
        'Reindexing': reindexing_time,
        'Total': total_time
    })

# 创建DataFrame
df = pd.DataFrame(results)

# 设置显示格式，保留4位小数
pd.set_option('display.float_format', '{:.4f}'.format)

# 打印表格
print(df.to_string(index=False))

# # 保存为CSV文件
# if save_name:
#     csv_path = f"{save_name}.csv"
#     df.to_csv(csv_path, index=False, float_format='%.4f')
#     print(f"Results saved to {csv_path}")
#
#     # 可选：保存为Excel文件
#     try:
#         excel_path = f"{save_name}.xlsx"
#         df.to_excel(excel_path, index=False, float_format='%.4f')
#         print(f"Results also saved to {excel_path}")
#     except ImportError:
#         print("Note: Install 'openpyxl' to save as Excel format: pip install openpyxl")
#
#     # 可选：保存为JSON文件
#     json_path = f"{save_name}.json"
#     df.to_json(json_path, orient='records', indent=2)
#     print(f"Results also saved to {json_path}")
#
# print("=" * 80)
