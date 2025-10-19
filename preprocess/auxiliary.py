
import numpy as np
import torch
from torch import nn
from preprocess.parse_csv import EHRParser

def generate_code_code_adjacent(pids, patient_admission, admission_codes_encoded, code_num, threshold=0.01):
    print('generating code code adjacent matrix ...')
    n = code_num # 4856
    adj = np.zeros((n, n), dtype=int)
    for i, pid in enumerate(pids):
        print('\r\t%d / %d' % (i, len(pids)), end='')
        for admission in patient_admission[pid]: # 获取每个病人的入院编码和时间
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            for row in range(len(codes) - 1):
                for col in range(row + 1, len(codes)):
                    c_i = codes[row]
                    c_j = codes[col]
                    adj[c_i, c_j] += 1
                    adj[c_j, c_i] += 1
    print('\r\t%d / %d' % (len(pids), len(pids)))
    norm_adj = normalize_adj(adj)
    a = norm_adj < threshold # 和阈值做比较
    b = adj.sum(axis=-1, keepdims=True) > (1 / threshold) #adj.sum(axis=-1, keepdims=True) adj按列求和，为什么求和数要大于100
    adj[np.logical_and(a, b)] = 0 # np.logical_and(a, b)为真的位置全是0
    return adj


def generate_code_code_adjacent_and_temporal(pids, patient_admission, admission_codes_encoded, code_num,
                                             threshold=0.01):
    n = code_num
    adj_cooccur = np.zeros((n, n), dtype=int)
    adj_temporal = np.zeros((n, n), dtype=int)

    for i, pid in enumerate(pids):
        admissions = patient_admission[pid]
        prev_codes = []
        for j, admission in enumerate(admissions):
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]]
            if j > 0:
                for prev_code in prev_codes:
                    for code in codes:
                        adj_temporal[prev_code, code] += 1
            for row in range(len(codes) - 1):
                for col in range(row + 1, len(codes)):
                    c_i = codes[row]
                    c_j = codes[col]
                    adj_cooccur[c_i, c_j] += 1
                    adj_cooccur[c_j, c_i] += 1
            prev_codes = codes

    adj_cooccur = normalize_adj(adj_cooccur)
    # a = adj_cooccur < threshold  # 和阈值做比较
    # b = adj_cooccur.sum(axis=-1, keepdims=True) > (1 / threshold)
    # adj_cooccur[np.logical_and(a, b)] = 0

    adj_temporal = normalize_adj(adj_temporal)
    # a = adj_temporal < threshold  # 和阈值做比较
    # b = adj_temporal.sum(axis=-1, keepdims=True) > (1 / threshold)
    # adj_temporal[np.logical_and(a, b)] = 0

    return adj_cooccur, adj_temporal





def normalize_adj(adj):
    s = adj.sum(axis=-1, keepdims=True)
    s[s == 0] = 1
    result = adj / s # adj一列一列的去除以s
    return result


def generate_neighbors(code_x, lens, adj): # 得到全局诊断邻接图
    n = len(code_x)
    neighbors = np.zeros_like(code_x, dtype=bool)
    # a = 0
    # b = 100000
    # c = -1
    # nn = 0
    for i, admissions in enumerate(code_x): #admissions，每次就诊的结果
        print('\r\t%d / %d' % (i + 1, n), end='')
        for j in range(lens[i]):
            codes_set = set(np.where(admissions[j] == 1)[0]) #np.where输出满足条件元素的坐标，得到每次就诊的疾病位置
            all_neighbors = set()
            for code in codes_set:
                code_neighbors = set(np.where(adj[code] > 0)[0]).difference(codes_set) #mySet1.difference(mySet2)的结果是：返回集合mySet1中有，但是在mySet2集合中没有的元素。
                # code_neighbors 获取code在adj中不在当前就诊中的邻接点
                all_neighbors.update(code_neighbors) #update()函数用于将两个字典合并操作，有相同的就覆盖
                # all_neighbors 获取codes_set中邻接点，构成完全图
            if len(all_neighbors) > 0:
                neighbors[i, j, np.array(list(all_neighbors))] = 1 #将病人的每个就诊记录中确诊疾病的序号和该疾病序号对应领接矩阵为大于0的位置定为1
            # a += len(all_neighbors)
            # if b > len(all_neighbors):
            #     b = len(all_neighbors)
            # if c < len(all_neighbors):
            #     c = len(all_neighbors)
            # nn += 1
    print('\r\t%d / %d' % (n, n))
    # print(b, c, a / nn);exit()
    return neighbors # 疾病的邻居子图


def divide_middle(code_x, neighbors, lens):
    n = len(code_x)
    divided = np.zeros((*code_x.shape, 3), dtype=bool)  # 构建一个6000*42*4856*3的矩阵
    for i, admissions in enumerate(code_x):
        print('\r\t%d / %d' % (i + 1, n), end='')
        divided[i, 0, :, 0] = admissions[0]
        for j in range(1, lens[i]):
            codes_set = set(np.where(admissions[j] == 1)[0]) #第j次出现的疾病
            m_set = set(np.where(admissions[j - 1] == 1)[0]) #第j-1次出现的疾病
            n_set = set(np.where(neighbors[i][j - 1] == 1)[0]) #第j-1次出现的邻接疾病
            m1 = codes_set.intersection(m_set) # 连续两次出现的疾病 顽固疾病 .intersection求集合交集的函数 在最早两就诊均出现的疾病
            m2 = codes_set.intersection(n_set) # 在第一次和前一次邻接均出现的疾病 新兴疾病
            m3 = codes_set.difference(m_set).difference(n_set) # 新兴无关疾病 在m_set不在n_set的
            if len(m1) > 0:
                divided[i, j, np.array(list(m1)), 0] = 1 # 顽固疾病
            if len(m2) > 0:
                divided[i, j, np.array(list(m2)), 1] = 1 # 新兴邻接
            if len(m3) > 0:
                divided[i, j, np.array(list(m3)), 2] = 1 # 新兴无关疾病
    print('\r\t%d / %d' % (n, n))
    return divided


def parse_icd9_range(range_: str) -> (str, str, int, int):
    ranges = range_.lstrip().split('-')
    if ranges[0][0] == 'V':
        prefix = 'V'
        format_ = '%02d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    elif ranges[0][0] == 'E':
        prefix = 'E'
        format_ = '%03d'
        start, end = int(ranges[0][1:]), int(ranges[1][1:])
    else:
        prefix = ''
        format_ = '%03d'
        if len(ranges) == 1:
            start = int(ranges[0])
            end = start
        else:
            start, end = int(ranges[0]), int(ranges[1])
    return prefix, format_, start, end


def generate_code_levels(path, code_map: dict) -> np.ndarray:
    print('generating code levels ...')
    import os
    three_level_code_set = set(code.split('.')[0] for code in code_map)
    icd9_path = os.path.join(path, 'icd9.txt')
    icd9_range = list(open(icd9_path, 'r', encoding='utf-8').readlines())
    three_level_dict = dict()
    level1, level2, level3 = (0, 0, 0)
    level1_can_add = False
    for range_ in icd9_range:
        range_ = range_.rstrip()
        if range_[0] == ' ':
            prefix, format_, start, end = parse_icd9_range(range_)
            level2_cannot_add = True
            for i in range(start, end + 1):
                code = prefix + format_ % i
                if code in three_level_code_set:
                    three_level_dict[code] = [level1, level2, level3]
                    level3 += 1
                    level1_can_add = True
                    level2_cannot_add = False
            if not level2_cannot_add:
                level2 += 1
        else:
            if level1_can_add:
                level1 += 1
                level1_can_add = False

    code_level = dict()
    for code, cid in code_map.items():
        three_level_code = code.split('.')[0]
        three_level = three_level_dict[three_level_code]
        code_level[code] = three_level + [cid]

    code_level_matrix = np.zeros((len(code_map), 4), dtype=int)
    for code, cid in code_map.items():
        code_level_matrix[cid] = code_level[code]

    return code_level_matrix


def generate_subgraphs_for_patients(code_x, visit_lens, adj_cooccur, adj_temporal):
    """
    为每个患者预计算共现子图和时序子图，用于减少模型训练时的计算开销
    
    参数:
        code_x: 患者的就诊记录数据，shape=(n_patients, max_visits, n_codes)
        visit_lens: 每个患者的有效就诊次数，shape=(n_patients,)
        adj_cooccur: 诊断码共现邻接矩阵
        adj_temporal: 诊断码时序邻接矩阵
        
    返回:
        cooccur_subgraphs: 每个患者的共现子图 [(adj_matrix, node_indices), ...]
        temporal_subgraphs: 每个患者的时序子图 [(adj_matrix, node_indices), ...]
    """
    print('Generating subgraphs for patients...')
    n_patients = len(code_x)
    cooccur_subgraphs = []
    temporal_subgraphs = []
    
    # for i, (patient_visits, visit_len) in enumerate(zip(code_x, visit_lens)):
    #     print('\r\t%d / %d' % (i + 1, n_patients), end='')
    #
    #     # 提取有效就诊记录
    #     visits = patient_visits[:visit_len]
    #
    #     # 生成共现子图
    #     # 找出所有出现过的诊断码
    #     all_codes = set()
    #     for visit in visits:
    #         codes = np.where(visit == 1)[0]
    #         all_codes.update(codes)
    #     all_codes = np.array(sorted(list(all_codes)))
    #
    #     # 提取子图邻接矩阵
    #     sub_cooccur = adj_cooccur[all_codes][:, all_codes]
    #
    #     # 生成时序子图
    #     node_dict = {}
    #     edge_list = []
    #
    #     # 遍历相邻就诊记录
    #     for j in range(len(visits) - 1):
    #         # 获取当前和下一就诊的诊断码
    #         src_codes = np.where(visits[j] == 1)[0]
    #         dst_codes = np.where(visits[j + 1] == 1)[0]
    #
    #         # 生成有效边
    #         for src in src_codes:
    #             for dst in dst_codes:
    #                 if adj_temporal[src, dst] > 0:
    #                     edge_list.append((src, dst))
    #                     # 动态维护节点映射
    #                     if src not in node_dict:
    #                         node_dict[src] = len(node_dict)
    #                     if dst not in node_dict:
    #                         node_dict[dst] = len(node_dict)
    #
    #     # 如果没有时序边，使用共现子图的节点
    #     if len(node_dict) == 0:
    #         temporal_nodes = all_codes
    #         sub_temporal = np.zeros((len(temporal_nodes), len(temporal_nodes)), dtype=adj_temporal.dtype)
    #     else:
    #         # 创建子图矩阵
    #         temporal_nodes = np.array(sorted(node_dict.keys()))
    #         sub_temporal = np.zeros((len(temporal_nodes), len(temporal_nodes)), dtype=adj_temporal.dtype)
    #
    #         # 填充边权值
    #         for src, dst in edge_list:
    #             i = node_dict[src]
    #             j = node_dict[dst]
    #             sub_temporal[i, j] = adj_temporal[src, dst]
    #

    for code_x_i, len_i in zip(code_x, visit_lens):
        visits = code_x_i[:len_i]
        # 获取现子图
        cooccur = adj_cooccur
        visits = [torch.from_numpy(v) if isinstance(v, np.ndarray) else v for v in visits]
        nodes = torch.unique(torch.cat([torch.where(v == 1)[0] for v in visits]))
        # nodes = torch.unique(torch.cat([torch.where(v == 1)[0] for v in visits]))
        sub_cooccur = cooccur[nodes][:, nodes]

        # 获取时序共现子图
        temporal = adj_temporal
        node_dict = {}
        edge_list = []

        # 遍历相邻就诊记录
        for i in range(len(visits) - 1):
            # 获取当前和下一就诊的有效编码索引
            src_codes = torch.where(visits[i])[0].tolist()  # 使用实际非零位置
            dst_codes = torch.where(visits[i + 1])[0].tolist()

            # 生成有效边
            for src in src_codes:
                for dst in dst_codes:
                    if temporal[src, dst] > 0:
                        edge_list.append((src, dst))
                        # 动态维护节点映射
                        if src not in node_dict:
                            node_dict[src] = len(node_dict)
                        if dst not in node_dict:
                            node_dict[dst] = len(node_dict)
        # 创建子图矩阵
        node_count = len(node_dict)
        torch_dtype = torch.float64 if temporal.dtype == np.float64 else torch.float32
        sub_temporal = torch.zeros((node_count, node_count), dtype=torch_dtype)
        # sub_temporal = torch.zeros((node_count, node_count), dtype=temporal.dtype)

        # 填充边权值
        for src, dst in edge_list:
            i = node_dict[src]
            j = node_dict[dst]
            sub_temporal[i, j] = temporal[src, dst]

        # 获取有序节点列表
        nodes1 = torch.tensor(sorted(node_dict.keys(), key=lambda x: node_dict[x]))

        cooccur_subgraphs.append((sub_cooccur, nodes))
        temporal_subgraphs.append((sub_temporal, nodes1))
    
    print('\r\t%d / %d' % (n_patients, n_patients))
    return cooccur_subgraphs, temporal_subgraphs
