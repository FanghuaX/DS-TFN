import numpy as np

from preprocess.parse_csv import EHRParser


def split_patients(patient_admission, admission_codes, code_map, train_num, test_num, seed=6669):
    np.random.seed(seed) # 设置随机种子
    common_pids = set() #初始化一个set
    for i, code in enumerate(code_map): # code_map来源于admission_codes,
        # i 获取code_map中的序号[0到4855]， code，获取code_map中的住院号，ICD9_CODE，
        print('\r\t%.2f%%' % ((i + 1) * 100 / len(code_map)), end='')
        for pid, admissions in patient_admission.items():
        # pid 获取patient_admission.items()中的病人编号，admissions获取每个病人的所有入院编号及时间，
            for admission in admissions:
                codes = admission_codes[admission[EHRParser.adm_id_col]] # codes获取每个入院编号在admission_codes中对应的疾病编码ICD_CODE；
                if code in codes: # 如果code_map中的住院号，ICD9_CODE在patient_admission.items()中的病人编号HADM_ID中对应的ICD_CODE中。
                    common_pids.add(pid) # 将病人编号加到common_pids中去
                    break
            else:
                continue
            break
        # common_pids，找出每一种病首次在首次出现在admission_codes中的病人的pid
    print('\r\t100%')
    max_admission_num = 0
    pid_max_admission_num = 0
    for pid, admissions in patient_admission.items():
        if len(admissions) > max_admission_num:
            max_admission_num = len(admissions)
            pid_max_admission_num = pid
    common_pids.add(pid_max_admission_num) # 加入就诊次数最大的pid
    # set(patient_admission.keys()表示patient_admission中所有pid的集合，
    # list(set(patient_admission.keys()).difference(common_pids))，返回在set(patient_admission.keys())中，而不在common_pids中的集合
    remaining_pids = np.array(list(set(patient_admission.keys()).difference(common_pids)))
    np.random.shuffle(remaining_pids) # 将remaining_pids随机打乱

    valid_num = len(patient_admission) - train_num - test_num # valid_num=493
    train_pids = np.array(list(common_pids.union(set(remaining_pids[:(train_num - len(common_pids))].tolist())))) # .union,返回多个集合（集合的数量大于等于2）的并集
    valid_pids = remaining_pids[(train_num - len(common_pids)):(train_num + valid_num - len(common_pids))]
    test_pids = remaining_pids[(train_num + valid_num - len(common_pids)):]
    return train_pids, valid_pids, test_pids


def build_code_xy(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num):
    n = len(pids) #n表示病人的序号，max_admission_num表示病人的就诊次数，code_num，表示该病人每次就诊的code对应的序号
    x = np.zeros((n, max_admission_num, code_num), dtype=float) # 生成6000*42*4865的三维矩阵，记录每个病人每次就诊是否诊断code中的疾病
    y = np.zeros((n, code_num), dtype=int) # 生成6000*4865的二维矩阵，表示y的值
    lens = np.zeros((n,), dtype=int) # 生成6000*1的矩阵 表示历史就诊次数
    for i, pid in enumerate(pids):
        # i 获取序号，pid获取序号对应的编码
        print('\r\t%d / %d' % (i + 1, len(pids)), end='')
        admissions = patient_admission[pid] # 获取pid获取在patient_admission中对应的就诊编码和就诊时间
        for k, admission in enumerate(admissions[:-1]): #admissions[:-1]表示从第一项到倒数第二项逐项读取
            # k表示每个病人的就诊次数，admission获取每一个就诊编码和就诊时间
            codes = admission_codes_encoded[admission[EHRParser.adm_id_col]] # 获取就诊编码在admission_codes_encoded中对应的ICD_CODE对应的序号
            x[i, k, codes] = 1
        codes = np.array(admission_codes_encoded[admissions[-1][EHRParser.adm_id_col]]) # 获取每就诊编码对应的ICD编码的顺序
        y[i, codes] = 1 # 表示病人最后一次出现的code对应的序号
        lens[i] = len(admissions) - 1 #表示之前就诊的次数
    print('\r\t%d / %d' % (len(pids), len(pids)))
    return x, y, lens


def build_heart_failure_y(hf_prefix, codes_y, code_map):
    hf_list = np.array([cid for code, cid in code_map.items() if code.startswith(hf_prefix)])
    hfs = np.zeros((len(code_map),), dtype=int)
    hfs[hf_list] = 1
    hf_exist = np.logical_and(codes_y, hfs)
    y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
    return y


# def build_heart_failure_y(hf_prefix, codes_y, code_map):
#     # 确保 hf_list 中的元素为整数类型
#     hf_list = np.array([int(cid) for code, cid in code_map.items() if code.startswith(hf_prefix)])
#
#     # 打印 hf_list 的元素类型进行调试
#     print(f"hf_list dtype: {hf_list.dtype}")
#
#     hfs = np.zeros((len(code_map),), dtype=int)
#     hfs[hf_list] = 1
#     hf_exist = np.logical_and(codes_y, hfs)
#     y = (np.sum(hf_exist, axis=-1) > 0).astype(int)
#     return y