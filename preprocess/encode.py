from collections import OrderedDict

from preprocess.parse_csv import EHRParser


def encode_code(patient_admission, admission_codes):
    code_map = OrderedDict()
    for pid, admissions in patient_admission.items(): # pid——病人ID SUBJECT_ID, admissions-对应的HAMD_ID和就诊时间
        for admission in admissions:# 针对病人的每次就诊序号HADM_ID和就诊时间ADMITTIME
            codes = admission_codes[admission[EHRParser.adm_id_col]] # 获取HADM_ID在admission_codes中对应的所有ICD_CODE病症编码；
            for code in codes:
                if code not in code_map:
                    code_map[code] = len(code_map)
            # 按照ICD_CODE病症编码的顺序，分别获取每个ICD_CODE病症编码值和对应次序（从0开始计算） 最终len(code_map)=4856
    admission_codes_encoded = {
        admission_id: list(set(code_map[code] for code in codes)) # 获取HADM_ID和在HADM_ID在admission_codes中对应的所有ICD_CODE病症编码对应在code_map中的序号
        for admission_id, codes in admission_codes.items()
    }
    return admission_codes_encoded, code_map
    # code_map，按照ICD_CODE病症编码的顺序，分别获取每个ICD_CODE病症编码值和对应次序（从0开始计算） 最终len(code_map)=4856
    #admission_codes_encoded，获取HADM_ID和在HADM_ID在admission_codes中对应的所有ICD_CODE病症编码对应在code_map中的序号，最终len(admission_codes_encoded)=19894