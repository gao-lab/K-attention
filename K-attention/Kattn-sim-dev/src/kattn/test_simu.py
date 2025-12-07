import numpy as np
import pdb
import os
import h5py
import random
from sklearn.model_selection import StratifiedKFold
import math
import time
from matplotlib import pyplot as plt
# import seaborn as sns
import pandas as pd
plt.switch_backend('agg')
import glob
import pdb
import argparse
import sys


def init_parser():
    parser = argparse.ArgumentParser(description='Kattention Training Module.')

    # Data and Run Directories
    parser.add_argument('--position', dest='position', default='random',
                        help="insert position ['absolute','random','abs-ran','relative']")
    parser.add_argument('--feature', dest='feature', default='F_Markov',
                        help="insert feature ['F_Markov','S_Markov']")

    return parser.parse_args()

def mkdir(path):
    """

    :param path:
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def SpecialMotif():

    motif1 = np.asarray([[0.833021, 0.022514, 0.131332, 0.013133],
       [0.911704, 0.036961, 0.049281, 0.002053],
       [0.      , 1.      , 0.      , 0.      ],
       [0.977974, 0.006608, 0.015419, 0.      ],
       [0.347921, 0.065646, 0.560175, 0.026258],
       [0.026316, 0.774123, 0.076754, 0.122807],
       [0.024176, 0.      , 0.      , 0.975824],
       [0.      , 0.004484, 0.995516, 0.      ],
       [0.004274, 0.047009, 0.      , 0.948718],
       [0.      , 0.260656, 0.011475, 0.727869]])


    return motif1

def GenerateRandomMatrix(seqNum, seqLen):
    def seqSeriesToMatrix(seqSeries, seqLen):
        seqMatrix = np.zeros([seqLen, 4])
        for i in range(seqLen):
            seqMatrix[i, seqSeries[i]] = 1
        return seqMatrix

    Allseq = np.random.randint(0, 4, [seqNum, seqLen])
    AllseqArray = np.zeros([seqNum, seqLen, 4])
    for i in range(seqNum):
        seqTem = Allseq[i, :]
        seqTemMatrix = seqSeriesToMatrix(seqTem, seqLen)
        AllseqArray[i, :, :] = seqTemMatrix

    return AllseqArray

# 根据PWM矩阵生成长度为10的one-hot序列（基于概率）
def pwm_to_onehot(pwm_matrix):
    onehot_sequence = []
    for row in pwm_matrix:
        # 将PWM行归一化为概率分布
        probabilities = row / np.sum(row)
        # 根据概率分布随机选择一个索引
        chosen_index = np.random.choice(4, p=probabilities)
        # 生成one-hot编码
        onehot = [0, 0, 0, 0]
        onehot[chosen_index] = 1
        onehot_sequence.append(onehot)
    return np.array(onehot_sequence)


def one_hot_to_dna(one_hot_array):
    """
    将one-hot编码的DNA序列转换为ATCG字符串形式。

    参数:
    one_hot_array (numpy.ndarray): 形状为(n, 4)的one-hot编码数组。

    返回:
    str: 转换后的DNA序列字符串。
    """
    # 确保输入是numpy数组，并且形状是(n, 4)
    if not isinstance(one_hot_array, np.ndarray) or one_hot_array.shape[1] != 4:
        raise ValueError("输入必须是形状为(n, 4)的numpy数组")

        # 使用argmax找到每一行的最大值索引
    indices = np.argmax(one_hot_array, axis=1)

    # 定义碱基字母对应关系
    base_mapping = {0: 'A', 3: 'T', 1: 'C', 2: 'G'}

    # 映射索引到相应的碱基
    dna_sequence = ''.join(base_mapping[index] for index in indices)

    return dna_sequence

def dna_onehot(length=None, dna_sequence=None):
    """
    生成 DNA 序列的 one-hot 编码。

    参数：
    length (int): 生成的随机 DNA 序列长度。如果提供了 dna_sequence，此参数将被忽略。
    dna_sequence (str): 要编码的 DNA 字符串。如果提供了此参数，length 也将被忽略。

    返回：
    np.ndarray: DNA 序列的 one-hot 编码，形状为 (L, 4)，其中 L 是序列的长度。

    异常：
    ValueError: 如果同时提供了 length 和 dna_sequence，或两个参数均未提供，或包含无效碱基字符。
    """

    # 参数检查：必须提供 length 或 dna_sequence 之一
    if (length is not None) and (dna_sequence is not None):
        raise ValueError("不能同时指定 length 和 dna_sequence 参数。")
    elif dna_sequence is not None:
        seq = dna_sequence.upper()
    elif length is not None:
        bases = ['A', 'T', 'C', 'G']
        seq = ''.join(np.random.choice(bases, size=length))
    else:
        raise ValueError("必须提供 length 或 dna_sequence 参数。")

        # 检查序列中的非法字符
    valid_bases = {'A', 'T', 'C', 'G'}
    for base in seq:
        if base not in valid_bases:
            raise ValueError(f"非法碱基字符: {base}")

            # 创建映射字典
    base_to_index = {'A': 0, 'T': 3, 'C': 1, 'G': 2}

    # 生成 one-hot 编码
    indices = [base_to_index[base] for base in seq]
    onehot = np.eye(4)[indices]

    return onehot

# 生成两个随机的插入位置，确保不重叠且不越界
def generate_insert_positions(sequence_length, insert_length, num_positions=2):
    if sequence_length < insert_length * num_positions:
        raise ValueError("序列长度不足以插入指定数量的序列")

    positions = []
    while len(positions) < num_positions:
        # 随机生成一个起始位置
        pos = np.random.randint(0, sequence_length - insert_length + 1)
        # 检查是否与已生成的位置重叠
        overlap = False
        for existing_pos in positions:
            if abs(pos - existing_pos) < insert_length:
                overlap = True
                break
        if not overlap:
            positions.append(pos)
    return positions

# 将生成的one-hot序列插入到指定位置
def insert_sequence(original_sequence, insert_sequence, positions):
    new_sequence = original_sequence.copy()
    # for pos in positions:
    if positions + len(insert_sequence) > len(original_sequence):
        raise ValueError("插入位置超出序列长度")
    new_sequence[positions:positions+len(insert_sequence)] = insert_sequence
    return new_sequence

def cross_validation(number_of_folds, total_number, random_seeds=233):
    """
    :param number_of_folds:
    :param total_number:
    :param random_seeds:
    :return:
    """
    x = np.zeros((total_number, ), dtype=np.int32)
    split_iterator = StratifiedKFold(n_splits=number_of_folds, random_state=random_seeds, shuffle=True)
    split_train_index_and_test_index_list = [
        (train_index, test_index)
        for train_index, test_index in split_iterator.split(x,x)
    ]
    return(split_train_index_and_test_index_list)

def split_dataset(split_index_list, fold, data_x, data_y, data_id=None):
    """
    generate training dataset and test data set
    :param split_index_list:
    :param fold:
    :param data_id:
    :param data_x:X
    :param data_y:Y
    :return:
    """
    x_train=data_x[split_index_list[fold][0].tolist()]
    y_train=data_y[split_index_list[fold][0].tolist()]
    x_test=data_x[split_index_list[fold][1].tolist()]
    y_test=data_y[split_index_list[fold][1].tolist()]
    return [x_train, y_train, x_test, y_test]

def StoreTrainSet(rootPath, allData,ValNum=10, RandomSeeds=233):
    """
    store different dataset
    :param rootPath:
    :param ValNum: all data size /test size
    :param RandomSeeds: for generating testing dataset
    :param allData: All data
    """
    dataNum = allData[1].shape[0]
    split_train_index_and_test_index_list = cross_validation(number_of_folds=ValNum, total_number=dataNum, random_seeds=RandomSeeds)
    i=0
    outDataTem = split_dataset(split_train_index_and_test_index_list, fold=i, data_x=allData[0], data_y=allData[1])

    mkdir(rootPath)
    training_path = rootPath + "/train.hdf5"
    test_path = rootPath + "/test.hdf5"

    f_train = h5py.File(training_path,"w")
    f_test = h5py.File(test_path,"w")

    f_train.create_dataset("sequences",data = outDataTem[0])
    f_train.create_dataset("labs",data=outDataTem[1])
    f_train.close()
    f_test.create_dataset("sequences",data=outDataTem[2])
    f_test.create_dataset("labs",data=outDataTem[3])
    f_test.close()

def normalize_list(input_list):
    total = sum(input_list)
    if total == 0:  # 避免除以零的情况
        return [0] * len(input_list)
    return [x / total for x in input_list]

def generate_first_order(first_pro,trans_matrix):
    """
    根据一阶马尔可夫转移矩阵生成DNA序列
    :param trans_matrix: 转移概率矩阵，格式为{'A': {'A': 0.1, 'C': 0.2, ...}, ...}
    :param length: 生成序列长度
    :return: 包含one-hot编码的列表，每个元素是四维列表
    """
    current = np.random.choice(list(first_pro.keys()), p=normalize_list((list(first_pro.values()))))
    sequence = [current]

    # 生成后续状态
    for _ in range(len(trans_matrix)):
        next_probs = trans_matrix[_][current]
        current = np.random.choice(list(next_probs.keys()), p=normalize_list(list(next_probs.values())))
        sequence.append(current)
    return sequence

from collections import defaultdict

def generate_second_order(first_pro,second_pro,trans_matrix):
    """
    根据一阶马尔可夫转移矩阵生成DNA序列
    :param trans_matrix: 转移概率矩阵，格式为{'A': {'A': 0.1, 'C': 0.2, ...}, ...}
    :param length: 生成序列长度
    :return: 包含one-hot编码的列表，每个元素是四维列表
    """
    seq1 = np.random.choice(list(first_pro.keys()), p=list(first_pro.values()))
    second_probs = second_pro[seq1]
    seq2 = np.random.choice(list(second_probs.keys()), p=list(second_probs.values()))
    sequence = [seq1,seq2]

    # 生成后续状态
    for _ in range(len(trans_matrix)):
        next_probs = trans_matrix[_][(sequence[_],sequence[_+1])]
        current = np.random.choice(list(next_probs.keys()), p=list(next_probs.values()))
        sequence.append(current)
    return sequence

from collections import defaultdict

def generate_randomized_transition_matrix(states=['A', 'C', 'G', 'T'],rs=12):
    """
    生成具有增强随机性的二阶马尔可夫转移矩阵
    特点：
    - 主导状态总概率在0.7~0.9随机波动
    - 剩余状态概率非均匀分配
    - 保证所有转移概率严格归一化
    """
    np.random.seed(rs)  # 确保每次运行结果不同
    transition_matrix = defaultdict(dict)
    state_count = len(states)

    for prev1 in states:
        for prev2 in states:
            prev_pair = (prev1, prev2)

            # 随机参数配置
            k = random.choices([1, 2], weights=[0.7, 0.3])[0]  # 主导状态数量
            dominant_prob = np.random.uniform(0.4, 0.6)  # 主导状态总概率

            # 选择主导状态
            dominant_states = random.sample(range(state_count), k)

            # 生成主导状态权重（使用指数分布增强差异性）
            dominant_weights = np.random.exponential(scale=1.0, size=k)
            dominant_weights = dominant_weights / dominant_weights.sum()

            # 生成剩余状态权重（同样加入随机性）
            residual_count = state_count - k
            if residual_count > 0:
                residual_weights = np.random.exponential(scale=0.5, size=residual_count)
                residual_weights = residual_weights / residual_weights.sum() * (1 - dominant_prob)

            # 构建完整概率分布
            probs = np.zeros(state_count)
            for i, state_idx in enumerate(dominant_states):
                probs[state_idx] = dominant_weights[i] * dominant_prob

            # 填充剩余状态概率
            residual_states = [i for i in range(state_count) if i not in dominant_states]
            for i, state_idx in enumerate(residual_states):
                probs[state_idx] = residual_weights[i] if residual_count > 0 else 0

            # 最终归一化处理（防御性编程）
            probs /= probs.sum()

            # 转换为字典格式
            transition_matrix[prev_pair] = {
                states[i]: float(probs[i])
                for i in range(state_count)
            }

    return transition_matrix

# 可视化辅助函数
def print_transition_matrix(matrix, states=['A', 'C', 'G', 'T']):
    """打印转移矩阵的可视化概览"""
    print("二阶马尔可夫转移概率矩阵示例：")
    for i, prev_pair in enumerate(matrix):
        if i >= 3: break  # 仅显示前3个组合
        print(f"\n前序组合 {prev_pair}:")
        for state, prob in matrix[prev_pair].items():
            print(f"  → {state}: {prob:.2f}", end=" | ")
        print()

def main():
    """
    :return:
    """
    args = init_parser()
    # position = args.position
    feature =  args.feature
    seqNum = 100000
    seqLen = 100
    output_dir = f"/Kattn-sim-dev/resources/{feature}.fa"
    first_p = {'A': 0.4, 'C': 0.5, 'G': 0.5, 'T': 0.4}
    # other_order_trans = [
    #     {'A': {'A': 0.5, 'C': 0.5, 'G': 0.0, 'T': 0.5},
    #      'C': {'A': 0.5, 'C': 0.5, 'G': 0.0, 'T': 0.5},
    #      'G': {'A': 0.8, 'C': 0.0, 'G': 0.5, 'T': 0.5},
    #      'T': {'A': 0.8, 'C': 0.8, 'G': 0.8, 'T': 0.0}},
    #     {'A': {'A': 0.8, 'C': 0.7, 'G': 0.1, 'T': 0.9},
    #      'C': {'A': 0.7, 'C': 0.5, 'G': 0.7, 'T': 0.3},
    #      'G': {'A': 0.0, 'C': 0.9, 'G': 0.1, 'T': 0.7},
    #      'T': {'A': 0.7, 'C': 0.1, 'G': 0.8, 'T': 0.9}},
    #     {'A': {'A': 0.5, 'C': 0.4, 'G': 0.5, 'T': 0.0},
    #      'C': {'A': 0.1, 'C': 0.6, 'G': 0.1, 'T': 0.7},
    #      'G': {'A': 0.8, 'C': 0.9, 'G': 0.9, 'T': 0.2},
    #      'T': {'A': 0.7, 'C': 0.8, 'G': 0.8, 'T': 0.1}},
    #     {'A': {'A': 0.5, 'C': 0.0, 'G': 0.5, 'T': 0.6},
    #      'C': {'A': 0.8, 'C': 0.7, 'G': 0.9, 'T': 0.1},
    #      'G': {'A': 0.1, 'C': 0.4, 'G': 0.3, 'T': 0.2},
    #      'T': {'A': 0.8, 'C': 0.6, 'G': 0.0, 'T': 0.8}},
    #     {'A': {'A': 0.0, 'C': 0.3, 'G': 0.7, 'T': 0.8},
    #      'C': {'A': 0.5, 'C': 0.6, 'G': 0.4, 'T': 0.5},
    #      'G': {'A': 0.1, 'C': 0.7, 'G': 0.8, 'T': 0.7},
    #      'T': {'A': 0.4, 'C': 0.5, 'G': 0.6, 'T': 0.0}},
    #     {'A': {'A': 0.7, 'C': 0.3, 'G': 0.6, 'T': 0.7},
    #      'C': {'A': 0.8, 'C': 0.2, 'G': 0.8, 'T': 0.0},
    #      'G': {'A': 0.7, 'C': 0.8, 'G': 0.6, 'T': 0.2},
    #      'T': {'A': 0.4, 'C': 0.1, 'G': 0.6, 'T': 0.5}},
    #     {'A': {'A': 0.5, 'C': 0.5, 'G': 0.6, 'T': 0.4},
    #      'C': {'A': 0.3, 'C': 0.8, 'G': 0.7, 'T': 0.7},
    #      'G': {'A': 0.0, 'C': 0.7, 'G': 0.8, 'T': 0.2},
    #      'T': {'A': 0.4, 'C': 0.6, 'G': 0.5, 'T': 0.1}},
    #     {'A': {'A': 0.5, 'C': 0.5, 'G': 0.5, 'T': 0.0},
    #      'C': {'A': 0.7, 'C': 0.2, 'G': 0.7, 'T': 0.8},
    #      'G': {'A': 0.8, 'C': 0.0, 'G': 0.7, 'T': 0.2},
    #      'T': {'A': 0.6, 'C': 0.5, 'G': 0.5, 'T': 0.0}},
    #     {'A': {'A': 0.9, 'C': 0.8, 'G': 0.9, 'T': 0.0},
    #      'C': {'A': 0.1, 'C': 0.2, 'G': 0.7, 'T': 0.9},
    #      'G': {'A': 0.4, 'C': 0.1, 'G': 0.1, 'T': 0.4},
    #      'T': {'A': 0.8, 'C': 0.1, 'G': 0.6, 'T': 0.2}},
    # ]
    other_order_trans = [
        {'A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'C': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'G': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'T': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}},
        {'A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'C': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'G': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'T': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}},
        {'A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'C': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'G': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'T': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}},
        {'A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'C': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'G': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'T': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}},
        {'A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'C': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'G': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'T': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}},
        {'A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'C': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'G': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'T': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}},
        {'A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'C': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'G': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'T': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}},
        {'A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'C': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'G': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'T': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}},
        {'A': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'C': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'G': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25},
         'T': {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}},
    ]

    second_p = {'A': {'A': 0.2, 'C': 0.8, 'G': 0.0, 'T': 0.0},
                         'C': {'A': 0.4, 'C': 0.1, 'G': 0.1, 'T': 0.4},
                         'G': {'A': 0.8, 'C': 0.0, 'G': 0.0, 'T': 0.2},
                         'T': {'A': 0.0, 'C': 0.3, 'G': 0.5, 'T': 0.2}}
    trans_mat = []
    for _ in range(8):
        trans_mat.append(generate_randomized_transition_matrix(rs=_))

    # load motif
    # Motiflist = LoadMotif(DataDir,randomseeds)
    Motif = SpecialMotif()
    # GeneRate Positive Dataset
    InitMatrix = GenerateRandomMatrix(int(seqNum/2), seqLen)
    # insert_length = len(Motif)  # 插入序列的长度
    # for num in range(InitMatrix.shape[0]):
    #     seq = InitMatrix[num]
    #     insert_positions = generate_insert_positions(len(seq), insert_length, num_positions=1)
    #     if feature == 'F_Markov':
    #         seq_list = generate_first_order(first_p, other_order_trans)
    #         print(seq_list)
    #     elif feature == 'S_Markov':
    #         seq_list = generate_second_order(first_p, second_p, trans_mat)
    #
    #     insert_seq = dna_onehot(length=None, dna_sequence=''.join(seq_list))
    #     final_sequence = insert_sequence(seq, insert_seq, insert_positions[0])
    #     InitMatrix[num] = final_sequence
    seq_pos_matrix_out = InitMatrix
    seq_pos_label_out = np.ones(InitMatrix.shape[0],)

    InitMatrix = GenerateRandomMatrix(int(seqNum/2), seqLen)
    seq_neg_matrix_out = InitMatrix
    seq_neg_label_out = np.zeros(seq_pos_matrix_out.shape[0],)

    with open(output_dir, 'w') as f:
        for index,seq_pos in enumerate(seq_pos_matrix_out):
            seq_ = one_hot_to_dna(seq_pos)
            f.write(f'>seq{index} positive\n')
            f.write(f'{seq_}\n')
        for index,seq_neg in enumerate(seq_neg_matrix_out):
            seq_ = one_hot_to_dna(seq_neg)
            f.write(f'>seq{index} negative\n')
            f.write(f'{seq_}\n')


if __name__ == '__main__':
    main()