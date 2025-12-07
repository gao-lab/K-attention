import pandas as pd
import pdb
# sys.path.append("../../corecode/")
from build import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.colors as colors
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from pathlib import Path
import h5py
from tqdm import tqdm
from statannotations.Annotator import Annotator
import logomaker
from collections import Counter


def mkdir(path):
    """
    Determine if the path exists, if it does not exist, generate this path
    :param path: Path to be generated
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return (False)


def loadData(path, Modeltype=None):
    """
    load data

    :param path:
    :return:
    """

    f_test = h5py.File(path + "/test.hdf5", "r")
    TestX = f_test["sequences"][()]
    TestX2 = f_test["seq_struct"][()]
    TestY = f_test["labs"][()].squeeze()
    TestY = TestY.squeeze()
    f_test.close()

    TestX2 = np.swapaxes(TestX2, 1, 2)
    TestX3 = np.concatenate([TestX, TestX2], axis=2)
    TestX = [TestX, TestX3]

    return [TestX, TestY]


class TorchDataset_multi(Dataset):
    def __init__(self, dataset):
        self.X1 = dataset[0][0].astype("float32")
        self.X2 = dataset[0][1].astype("float32")
        self.Y = dataset[1].astype("float32")

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, i):
        return self.X1[i], self.X2[i], self.Y[i]


def ChangePwmtoInputFormat(pwm):
    """

    """
    output = {"A": [], "C": [], "G": [], "T": []}
    # sortlist = {"A":[],"C":[],"G":[],"T":[]}
    sortlist = ["A", "C", "G", "T"]
    for i in range(pwm.shape[0]):
        ShanoyE = 0
        for m in range(4):
            if pwm[i, m] > 0:
                ShanoyE = ShanoyE - pwm[i, m] * np.log(pwm[i, m]) / np.log(2)
        IC = np.log(4) / np.log(2) - (ShanoyE)
        for j in range(4):
            # output[i].append([sortlist[j], pwm[i,j]*IC])
            output[sortlist[j]].append(pwm[i, j] * IC)
    output = pd.DataFrame(output)
    return output


def softmax(x):
    # 归一化,将权重转换为概率
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def draw_(softmax_seq1, name):
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    pwm_e_pos = ChangePwmtoInputFormat(softmax_seq1.T)
    # pwm_e_pos = pwm_e_pos * corresponding_matrix
    logo_pos = logomaker.Logo(pwm_e_pos, ax=axs)
    logo_pos.style_spines(visible=False)
    logo_pos.ax.set_ylabel('IC')
    logo_pos.ax.set_xlabel('Position')
    axs.set_ylim(-2, 2)
    logo_pos.ax.set_title('DNA Sequence Logo conv')
    plt.savefig(savepath + name)  # 保存图像
    plt.clf()
    plt.close()


def get_dataname():
    pathlist = glob.glob("../../external/RBP/HDF5/*")
    namelist = []
    for path in pathlist:
        name = path.split("/")[-1]
        namelist.append(name)
    return namelist


# 设置 hook 权限
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()  # 将输出保存到字典中

    return hook


def generate_random_dna_sequence(length):
    """生成随机DNA序列"""
    bases = ['A', 'C', 'G', 'T']
    return ''.join(np.random.choice(bases, length))


def dna_to_one_hot(dna_sequence):
    """将DNA序列转换为one-hot编码"""
    one_hot_encoding = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    return [one_hot_encoding[base] for base in dna_sequence]


def convert_one_hot_to_atcg(one_hot_tensor):
    # DNA 碱基的映射
    nucleotides = ['A', 'C', 'G', 'T']
    # 使用 argmax 找到最大值索引，然后将其映射到对应的碱基
    sequences = []
    for i in range(one_hot_tensor.shape[0]):  # 遍历批次
        sequence = ''.join(
            nucleotides[torch.argmax(one_hot_tensor[i, j]).item()] for j in range(one_hot_tensor.shape[1]))
        sequences.append(sequence)
    return sequences


tmp_cmd1 = "grep  '^record' ./log/Train_KNET_plus_seq2_sup3.out > ./log/Train_KNET_plus_seq2_sup3.test"
os.system(tmp_cmd1)

GPUID = '0'
df1 = pd.read_csv('./log/Train_KNET_plus_seq2.test', sep='\t', header=None,
                  names=['cls', 'dataset', 'model', 'kernel_len', 'random_seed', 'AUC', 'loss'])
df2 = pd.read_csv('./log/Train_KNET_plus_seq2_sup1.test', sep='\t', header=None,
                  names=['cls', 'dataset', 'model', 'kernel_len', 'random_seed', 'AUC', 'loss'])
df3 = pd.read_csv('./log/Train_KNET_plus_seq2_sup2.test', sep='\t', header=None,
                  names=['cls', 'dataset', 'model', 'kernel_len', 'random_seed', 'AUC', 'loss'])
df4 = pd.read_csv('./log/Train_KNET_plus_seq2_sup3.test', sep='\t', header=None,
                  names=['cls', 'dataset', 'model', 'kernel_len', 'random_seed', 'AUC', 'loss'])
df = pd.concat([df1, df2, df3, df4]).reset_index(drop=True)

datasetnamelist = list(set(df['dataset'].to_list()))
datasetnamelist = ['ABCF1_K562', 'FMR1_K562']
for dataet_ in tqdm(datasetnamelist, desc="dataet"):
    # print(dataet_)
    # dataet_ = 'C17ORF85_HEK293'
    best_para = df[df['dataset'] == dataet_].sort_values(by='AUC', ascending=False).iloc[0, :][
        ['random_seed', 'kernel_len']]
    random_seed = best_para[0]
    kernel_len = int(best_para[1])
    # best_para = df[df['dataset']==dataet_].sort_values(by='AUC', ascending=False).iloc[0,:][['random_seed','kernel_len']]
    # random_seed = best_para[0]
    # kernel_len = int(best_para[1])
    # random_seed = 3407
    # kernel_len = 16
    model1 = KNET_plus_seq2(5, kernel_len, 128, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = model1.to(device)
    path = f'/Kattention_aten_test/result/RBP/HDF5/{dataet_}/KNET_plus_seq2_L1/'
    modelsave_tem = path + f'model_KernelNum-128kernel_size-{kernel_len}_seed-{random_seed}_opt-adamw.checkpointer.pt'
    new_state = {}
    state_dict = torch.load(modelsave_tem, map_location=torch.device('cpu'))
    for key, value in state_dict.items():
        new_state[key.replace('module.', '')] = value
    model1.load_state_dict(new_state)

    ####################################################
    # 提取线性层的权重
    linear_weights = model1.linear[0].weight.data.cpu().detach().numpy()[0]
    # 使用 argsort 获取从小到大的索引
    sorted_indices = np.argsort(linear_weights)
    # 获取最大的十个数的索引
    top_indices = sorted_indices[-64:]
    DataPath = f'../../external/RBP/HDF5/{dataet_}/'
    data_set = loadData(DataPath)
    # Load dataset
    test_set = data_set
    test_set = TorchDataset_multi(test_set)
    device = 'cuda:' + GPUID

    model1 = model1.to(device)

    test_dataloader1 = DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=False)

    # 实例化模型和字典
    activations = {}
    # 注册 hook
    attention1 = model1.conv1
    attention1.register_forward_hook(get_activation("conv1"))
    attention2 = model1.conv2
    attention2.register_forward_hook(get_activation("conv2"))
    attention3 = model1.conv3
    attention3.register_forward_hook(get_activation("conv3"))
    attention4 = model1.conv4
    attention4.register_forward_hook(get_activation("conv4"))

    Y_test = torch.tensor([])
    Y_pred_o = torch.tensor([])
    Y_pred_p = torch.tensor([])
    seq1_list = []
    seq2_list = []
    pos_list = []
    for _ in range(64):
        pos_list.append([])
        seq1_list.append([])
        seq2_list.append([])

    kn = 38
    with torch.no_grad():
        for X_iter1, X_iter2, Y_test_iter in tqdm(test_dataloader1, desc="Processing TestDataset"):
            X_iter1 = X_iter1.to(device)
            X_iter2 = X_iter2.to(device)
            Y_pred_iter = model1(X_iter1, X_iter2)
            Y_test = torch.concat([Y_test, Y_test_iter])
            Y_pred_iter = torch.reshape(Y_pred_iter, (1,))
            Y_pred_o = torch.concat([Y_pred_o, Y_pred_iter.cpu().detach()])
            if Y_test_iter == 1:
                atten1 = activations["conv1"]
                atten2 = activations["conv2"]
                atten3 = activations["conv3"]
                atten4 = activations["conv4"]
                atten = torch.cat((atten1, atten2, atten3, atten4), dim=1)[0]
                len_ = atten.shape[-1]
                count = 0
                for kn in range(64):
                    n = top_indices[kn]

                    atten_ = atten[n, :, :].view(-1)
                    values, indices = torch.topk(atten_, 1)
                    max_coords = torch.stack((indices // len_, indices % len_), dim=-1).cpu().detach().numpy()[0]
                    pos_list[count].append([max_coords[0], max_coords[1]])
                    seq1 = X_iter1[0][max_coords[0]:max_coords[0] + kernel_len].cpu().detach()
                    seq1_list[kn].append(seq1)
                    seq2 = X_iter1[0][max_coords[1]:max_coords[1] + kernel_len].cpu().detach()
                    seq2_list[kn].append(seq2)
                    count += 1

    for _ in range(64):
        stacked_tensors1 = torch.stack(seq1_list[_])
        stacked_tensors2 = torch.stack(seq2_list[_])
        non_pwm_list = []
        for frag in range(kernel_len):
            # test = stacked_tensors1[:,frag:frag+2,:]
            test = torch.stack((stacked_tensors1[:, frag, :], stacked_tensors2[:, frag, :]), dim=1)
            test_seq = convert_one_hot_to_atcg(test)
            # 统计每个唯一元素的出现次数
            counted_elements = Counter(test_seq)

            # 将计数结果转换为列表并排序
            sorted_elements = sorted(counted_elements.items())
            keys = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT',
                    'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']

            counts_dict = {key: 0 for key in keys}
            for key, value in sorted_elements:
                if key in counts_dict:
                    counts_dict[key] = value  # 更新为新值
            count = [counts_dict[key] for key in counts_dict]
            sum_ = sum(count)
            proportions = torch.tensor([i / sum_ for i in count])
            # non_pwm = np.log([p/(1/16) for p in proportions])
            H = -torch.sum(proportions * torch.log2(proportions + 1e-10))  # 加小的epsilon以避免log(0)
            IC = 4 - H
            test = proportions * IC
            non_pwm_list.append(test)

        plt.figure(figsize=(12, 12))  # 调整整体图形大小
        result1 = torch.sum(stacked_tensors1, dim=0)
        result2 = torch.sum(stacked_tensors2, dim=0)

        # plt.subplot(3, 1, 1)  # 3行1列的第1个子图
        # # 计算每行的总和
        # row_sums = result1.sum(dim=1, keepdim=True)
        # # 计算每个元素占该行总和的比例
        # proportions = result1 / row_sums
        # H = -torch.sum(proportions * torch.log2(proportions + 1e-10), axis=1)  # 加小的epsilon以避免log(0)
        # IC = 2 - H
        # test = proportions * IC.unsqueeze(1)
        # pwm_df = pd.DataFrame(test,columns=['A', 'C', 'G', 'U'])
        # # 绘制 motif logo
        # logo = logomaker.Logo(pwm_df, ax=plt.gca())
        # # 设置图形样式
        # logo.style_spines(visible=False)  # 隐藏边框
        # logo.style_spines(spines=['left', 'bottom'], visible=True)  # 显示左和底部边框
        # logo.style_xticks(rotation=0, fmt='%d')  # 设置 x 轴刻度样式
        # logo.ax.set_ylim([0, 2])
        # logo.ax.set_ylabel('Bits', labelpad=-1)
        #
        # plt.subplot(3, 1, 2)  # 3行1列的第1个子图
        # row_sums = result2.sum(dim=1, keepdim=True)
        # proportions = result2 / row_sums
        # H = -torch.sum(proportions * torch.log2(proportions + 1e-10), axis=1)  # 加小的epsilon以避免log(0)
        # IC = 2 - H
        # test = proportions * IC.unsqueeze(1)
        # pwm_df = pd.DataFrame(test,columns=['A', 'C', 'G', 'U'])
        # logo = logomaker.Logo(pwm_df, ax=plt.gca())
        # # 设置图形样式
        # logo.style_spines(visible=False)  # 隐藏边框
        # logo.style_spines(spines=['left', 'bottom'], visible=True)  # 显示左和底部边框
        # logo.style_xticks(rotation=0, fmt='%d')  # 设置 x 轴刻度样式
        # logo.ax.set_ylim([0, 2])
        # logo.ax.set_ylabel('Bits', labelpad=-1)
        #
        # #  Non-PWN Motif Logo
        # plt.subplot(3, 1, 3)  # 3行1列的第1个子图
        # non_pwm_array = torch.stack(non_pwm_list)
        # # 使用 clamp 将小于 0 的值赋为 0
        # clamped_array = torch.clamp(non_pwm_array, min=0)
        # non_pwm_df = pd.DataFrame(clamped_array,columns=[chr(i) for i in range(ord('A'), ord('P') + 1)])
        # # 绘制 motif logo
        # logo = logomaker.Logo(non_pwm_df, ax=plt.gca())
        # # 设置图形样式
        # logo.style_spines(visible=False)  # 隐藏边框
        # logo.style_spines(spines=['left', 'bottom'], visible=True)  # 显示左和底部边框
        # logo.style_xticks(rotation=0, fmt='%d')  # 设置 x 轴刻度样式
        # logo.ax.set_ylim([0, 4])
        # logo.ax.set_ylabel('Bits', labelpad=-1)
        # # plt.title("Non-PWN Motif Logo", fontsize=14)  # 添加标题
        # savepath = f"./picture/Non-PWN-all-L1/{dataet_}/HAR/"
        # path = Path(savepath)
        # # 判断路径是否存在
        # if path.exists():
        #     pass
        # else:
        #     mkdir(savepath)
        # savepathtmp = savepath + f"{_}({linear_weights[top_indices[_]]}).png"
        # plt.savefig(savepathtmp, bbox_inches='tight')
        # plt.close()
        # plt.clf()

    for i in tqdm(range(len(pos_list)), desc="Processing distance"):

        matrix = np.zeros((len_, len_))
        for coord in pos_list[i]:
            x, y = coord
            matrix[x, y] += 1
        # pdb.set_trace()
        # flipped_matrix = np.flip(matrix, axis=0)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, cmap='YlGnBu', cbar=True)
        # inter_range = list(range(-kernel_len, kernel_len+1))
        # outer_range = [i for i in range(-(len_-1),len_) if i not in inter_range]
        # 翻转 Y 轴
        plt.gca().invert_yaxis()
        inter_sum = 0
        outer_sum = 0
        inter_bg = 0
        outer_bg = 0
        for row in range(len_):
            for col in range(len_):
                if abs(row - col) < kernel_len:
                    inter_sum += matrix[row, col]
                    inter_bg += 1
                else:
                    outer_sum += matrix[row, col]
                    outer_bg += 1
        # plt.title(f'inter:{round(inter_sum/inter_bg,2)}  outer:{round(outer_sum/outer_bg,2)}')
        # 添加主对角线
        plt.plot([0, 0], [matrix.shape[0], matrix.shape[1]], color='black', alpha=0.1, linewidth=1)

        savepath = f"./picture/Non-PWN-all-L1-flip/{dataet_}/pos/"
        path = Path(savepath)
        # 判断路径是否存在
        if path.exists():
            pass
        else:
            mkdir(savepath)
        savepathtmp = savepath + f"{i}({linear_weights[top_indices[i]]}).png"
        plt.savefig(savepathtmp, bbox_inches='tight')
        plt.close()

pdb.set_trace()

letter_ = [chr(i) for i in range(ord('A'), ord('P') + 1)]
for i in range(17):
    print(f'{letter_[i]}:{keys[i]}')

