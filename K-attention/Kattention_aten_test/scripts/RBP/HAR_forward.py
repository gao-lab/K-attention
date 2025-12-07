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


tmp_cmd1 = "grep  '^record' ./log/Train_KNET_plus_seq2.out > ./log/Train_KNET_plus_seq2.test"
os.system(tmp_cmd1)
tmp_cmd2 = "grep  '^record' ./log/Train_KNET_plus_seq2_sup1.out > ./log/Train_KNET_plus_seq2_sup1.test"
os.system(tmp_cmd2)
tmp_cmd3 = "grep  '^record' ./log/Train_KNET_plus_seq2_sup2.out > ./log/Train_KNET_plus_seq2_sup2.test"
os.system(tmp_cmd3)
tmp_cmd4 = "grep  '^record' ./log/Train_KNET_plus_seq2_sup3.out > ./log/Train_KNET_plus_seq2_sup3.test"
os.system(tmp_cmd4)

dataet_ = 'RBM22_K562'
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
best_para = df[df['dataset'] == dataet_].sort_values(by='AUC', ascending=False).iloc[0, :][
    ['random_seed', 'kernel_len']]
random_seed = best_para[0]
kernel_len = int(best_para[1])

# random_seed = 888
# kernel_len = 12

model1 = KNET_plus_seq2(5, kernel_len, 128, 1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1 = model1.to(device)
path = f'Kattention_aten_test/result/RBP/HDF5/{dataet_}/KNET_plus_seq2/'
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

test_dataloader1 = DataLoader(dataset=test_set, batch_size=256, shuffle=False, drop_last=False)
model1.eval()
with torch.no_grad():
    Y_test = torch.tensor([])
    Y_pred1 = torch.tensor([])
    # Y_pred2 = torch.tensor([])

    for X1_iter, X2_iter, Y_test_iter in tqdm(test_dataloader1, desc="Processing TestDataset"):
        Y_test = torch.concat([Y_test, Y_test_iter])
        X1_iter = X1_iter.to(device)
        X2_iter = X2_iter.to(device)
        Y_pred_iter1 = model1(X1_iter, X2_iter)
        # Y_pred_iter2 = model2(X2_iter.unsqueeze(1))
        try:
            Y_pred1 = torch.concat([Y_pred1, Y_pred_iter1.cpu().detach()])
        except:
            Y_pred_iter1 = Y_pred_iter1.cpu().detach()
            Y_pred_iter1 = torch.reshape(Y_pred_iter1, (1,))
            Y_pred1 = torch.concat([Y_pred1, Y_pred_iter1.cpu().detach()])
        # try:
        #     Y_pred2 = torch.concat([Y_pred2, Y_pred_iter2.cpu().detach()])
        # except:
        #     Y_pred_iter2 = Y_pred_iter2.cpu().detach()
        #     Y_pred_iter2 = torch.reshape(Y_pred_iter2, (1,))
        #     Y_pred2 = torch.concat([Y_pred2, Y_pred_iter2.cpu().detach()])

from sklearn.metrics import f1_score, precision_recall_curve

# 计算精确率、召回率和阈值
precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred1)
# 计算 F1 分数
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
# 找到最佳阈值
best_index = np.argmax(f1_scores)
best_threshold1 = thresholds[best_index]
best_f1_score = f1_scores[best_index]
test_dataloader2 = DataLoader(dataset=test_set, batch_size=1, shuffle=False, drop_last=False)


# 设置 hook 权限
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()  # 将输出保存到字典中

    return hook


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

pos_list = []
seq1_list = []
seq2_list = []
for _ in range(len(top_indices)):
    pos_list.append([])
    seq1_list.append([])
    seq2_list.append([])

with torch.no_grad():
    for X_iter1, X_iter2, Y_test_iter in tqdm(test_dataloader2, desc="Processing TestDataset"):
        X_iter1 = X_iter1.to(device)
        X_iter2 = X_iter2.to(device)
        Y_pred_iter = model1(X_iter1, X_iter2)
        if Y_pred_iter > best_threshold1:
            atten1 = activations["conv1"]
            atten2 = activations["conv2"]
            atten3 = activations["conv3"]
            atten4 = activations["conv4"]
            atten = torch.cat((atten1, atten2, atten3, atten4), dim=1)[0]
            len_ = atten.shape[-1]
            count = 0
            # for n in range(len(atten)):
            #     if n in top_indices:
            for n in top_indices:
                atten_ = atten[n, :, :].view(-1)
                values, indices = torch.topk(atten_, 1)
                max_coords = torch.stack((indices // len_, indices % len_), dim=-1).cpu().detach().numpy()[0]
                seq1 = X_iter1[0][max_coords[0]:max_coords[0] + kernel_len].cpu().detach()
                seq2 = X_iter1[0][max_coords[1]:max_coords[1] + kernel_len].cpu().detach()
                pos_list[count].append([max_coords[0], max_coords[1]])
                seq1_list[count].append(seq1)
                seq2_list[count].append(seq2)
                count += 1


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


for i in tqdm(range(len(pos_list)), desc="Processing distance"):
    stacked_tensors1 = torch.stack(seq1_list[i])
    result1 = torch.sum(stacked_tensors1, dim=0)
    stacked_tensors2 = torch.stack(seq2_list[i])
    result2 = torch.sum(stacked_tensors2, dim=0)

    matrix = np.zeros((len_, len_))
    for coord in pos_list[i]:
        x, y = coord
        matrix[x, y] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap='YlGnBu', cbar=True)
    # inter_range = list(range(-kernel_len, kernel_len+1))
    # outer_range = [i for i in range(-(len_-1),len_) if i not in inter_range]
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
    plt.title(f'inter:{round(inter_sum / inter_bg, 2)}  outer:{round(outer_sum / outer_bg, 2)}')
    # 添加主对角线
    plt.plot([0, matrix.shape[0]], [0, matrix.shape[1]], color='black', alpha=0.1, linewidth=1)

    savepath = f"./picture/HAR_forward/{dataet_}/pos/"
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

