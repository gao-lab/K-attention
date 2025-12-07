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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

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


df2 = pd.read_csv('./log/record_cp.tsv', sep='\t', header=None, names=['cls','dataset','model','random_seed','AUC','loss'])
df1 = pd.read_csv('./log/record_cp_12.tsv', sep='\t', header=None, names=['cls','dataset','model','random_seed','AUC','loss'])
df1['kernel_len'] = '12'
df2['kernel_len'] = '16'
df1 = df1[['dataset','model','AUC','kernel_len','random_seed']]
df2 = df2[['dataset','model','AUC','kernel_len','random_seed']]

df11 = pd.read_csv('./log/Train_KNET_plus_ic_test.tsv', sep='\t', header=None, names=['cls','dataset','model','kernel_len','random_seed','AUC','loss'])
df = pd.concat([df1, df2, df11])

df = df[['dataset','model','AUC','kernel_len','random_seed']].reset_index(drop=True)
dataet_ = 'RBFOX2_HepG2'
# dataet_ = 'NUDT21_HEK293'
# dataet_ = 'QKI_HEK293'
# dataet_ = 'U2AF2_Hela'

GPUID = '0'
best_para = df11[(df11['dataset'] == dataet_) &(df11['model'] == 'KNET_plus_ic')].sort_values(by='AUC', ascending=False).iloc[0,:][['random_seed','kernel_len']]
random_seed = best_para[0]
kernel_len = int(best_para[1])

random_seed = 666
kernel_len = 12

model1 = KNET_plus_ic(5, kernel_len, 128, 1)
path = f'/lustre/grp/gglab/liut/Kattention_aten_test/result/RBP/HDF5/{dataet_}/KNET_plus_ic/'
modelsave_tem = path + f'model_KernelNum-128kernel_size-{kernel_len}_seed-{random_seed}_opt-adamw.checkpointer.pt'
new_state = {}
state_dict = torch.load(modelsave_tem, map_location=torch.device('cpu'))
for key, value in state_dict.items():
    new_state[key.replace('module.', '')] = value
model1.load_state_dict(new_state)

best_para2 = df[(df['dataset'] == dataet_) &(df['model'] == 'Prismnet')].sort_values(by='AUC', ascending=False).iloc[0,:][['random_seed','kernel_len']]
random_seed = best_para2[0]
kernel_len = int(best_para2[1])
model2 = PrismNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = f'/lustre/grp/gglab/liut/Kattention_aten_test/result/RBP/HDF5/{dataet_}/Prismnet/'
modelsave_tem = path + f'model_seed-{random_seed}.checkpointer.pt'
new_state = {}
state_dict = torch.load(modelsave_tem, map_location=torch.device('cpu'))
for key, value in state_dict.items():
    new_state[key.replace('module.', '')] = value
model2.load_state_dict(new_state)
model2 = model2.to(device)

DataPath = f'../../external/RBP/HDF5/{dataet_}/'
data_set = loadData(DataPath)

# Load dataset
test_set = data_set
test_set = TorchDataset_multi(test_set)
test_dataloader = DataLoader(dataset=test_set, batch_size=128, shuffle=False, drop_last=False)

base_dir = './picture/IG/'
os.makedirs(base_dir, exist_ok=True)

model1.to(device).eval()
model2.to(device).eval()

all_labels = []
all_pred1 = []
all_pred2 = []
all_X1 = []        # icshape
all_X2 = []        # RNA one-hot + 1 extra channel, shape [B,101,5]

# ================== 推理 & 收集结果 ==================
with torch.no_grad():
    for X1_iter, X2_iter, Y_test_iter in tqdm(test_dataloader, desc="Accumulating predictions"):
        X1_iter = X1_iter.to(device)
        X2_iter = X2_iter.to(device)
        Y_test_iter = Y_test_iter.to(device).float().view(-1)

        # model1: 已 sigmoid
        prob1 = model1(X1_iter, X2_iter)     # [B] 或 [B,1]
        prob1 = prob1.view(-1)

        # model2: logit -> sigmoid
        logit2 = model2(X2_iter.unsqueeze(1))  # X2: [B,101,5] -> [B,1,101,5]
        logit2 = logit2.view(-1)
        prob2 = torch.sigmoid(logit2)

        pred1 = (prob1 >= 0.5).long()
        pred2 = (prob2 >= 0.5).long()

        all_labels.append(Y_test_iter.detach().cpu())
        all_pred1.append(pred1.detach().cpu())
        all_pred2.append(pred2.detach().cpu())
        all_X1.append(X1_iter.detach().cpu())
        all_X2.append(X2_iter.detach().cpu())

y_true  = torch.cat(all_labels).numpy().astype(int)
y_pred1 = torch.cat(all_pred1).numpy().astype(int)
y_pred2 = torch.cat(all_pred2).numpy().astype(int)
X1_all  = torch.cat(all_X1, dim=0)
X2_all  = torch.cat(all_X2, dim=0)      # [N,101,5]

from sklearn.metrics import recall_score

# 这里默认是二分类，正类标签为 1
recall1 = recall_score(y_true, y_pred1, pos_label=1)
recall2 = recall_score(y_true, y_pred2, pos_label=1)

print(f"Model1 recall: {recall1:.4f}")
print(f"Model2 recall: {recall2:.4f}")

def plot_recall_bar(recall1, recall2, save_path):
    models = ["K-NET-RBP", "PrismNet"]
    recalls = [recall1, recall2]

    x = np.arange(len(models))

    # 颜色从浅到深的蓝色，类似经典文献配色
    colors = ["#3182bd","#9ecae1"]

    fig, ax = plt.subplots(figsize=(4, 4))

    bars = ax.bar(x, recalls, color=colors, width=0.6)

    # y 轴范围 0~1，更直观对比
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Recall", fontsize=11)

    # 美化：去掉顶部和右侧边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 较细的 y 轴网格线（只在 y 方向）
    # ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    # ax.xaxis.grid(False)

    # 在每个柱子顶部标注数值
    for bar, val in zip(bars, recalls):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    # 标题稍微小一点，类似 figure panel 的子标题
    ax.set_title("Recall comparison", fontsize=12, pad=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format="svg")
    plt.close(fig)

# 调用绘图
test_dir = '/lustre/grp/gglab/liut/Kattn-sim-dev/src/draw/Figures/'
recall_plot_path = os.path.join(base_dir, "recall_barplot.pdf")
plot_recall_bar(recall1, recall2, recall_plot_path)
print("Recall barplot saved to:", recall_plot_path)
pdb.set_trace()
# ================== 画混淆矩阵 ==================
from matplotlib.colors import LinearSegmentedColormap

def plot_confmat(y_true, y_pred, save_path, title, labels=None):
    """
    y_true, y_pred: 一维数组/列表
    save_path: 保存路径
    title: 图标题
    labels: 类别名称列表（例如 ['Negative', 'Positive']），不传则用默认标签
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # 自定义从白到蓝的 colormap（小值白，大值深蓝）
    white_blue_cmap = LinearSegmentedColormap.from_list(
        "white_blue",
        ["#ffffff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"]
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(
        cmap=white_blue_cmap,
        values_format="d",
        ax=ax,
        colorbar=False    # 如果想要右侧 colorbar，可以改成 True
    )

    # 美化字体和轴标签
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)

    # 让刻度标签稍微大一点
    ax.tick_params(axis="both", labelsize=9)

    # 让格子线更柔和一点
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, format="svg")
    plt.close(fig)

plot_confmat(
    y_true, y_pred1,
    os.path.join(base_dir, "confusion_matrix_K-NET-RBP.svg"),
    "K-NET-RBP",
    labels=[0, 1]
)
plot_confmat(
    y_true, y_pred2,
    os.path.join(base_dir, "confusion_matrix_PrismNet.svg"),
    "PrismNet",
    labels=[0, 1]
)

print("Confusion matrices saved to", base_dir)

# ================== 选择样本：model1 对、model2 错 ==================
mis_indices = np.where((y_pred1 == y_true) & (y_pred2 != y_true))[0]
print("Num(model2 wrong & model1 correct):", len(mis_indices))

# ================== Integrated Gradients 实现 ==================
def integrated_gradients(forward_fn, x, steps=50, baseline=None):
    x = x.detach()
    if baseline is None:
        baseline = torch.zeros_like(x)
    else:
        baseline = baseline.detach()

    grads_sum = torch.zeros_like(x)

    for i in range(1, steps + 1):
        alpha = float(i) / steps
        x_scaled = baseline + alpha * (x - baseline)
        x_scaled.requires_grad_(True)

        y = forward_fn(x_scaled)

        grad, = torch.autograd.grad(
            outputs=y,
            inputs=x_scaled,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )
        if grad is None:
            grad = torch.zeros_like(x_scaled)

        grads_sum += grad.detach()

    avg_grads = grads_sum / steps
    ig = (x - baseline) * avg_grads
    return ig

# ================== 画两个模型的 IG 图 ==================
from matplotlib.patches import Rectangle
BASES = np.array(["A", "C", "G", "U"])
BASE_COLORS = {
    "A": "green",
    "C": "blue",
    "G": "orange",
    "U": "red",
}

def plot_ig_pretty_four_rows(
    x2, ig,
    label, pred,
    model_name, save_path,
    sample_idx=None
):
    x_np  = x2.squeeze(0).detach().cpu().numpy()      # [L,5]
    ig_np = ig.squeeze(0).detach().cpu().numpy()      # [L,5]
    L = x_np.shape[0]

    onehot   = x_np[:, :4]
    base_idx = onehot.argmax(axis=-1)
    seq_bases = BASES[base_idx]

    seq_ig_mat     = ig_np[:, :4].T           # [4,L]
    icshape_ig_mat = ig_np[:, 4][None, :]     # [1,L]
    icshape_val    = x_np[:, 4]               # [L]

    max_abs = np.max(
        np.abs(np.concatenate([seq_ig_mat.flatten(),
                               icshape_ig_mat.flatten()]))
    )
    if max_abs == 0:
        max_abs = 1e-6

    # 第三行/折线共用纵轴范围
    vmin_val = float(icshape_val.min())
    vmax_val = float(icshape_val.max())
    margin = (vmax_val - vmin_val) * 0.05 if vmax_val > vmin_val else 0.05
    ymin = vmin_val - margin
    ymax = vmax_val + margin

    fig, axes = plt.subplots(
        3, 1,
        figsize=(max(14, L / 3), 6),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1, 4]}
    )

    # ================== 第1行：ACGU IG 热图 ==================
    ax0 = axes[0]
    im_seq = ax0.imshow(
        seq_ig_mat,
        aspect="auto",
        interpolation="nearest",
        cmap="bwr",
        vmin=-max_abs,
        vmax=max_abs
    )
    ax0.set_yticks(range(4))
    ax0.set_yticklabels(BASES)
    ax0.set_ylabel("Seq IG")
    ax0.set_xlim(-0.5, L - 0.5)

    # ================== 第2行：占满行高的碱基字母 ==================
    ax1 = axes[1]
    ax1.set_ylim(0, 1)
    ax1.set_xlim(-0.5, L - 0.5)
    ax1.set_ylabel("Base")

    for i in range(L):
        rect = Rectangle(
            (i - 0.5, 0),
            width=1.0, height=1.0,
            facecolor="white",
            edgecolor="lightgray",
            linewidth=0.4
        )
        ax1.add_patch(rect)

    for i, base in enumerate(seq_bases):
        color = BASE_COLORS.get(base, "black")
        ax1.text(
            i, 0.5, base,
            ha="center", va="center",
            fontsize=18, fontweight="bold",
            color=color
        )

    ax1.set_yticks([])
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # ================== 第3&4行：icSHAPE IG 热图 + 数值折线 ==================
    ax2 = axes[2]
    ax2.imshow(
        icshape_ig_mat,
        aspect="auto",
        interpolation="nearest",
        cmap="bwr",
        vmin=-max_abs,
        vmax=max_abs,
        extent=[-0.5, L - 0.5, ymin, ymax]
    )
    ax2.set_xlim(-0.5, L - 0.5)
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("icSHAPE\nIG + value")
    ax2.set_yticks(np.linspace(ymin, ymax, 3))

    ax2.plot(
        range(L), icshape_val,
        color="black", linewidth=1.0
    )

    # ================== 右侧单独的 colorbar ==================
    # 先用 tight_layout 为主体图预留到 0.9 宽度
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    # 在最右边加一个细长的轴作为 colorbar 容器
    # [left, bottom, width, height]，这里 left=0.92~0.94，完全在图外侧
    cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im_seq, cax=cax)
    cbar.set_label("Integrated Gradients")

    # 标题
    label_str = "positive(1)" if label == 1 else "negative(0)"
    title = f"{model_name} | idx {sample_idx} | true: {label_str}, pred: {pred}"
    fig.suptitle(title, fontsize=10, x=0.45)  # 往左一点，避免压到 colorbar

    # 再微调一次布局（只调主图区域）
    plt.subplots_adjust(right=0.9)

    plt.savefig(save_path, dpi=300)
    plt.close(fig)

N_PLOT = min(10, len(mis_indices))

for rank, idx in enumerate(mis_indices[:N_PLOT]):
    x2 = X2_all[idx:idx+1].to(device)   # [1,101,5]
    x1 = X1_all[idx:idx+1].to(device)
    y  = int(y_true[idx])
    p1 = int(y_pred1[idx])

    def f1(x2_var):
        out = model1(x1, x2_var)        # 注意 X2 在第二个位置（参与计算）
        return out.view(-1)[0]

    ig1 = integrated_gradients(f1, x2, steps=50)    # [1,101,5]

    save_path1 = os.path.join(
        base_dir,
        f"IG_model1_four_rows_idx{idx}.png"
    )

    plot_ig_pretty_four_rows(
        x2, ig1,
        label=y, pred=p1,
        model_name="Model1 KNET_plus_ic",
        save_path=save_path1,
        sample_idx=idx
    )


