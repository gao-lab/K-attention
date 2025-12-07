import pandas as pd
import pdb
# sys.path.append("../../corecode/")
from build import *
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.colors as colors
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from pathlib import Path
import h5py
from tqdm import tqdm


# from statannotations.Annotator import Annotator
# import logomaker

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

    f_train = h5py.File(path + "/train.hdf5", "r")
    TrainX = f_train["sequences"][()]
    TrainX2 = f_train["seq_struct"][()]
    TrainY = f_train["labs"][()]
    TrainY = TrainY.squeeze()
    f_train.close()

    f_test = h5py.File(path + "/test.hdf5", "r")
    TestX = f_test["sequences"][()]
    TestX2 = f_test["seq_struct"][()]
    TestY = f_test["labs"][()].squeeze()
    TestY = TestY.squeeze()
    f_test.close()

    TrainX2 = np.swapaxes(TrainX2, 1, 2)
    TrainX3 = np.concatenate([TrainX, TrainX2], axis=2)
    TrainX = [TrainX, TrainX3]
    TestX2 = np.swapaxes(TestX2, 1, 2)
    TestX3 = np.concatenate([TestX, TestX2], axis=2)
    TestX = [TestX, TestX3]

    return [[TrainX, TrainY], [TestX, TestY]]


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


data = pd.read_csv('./rbp.csv', sep=',')
tmp_cmd1 = "grep  '^record' ./log/Train_KNET_plus_seq2.out > ./log/Train_KNET_plus_seq2.test"
os.system(tmp_cmd1)
tmp_cmd2 = "grep  '^record' ./log/Train_KNET_plus_seq2_sup1.out > ./log/Train_KNET_plus_seq2_sup1.test"
os.system(tmp_cmd2)
tmp_cmd3 = "grep  '^record' ./log/Train_KNET_plus_seq2_sup2.out > ./log/Train_KNET_plus_seq2_sup2.test"
os.system(tmp_cmd3)
tmp_cmd4 = "grep  '^record' ./log/Train_KNET_plus_seq2_sup3.out > ./log/Train_KNET_plus_seq2_sup3.test"
os.system(tmp_cmd4)
datasetnamelist = data[data['KNET_seq-DeepBind'] > 0.05]['dataset'].to_list()
# dataet_ = 'RBM22_K562'
batch_size = 256
epoch_scheme = 1000
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

for dataet_ in datasetnamelist:
    best_para = df[df['dataset'] == dataet_].sort_values(by='AUC', ascending=False).iloc[0, :][
        ['random_seed', 'kernel_len']]
    random_seed = best_para[0]
    kernel_len = int(best_para[1])

    random_seed = random_seed
    modelsave_output_prefix = f'/Kattention_aten_test/result/RBP/HDF5/{dataet_}'
    outputName = 'KNET_plus_seq2_L1'
    number_of_kernel = 128
    kernel_size = kernel_len
    opt = 'adamw'

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    mkdir(modelsave_output_prefix + "/" + outputName)
    mkdir(modelsave_output_prefix.replace("result", "log") + "/" + outputName)

    modelsave_output_filename = modelsave_output_prefix + "/" + outputName + "/model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                                str(kernel_size) + "_seed-" + str(random_seed) + "_opt-" + str(
        opt) + ".pt"
    modellogname = modelsave_output_prefix.replace("result", "log") + "/" + outputName + "/" + "model_KernelNum-" + str(
        number_of_kernel) + "kernel_size-" + \
                   str(kernel_size) + "_seed-" + str(random_seed) + "_opt_" + str(opt)
    tmp_path = modelsave_output_filename.replace("pt", "pkl")
    test_prediction_output = tmp_path.replace("/model_KernelNum-", "/Report_KernelNum-")

    model1 = KNET_plus_seq2(5, kernel_len, 128, 1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = model1.to(device)
    path = f'/Kattention_aten_test/result/RBP/HDF5/{dataet_}/KNET_plus_seq2/'
    modelsave_tem = path + f'model_KernelNum-128kernel_size-{kernel_len}_seed-{random_seed}_opt-adamw.checkpointer.pt'
    new_state = {}
    state_dict = torch.load(modelsave_tem, map_location=torch.device('cpu'))
    for key, value in state_dict.items():
        new_state[key.replace('module.', '')] = value
    model1.load_state_dict(new_state)

    # 冻结前半部分参数
    for param in model1.parameters():
        param.requires_grad = False  # 将前半部分的参数固定
    for param in model1.linear.parameters():
        param.requires_grad = True
    model1 = model1.to(device)
    total_parameters = sum(p.numel() for p in model1.parameters())
    print(f'{total_parameters}')
    sys.stdout.flush()

    # Load dataset
    DataPath = f'../../external/RBP/HDF5/{dataet_}/'
    training_set, test_set = loadData(DataPath)
    train_set, test_set = TorchDataset_multi(training_set), TorchDataset_multi(test_set)
    device = 'cuda:' + GPUID

    BCEloss = nn.BCELoss()
    # BCEloss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
    if os.path.exists(test_prediction_output):
        trained = True
        print("already Trained")
    #     print(test_prediction_output)
    #     return 0,0
    else:
        trained = False
        auc_records = []
        loss_records = []

        training_set_len = len(training_set[1])
        train_set_len = int(training_set_len * 0.8)
        train_set, valid_set = torch.utils.data.random_split(train_set,
                                                             [train_set_len, training_set_len - train_set_len])

        # optimizer = torch.optim.Adadelta(model1.parameters(), lr=1, rho=0.9, eps=1.0e-8)
        # optimizer = optimizerSearch(opt, model1)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model1.parameters()), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=np.sqrt(0.1), patience=4,
                                                                  min_lr=0.0001)
        iterations = 0
        best_loss = 100000
        earlS_num = 0
        l1_lambda = 0.1  # 正则强度

        writer = SummaryWriter(modellogname)

        train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        valid_dataloader = DataLoader(dataset=valid_set, batch_size=batch_size, shuffle=True, drop_last=False)

        for epoch in range(int(epoch_scheme)):
            # 训练
            model1.train()
            for X1_iter, X2_iter, Y_true_iter in train_dataloader:

                X1_iter = X1_iter.to(device)
                X2_iter = X2_iter.to(device)
                Y_true_iter = Y_true_iter.to(device)
                optimizer.zero_grad()
                Y_pred = model1(X1_iter, X2_iter)
                # acc_train_batch = (Y_pred.argmax(dim=1) == Y_true_iter).float().mean().item() # 准确率

                loss = BCEloss(Y_pred, Y_true_iter.float())
                l1_loss = l1_lambda * torch.norm(model1.linear[0].weight, 1)
                total_loss = loss + l1_loss
                total_loss.backward()
                optimizer.step()
                # auc_train_batch = roc_auc_score(Y_true_iter.cpu().detach(), Y_pred.cpu().detach())

                iterations += 1
                if iterations % 64 == 0:
                    loss = loss.item()
                    # print(f'iterations={iterations}, loss={loss}, auc_train_batch={auc_train_batch}')
                    writer.add_scalar('train_batch_loss', loss, iterations)
            # 验证
            model1.eval()
            with torch.no_grad():
                total_loss = 0.0
                tem = 0
                for X_iter1, X_iter2, Y_true_iter in valid_dataloader:
                    X_iter1 = X_iter1.to(device)
                    X_iter2 = X_iter2.to(device)
                    Y_true_iter = Y_true_iter.to(device)
                    Y_pred = model1(X_iter1, X_iter2)
                    loss_iter = BCEloss(Y_pred, Y_true_iter.float())
                    l1_loss = l1_lambda * torch.norm(model1.linear[0].weight, 1)
                    total_loss_ = loss_iter + l1_loss
                    total_loss += total_loss_.item()
                    tem = tem + 1

            lr_scheduler.step(total_loss)
            print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
            print(f'valid: epoch={epoch}, val_loss={total_loss / tem}')
            sys.stdout.flush()
            writer.add_scalar('loss', total_loss, epoch)
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model1.state_dict(), modelsave_output_filename.replace(".pt", ".checkpointer.pt"))
                earlS_num = 0
                print("Save the best model\n")
                sys.stdout.flush()
                # print(model1.markonv.k_weights)
            else:
                earlS_num = earlS_num + 1

            if earlS_num >= 8:
                break

    test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, drop_last=False)

    # 测试
    model1.load_state_dict(
        torch.load(modelsave_output_filename.replace(".pt", ".checkpointer.pt"), map_location=device))
    model1.eval()
    with torch.no_grad():
        Y_test = torch.tensor([])
        Y_pred = torch.tensor([])

        for X_iter1, X_iter2, Y_test_iter in test_dataloader:
            Y_test = torch.concat([Y_test, Y_test_iter])
            X_iter1 = X_iter1.to(device)
            X_iter2 = X_iter2.to(device)
            Y_pred_iter = model1(X_iter1, X_iter2)
            try:
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
            except:
                Y_pred_iter = Y_pred_iter.cpu().detach()
                Y_pred_iter = torch.reshape(Y_pred_iter, (1,))
                Y_pred = torch.concat([Y_pred, Y_pred_iter.cpu().detach()])
                pass

        loss = BCEloss(Y_pred, Y_test.float()).item()
        test_auc = roc_auc_score(Y_test, Y_pred)
        print(f'test: test_auc={test_auc}, loss={loss}')
        # print(f'record\t{DataName}\t{outputName}\t{test_auc}\t{loss}')
        # print(f'record\t{DataName}\t{outputName}\t{kernel_size}\t{random_seed}\t{test_auc}\t{loss}')
        if not trained:
            report_dic = {}
            report_dic["auc"] = auc_records
            report_dic["loss"] = loss_records
            report_dic["test_auc"] = test_auc

            tmp_f = open(test_prediction_output, "wb")
            pickle.dump(np.array(report_dic), tmp_f)
            tmp_f.close()
