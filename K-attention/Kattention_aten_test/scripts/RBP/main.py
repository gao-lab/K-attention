import os
import pdb
import torch
import sys
import h5py
import numpy as np
import subprocess, re
from build import *


# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23

def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""
    memory_gpu_map = []
    for (gpu_id, memory) in gpu_memory_map().items():
        # if gpu_id not in [0, 3, 4]:
        memory_gpu_map.append((memory, gpu_id))

    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def GridSearch(data_set, KernelLen, KernelNum, RandomSeed, Modeltype, path, DataName, GPUID="0",
               opt="RMSprop", batch_size=32, epoch_scheme=1000):
    """
    ["RMSprop", "adam", "adamw", "adadelata", "sgd"]
    :param data_set:
    :param KernelLen:
    :param KernelNum:
    :param RandomSeed:
    :param type:
    :param path:
    :param batch_size:
    :param epoch_scheme:
    :return:
    """
    if Modeltype in ["Prismnet", "Prismnet_seq", 'Prismnet_plus']:
        trainPrismnet(path, data_set, KernelNum, KernelLen,
                      RandomSeed, batch_size, epoch_scheme, DataName, opt=opt, GPUID=GPUID, outputName=Modeltype)
    elif Modeltype in ['KNET_plus_seq', 'KNET_plus_ic']:
        trainKattentionPrismnet(path, data_set, KernelNum, KernelLen,
                                RandomSeed, batch_size, epoch_scheme, DataName, opt=opt, GPUID=GPUID,
                                outputName=Modeltype)


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
    if Modeltype in ["Prismnet", "Prismnet_seq", 'Prismnet_plus']:
        TrainX2 = np.swapaxes(TrainX2, 1, 2)
        TrainX = np.concatenate([TrainX, TrainX2], axis=2)

        TestX2 = np.swapaxes(TestX2, 1, 2)
        TestX = np.concatenate([TestX, TestX2], axis=2)
        TrainX = np.expand_dims(TrainX, axis=1)
        TrainY = np.expand_dims(TrainY, axis=1)
        TestX = np.expand_dims(TestX, axis=1)
        TestY = np.expand_dims(TestY, axis=1)

    if Modeltype in ['KNET_plus_seq', 'KNET_plus_ic']:
        TrainX2 = np.swapaxes(TrainX2, 1, 2)
        TrainX3 = np.concatenate([TrainX, TrainX2], axis=2)
        if Modeltype == "Prismnet_kattention":
            TrainX3 = np.expand_dims(TrainX3, axis=1)
        TrainX = [TrainX, TrainX3]
        if Modeltype == "Prismnet_kattention":
            TrainY = np.expand_dims(TrainY, axis=1)
        TestX2 = np.swapaxes(TestX2, 1, 2)
        TestX3 = np.concatenate([TestX, TestX2], axis=2)
        if Modeltype == "Prismnet_kattention":
            TestX3 = np.expand_dims(TestX3, axis=1)
        TestX = [TestX, TestX3]
        if Modeltype == "Prismnet_kattention":
            TestY = np.expand_dims(TestY, axis=1)

    return [[TrainX, TrainY], [TestX, TestY]]


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


if __name__ == '__main__':
    # 训练时CPU占用会达到500%,加完这行CPU占用变为300%
    torch.set_num_threads(3)
    # GPUID = "0"
    GPUID = str(pick_gpu_lowest_memory())
    # GPUID = str(torch.cuda.current_device())
    # os.environ["CUDA_VISIBLE_DEVICES"] = GPUID
    torch.cuda.set_device(int(GPUID))
    # print("use gpu:", GPUID)
    KernelNum2 = 0

    DataPath = sys.argv[1]
    KernelLen = int(sys.argv[2])
    KernelNum = int(sys.argv[3])
    RandomSeed = int(sys.argv[4])
    Modeltype = sys.argv[5]
    opt = sys.argv[6]
    # os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    data_set = loadData(DataPath, Modeltype)
    # print(f'{DataPath}:{len(data_set[0][0][0][0])}')
    Outpath = DataPath.replace("/external/", "/result/")
    mkdir(Outpath)
    DataName = DataPath.split('/')[-2]
    GridSearch(data_set, KernelLen, KernelNum, RandomSeed,
               Modeltype, Outpath, DataName, GPUID=GPUID, opt=opt, batch_size=64, epoch_scheme=1000)
