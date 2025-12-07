# -*- coding: utf-8 -*-
import pdb
import os
import glob
from multiprocessing import Pool
import sys
import glob
import time
import subprocess, re
import numpy as np
import pandas as pd

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

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    best_memory, best_gpu = sorted(memory_gpu_map)[0]

    return best_gpu

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return (True)
    else:
        return False

def run_model(dataset, model, max_lr):
    # selectGPU = "CUDA_VISIBLE_DEVICES=1 "
    # cmd = selectGPU+ "python torch_main.py"
    cmd = f"python run_bmk.py --model-type {model} --test-config {dataset} --max-lr {max_lr}"
    # dataset, model, max_lr

    print(cmd)

    os.system(cmd)

def generateParamPair():

    # datasets = ["markov_1_0","markov_1_25","markov_1_5","markov_1_75",'markov_2_0']
    # datasets1 = ["markov_1_25_5000","markov_1_25_10000","markov_1_25_20000","markov_1_25_50000","markov_1_25_100000"]
    # datasets2 = ["markov_1_5_5000","markov_1_5_10000","markov_1_5_20000","markov_1_5_50000","markov_1_5_100000"]
    # datasets3 = ["markov_1_75_5000","markov_1_75_10000","markov_1_75_20000","markov_1_75_50000","markov_1_75_100000"]
    datasets1 = ["markov_0_75_5000","markov_0_75_20000","markov_0_75_50000","markov_0_75_100000"]
    datasets2 = ["markov_1_0_5000", "markov_1_0_20000", "markov_1_0_50000", "markov_1_0_100000"]
    datasets3 = ["markov_1_25_5000", "markov_1_25_20000", "markov_1_25_50000", "markov_1_25_100000"]
    datasets = datasets1+datasets2+datasets3

    # model_types = ["cnn"]
    model_types = ["transformer_cls","mha"]
    paramlist = []

    for model in model_types:
        if model.startswith("transformer"):
            max_lr = 1e-5
        elif model.startswith("kattn"):
            max_lr = 1e-2
        elif model=="cnn":
            max_lr = 1e-4
        else:
            max_lr = 1e-4  # 默认值，适用于未匹配的情况
        for dataset in datasets:
            param = [dataset, model, max_lr]
            paramlist.append(param)
    return paramlist

if __name__ == '__main__':
    # grid search
    paramlist = generateParamPair()
    cpus = 1
    num = int(len(paramlist)/cpus)+1
    opt = "adamw"

    for i in range(num):

        pool = Pool(processes=cpus)

        for param in paramlist[i*cpus:min((i+1)*cpus,len(paramlist))]:
            dataset, model, max_lr = param
            pool.apply_async(run_model, (str(dataset), str(model), max_lr))
            time.sleep(5)
        pool.close()
        pool.join()
