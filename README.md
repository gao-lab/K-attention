readme.md

# K-attention: A Biologically Informed Attention Operator for Data-Efficient Omics Pattern Recognition

K-attention is a novel network architecture that effectively utilizes limited omics data by integrating biological priors, achieving performance superior to existing methods. This repository is the official implementation of [](https://arxiv.org/).

## Requirements

To test K-attention:

```setup
conda env create -f ./simu.yml -n kattn-sim

```

To apply K-attention to the RNA-RBP binding prediction task:

```setup
conda env create -f ./rbp.yml -n kattn-rbp

```

## K-attention Operator

For the PyTorch implementation of the K-attention operator, see the KattentionV4 section in ./Kattn-sim-dev/src/kattn/kattention.py.

When using the K-attention operator, you need to pass the following four main parameters:

1.  **channel_size**: Number of channels in the input data.
2.  **kernel_size**: Length of the K-attention kernel.
3.  **num_kernels**: Number of K-attention kernels to use.
4.  **reverse**: Indicates forward or backward interactions.

## Simulated Data Generation

To generate the reverse-complement dataset, run:

```train
python ./Kattn-sim-dev/src/kattn/simus/simugeneration.py

```

To apply K-attention to the RNA-RBP binding prediction task:

```train
python ./Kattn-sim-dev/src/kattn/simus/simu_markov_entropy.py --entropy <target entropy [0-2]> --num <number of dataset>

```

## RBP Data Preparation

We downloaded the original dataset from Sun et al. (available at [https://github.com/kuixu/PrismNet](https://github.com/kuixu/PrismNet)). The downloaded data should be placed in ./Kattention\_aten\_test/external/RBP/

Next, to process the raw data into a form suitable for model input, run:

```train
python ./Kattention_aten_test/external/RBP/code/generateData.py

```

## Training

To train the model for reverse-complement or first-order Markov, run this command:

```train
python ./Kattn-sim-dev/src/run/run_bmk.py --model-type <model> --test-config <data>

```

To train the model for RNA-RBP binding prediction task, run this command:

```train
python -u ./Kattention_aten_test/script/RBP/main.py <data> <K-attention kernel size> <kernel number> <random seed> <model> <optimizer>

```

## Plot the K-attention Kernel

After training is complete, the corresponding K-attention kernel can be plotted:

```eval
python ./Kattn-sim-dev/src/run/Draw_kernel.py --model-type <model> --test-config <data>

```

## Extract High Attention Regions (HARs)

To extract High Attention Regions from the sequence, run:

```eval
python ./Kattention_aten_test/script/RBP/HAR_forward_kdeplot.py

```
