Code for [SenSys'24] PieBridge: Fast and Parameter-Efficient On-Device Training via Proxy Networks.

## Project structure

```
PieBridge
|–– datasets/
|   |–– ... # datasets for on-device training
|–– doc/
|   |–– ... # documents
|–– proxynetworks/ # original NN, proxy networks, etc. 
|   |–– models/
|   |   |––jit/
|   |   |   |––...
|   |––pth/
|   |   |––...
|–– res/ # training configs and logs
|   |––train_log/ 
|–– scripts/
|   |––run_e2e.sh
|   |––...
|–– src/
|   |––training.py
|   |––...
```

## Installation

### Hardware configuration

To run the code, you must prepare an edge device with Pytorch support, e.g. Jetson TX2 or Jetson Orin.

By default, we use a Jetson TX2 with 8GB RAM and 16 GB swap size. We attach an extra 900 GB NVME SSD to TX2 since the born disk size is insufficient. 

```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 16G /mnt/16GB.swap
sudo mkswap /mnt/16GB.swap
sudo swapon /mnt/16GB.swap
```

Maximize the power mode of your device to get the highest performance (e.g., GPU frequency).

```bash
# check the current power mode
$ sudo nvpmodel -q
NV Power Mode: MODE_30W
2

# set it to mode 0 (typically the highest)
$ sudo nvpmodel -m 0

# reboot if necessary, and confirm the changes
$ sudo nvpmodel -q
NV Power Mode: MAXN
0
```

### Software environments

The basic environment of our device is 

```
Jetpack 4.5.1 [L4T 32.5.2] with

Distribution: Ubuntu 18.04
Release: 4.9.201-tegra
Python: 3.6.9
CUDA: 10.2.89
```

For the direct dependencies of PieBridge, we recommend you directly using the docker image, or checking the software version details in the docker image and manually installing them.

```bash
docker pull yinwangsong2000/piebridge_ae
```

### Downloading datasets and weights

The datasets and model weights are pre-uploaded in [Google Drive](https://drive.google.com/drive/folders/1aoZ9SorbMS_hEw9stMvm-nMfC6y0dKNG?usp=sharing).
You can download them manually by

```bash
gdown <file-id>
```

and put them in the corresponding path of `./datasets/*` and `./proxynetworks/*`.

We also provide an automated script for downloading in `./scripts/downloading.sh`.

## Running

Run the end-to-end experiments:

```bash
cd PieBridge
bash ./scripts/run_e2e.sh 0
```

Run per-dataset experiments:


```bash
cd PieBridge
bash ./scripts/run_standalone_datasets/run_caltech101.sh 0
```

The results will be shown in `./res/*/log.txt`.

## Artifact evaluation

We provide a docker image of reproducing PieBridge and its baselines on Jetson TX2.

See `doc/ae.pdf` for details.
