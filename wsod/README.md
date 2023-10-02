# Weakly Supervise Object Detection

## Prerequisites

### Install CUDA

Install Nvidia CUDA

```bash
sudo apt-get install aptitude
sudo aptitude install cuda
```

Verify your CUDA version after installation

```bash
nvidia-smi
```

```bash
Thu Sep 21 21:38:09 2023       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05             Driver Version: 535.104.05   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 3070        On  | 00000000:01:00.0  On |                  N/A |
| 63%   59C    P2             170W / 240W |   7719MiB /  8192MiB |     91%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      1264      G   /usr/lib/xorg/Xorg                          102MiB |
|    0   N/A  N/A     14264      G   /usr/lib/xorg/Xorg                          896MiB |
|    0   N/A  N/A     14391      G   /usr/bin/gnome-shell                        297MiB |
|    0   N/A  N/A     16087      G   ...0336381,12307476151079762674,262144      326MiB |
|    0   N/A  N/A     19463      G   ...sion,SpareRendererForSitePerProcess      311MiB |
|    0   N/A  N/A    249021      G   ...,WinRetrieveSuggestionsOnlyOnDemand      118MiB |
|    0   N/A  N/A   1193213      C   python                                     5632MiB |
+---------------------------------------------------------------------------------------+
```

### Install CUDA Toolkit

Install the CUDA toolkit with the same version of your CUDA version. In this case, because 12.2 cuda-toolkit is not released yet when this project was created so use 12.1 to replace. 

```bash
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```

### Install Conda

#### Windows

```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe
start /wait "" miniconda.exe /S
del miniconda.exe
```

#### macOS

```bash
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

#### Linux

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

### Create Python Virtual Environment

```bash
conda create -n open-mmlab python=3.10 -y
conda activate open-mmlab

# To exit the virtual env
# conda deactivate

# To delete the virtual env
# conda remove --name open-mmlab --all
```

### Install Pytorch

You need to install Pytorch version based on the CUDA version, my CUDA version is 12.2 but there is no pytorch version 12.2 so I use 12.1 here which is also compatible with CUDA 12.2. 

Please change the `121` in the url to the version of the CUDA installed on your machine.

For more information, please see [Pytorch website](https://pytorch.org/get-started/locally/).

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

### Install MMCV

Install mmcv from the source code

```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # package mmcv-full will be installed after this step
cd ..
```

### Install MMDetection

```bash
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
cd ..
```

### Install Other Python Dependencies

```bash
pip install mmengine
pip install pytest
```

### Prepare the Argoverse Dataset

Download the Argoverse dataset from https://drive.google.com/file/d/1st9qW3BeIwQsnR0t8mRpvbsSWIo16ACi/view. Then unzip it and move it data folder, make the directory tree looks like this:

```
wsod
├── data
│   ├── Argoverse
│   │   ├── Argoverse-1.1
│   │   |   ├── tracking
│   │   ├── Argoverse-HD
│   │   |   ├── annotations
```

### Generate Test Set

Argoverse's default test.json file does not contain annotations, so we can't use it to get test results. So we need to move some of the data from the train and val datasets to test to make a new test dataset. You need to do this step before training.

```bash
python tools/argoverse_test_set.py
```

## Train

```bash
python tools/train.py config/wsod/wsddn_faster_rcnn_r50_argoverse.py --auto-scale-lr
```
Please refer to [Test existing models on standard datasets](./docs/en/user_guides/train.md) to get more information.

## Test

Remember to comment `load_from=` in `wsddn_faster_rcnn_r50_argoverse.py` before we start to run testing.

Replace the `work_dirs/wsddn_faster_rcnn_r50_argoverse/epoch_2.pth` with the further trained or fully trained model weight when we have it.

```bash
python tools/test.py configs/wsod/wsddn_faster_rcnn_r50_argoverse.py work_dirs/wsddn_faster_rcnn_r50_argoverse/epoch_2.pth \
--work-dir work_dirs/wsddn_faster_rcnn_r50_argoverse/ \
--show-dir work_dirs/wsddn_faster_rcnn_r50_argoverse/
```

You can find the images with test result plotted in `work_dirs/wsddn_faster_rcnn_r50_argoverse/`

Please refer to [Test existing models on standard datasets](./docs/en/user_guides/test.md) to get more information.
