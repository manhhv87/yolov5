# Real-time detection of crop rows in maize fields based on autonomous extraction of ROI
It is a crop rows detection method using the object detection model.
1. ⚡Super fast: It takes only 25 ms to process a single image (640*360 pixels) and the frame rate of the video stream exceeds 40FPS.
2. 👍Multiple periods: The model is trained on the dataset including various crop row periods.
3. 🤗High accuracy: The average error angle of the detection lines is 1.88◦, which can meet the accuracy requirements of field navigation.

# Installation
## Dependency Setup

Create docker [Pytorch](https://pytorch.org/)>=1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html)>=0.9.0. For example:

[https://hub.docker.com/r/rapidsai/rapidsai/tags](https://hub.docker.com/r/rapidsai/rapidsai/tags)
```
module load singularity
singularity cache clean
singularity pull --name pytorch_1.11.0_py3.9_cuda11.5_cudnn8.3.2_0_ubuntu20.04.sif docker://anibali/pytorch:1.11.0-cuda11.5
```

Create an new conda virtual environment
```
module load /opt/hpc/modulefiles/python/anaconda3
conda create -n mlenv python=3.9 -y
conda init bash
conda activate mlenv

conda config --add channels conda-forge
conda config --show channels
```
Add `PYTHONPATH` environment variable.
```
nano ~/.bashrc
export PYTHONPATH="${PYTHONPATH}:$HOME/.conda/envs/mlenv/lib/python3.9/site-packages/"
source ~/.bashrc
conda activate mlenv
```

Clone this repo and install required packages:
```
git clone https://github.com/manhhv87/ConvNeXt-V2.git --recursive
cd yolov5
pip install -r requirements.txt
python3 -m pip install --no-cache-dir --user opencv-python
python3 -m pip install --no-cache-dir --user opencv-python-headless
conda install git
conda install gitpython=3.1.30
cd ~/miniconda3/envs/aienv/lib/python3.9/site-packages/git 
```

Goto git path:
```
cd ~/miniconda3/envs/aienv/lib/python3.9/site-packages/git
nano __init__.py
```
and modify: ```if not Git.refresh(path=path):``` in function ```def refresh(path: Optional[PathLike] = None) -> None:``` to ```if not Git.refresh(path='~/.conda/envs/mlenv/bin/git'):```

# Preparing dataset
The database is organized as follows:
```
yolov5
  datasets
    images
      img0.jpg
      img1.jpg
      ...
    labels
      img0.txt
      img1.txt
      ...
```

Create `train.txt` and `val.txt`:
```
python creating_train_test_txt_files.py
```

# Training
Create `submit.sh` with content:
```
#! /bin/bash

#SBATCH --job-name=MJ
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --chdir=/home/hanh.buithi/pytorch/yolov5
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --tasks-per-node=1

#SBATCH -o slurm.%N.%J.%u.out # STDOUT
#SBATCH -e slurm.%N.%J.%u.err # STDERR

module load singularity

## Run job
singularity run --nv /home/hanh.buithi/pytorch/pytorch_1.11.0_py3.9_cuda11.5_cudnn8.3.2_0_ubuntu20.04.sif python3 /home/hanh.buithi/pytorch/yolov5/train.py
```

Run bash file
```
sbatch submit.sh
```
# Usage (How to test our model)
Thanks to the contribution, the code is based on https://github.com/ultralytics/yolov5. 

    # 1. Download the trained weights and training log files.
    The trained model is uploaded on https://drive.google.com/file/d/1uca8t8SYReriOtuzo5_RZsCJqb2ggmte/view?usp=sharing. Model and training logs can be obtained after unzipping.
    # 2. Install the requirment.txt: `pip install -r requirements.txt`
    # 3. Run `python detect.py --weights runs/train/exp/weights/best.pt --source test_video/1.mp4`
    We have prepared 5 videos for testing, the root is test_video/*.mp4(avi). Images format has not been accepted yet.
    # 4. Change the test video.
    If you want to change the test video, you have to revise line 300 in detect.py

