# Adv3D

This repo is based on [BEVDet](https://github.com/HuangJunJie2017/BEVDet) to perform adversarial attack.

## Get Started
#### Installation and Data Preparation



**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment or use your existing one.

```shell
conda create --name adv3d python=3.10 -y
conda activate adv3d
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.

On GPU platforms:

```shell
conda install pytorch torchvision -c pytorch
```


## Installation

We recommend that users follow our best practices to install MMDetection3D. However, the whole process is highly customizable. See [Customize Installation](#customize-installation) section for more information.

### Best Practices
Assuming that you already have CUDA 11.0 installed, here is a full script for quick installation of MMDetection3D with conda.
Otherwise, you should refer to the step-by-step installation instructions in the next section.

```shell
pip install openmim
mim install mmcv-full
mim install mmdet
mim install mmsegmentation
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
pip install -e .
```

**Step 0.** Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

**Step 1.** Install [MMDetection](https://github.com/open-mmlab/mmdetection).


```shell
pip install mmdet
```

Optionally, you could also build MMDetection from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.24.0  # switch to v2.24.0 branch
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
```

**Step 2.** Install [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

```shell
pip install mmsegmentation
```

Optionally, you could also build MMSegmentation from source in case you want to modify the code:

```shell
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
git checkout v0.20.0  # switch to v0.20.0 branch
pip install -e .  # or "python setup.py develop"
```

**Step 3.** Clone the MMDetection3D repository.

```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
```

**Step 4.** Install build requirements and then install MMDetection3D.

```shell
pip install -v -e .  # or "python setup.py develop"
```




**Step 5.** Prepare nuScenes dataset as introduced in [nuscenes_det.md](docs/en/datasets/nuscenes_det.md) and create the pkl for BEVDet by running:
```shell
python tools/create_data_bevdet.py
```

**Step 5.** Download pretrained bevdet checkpoint and place it into ```./bev_model```


## Train model
```shell
bash tools/dist_command.sh  configs/bevdet/bevdet-r50-target.py  bev_model/bevdet-r50.pth  1  tools/attk_r50.py  --eval bbox
```

## Visualize the predicted result.
```shell
bash tools/dist_command.sh  configs/bevdet/bevdet-r50-target.py  bev_model/bevdet-r50.pth  1  tools/attk_loca_importance_pixelcount.py  --eval bbox
```



## Bibtex
If this work is helpful for your research, please consider citing the following BibTeX entry.
```
@article{li2023adv3d,
  title={Adv3D: Generating 3D Adversarial Examples in Driving Scenarios with NeRF},
  author={Li, Leheng and Lian, Qing and Chen, Ying-Cong},
  journal={arXiv preprint arXiv:2309.01351},
  year={2023}
}
```
