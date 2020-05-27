# TrackerSiamRPN 

Thisis a clean PyTorch implementation of SiamRPN tracker. The  implement is adapted from ([siamrpn-pytorch](https://github.com/huanglianghua/siamrpn-pytorch)), which provides a clean testing version for evaluating. 

## Reference
Details regarding the tracking algorithm can be found in the *following* paper:
>[High Performance Visual Tracking with Siamese Region Proposal Network](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf). 
>Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on. IEEE, 2018.

## Description

- TrackerSiamRPN is adapted from a **pytorch** version, along with the training code for the same network.
- The implementation uses GOT-10k for tracking performance evaluation. which ([GOT-10k toolkit](https://github.com/got-10k/toolkit)) is a excellent visual tracking toolkit for VOT evaluation on main tracking datasets.
- TrackerSiamRPN use the Got-10k dataset for training, and evaluate the tracker on OTB\VOT\Got-10k dataset.

## Performance
 
### During Training
![Val_loss trend during training (60 epoches)]()

### Compared with pretrained model
#### OTB Dataset
| Dataset       |  Model  | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|:----------------:|
| OTB2015       | [Provided](https://github.com/huanglianghua/siamrpn-pytorch)      | 0.630            | 0.837 |
| OTB2015       | Epoch_1   | 0.133            | 0.183 |
| OTB2015       | Epoch_45   | 0.325            | 0.633 |
| OTB2015       | Epoch_60   | 0.630            | 0.837 |

#### VOT Dataset

| Dataset       |  Model  | Accuracy    | Robustness (unnormalized) |
|:-----------   |:----------------: |:-----------:|:-------------------------:|
| VOT2019       | [Provided](https://github.com/huanglianghua/siamrpn-pytorch) | 0.571            | 36.576            |
| VOT2019       | Epoch_1 | 0.213            | 139.880            |
| VOT2019       | Epoch_45 | 0.341            | 57.056            |
| VOT2019       | Epoch_60 | 0.571            | 36.576            |

## Dependencies

Install PyTorch, opencv-python and GOT-10k toolkit:

```bash
pip install torch
pip install opencv-python
pip install --upgrade git+https://github.com/got-10k/toolkit.git@master
```

[GOT-10k toolkit](https://github.com/got-10k/toolkit) is a visual tracking toolkit that implements evaluation metrics and tracking pipelines for 7 main datasets (GOT-10k, OTB, VOT, UAV123, NfS, etc.).

## Running the tracker

In the root directory of `siamrpn-pytorch`:

1. Download pretrained `model.pth` from [Baidu Yun](https://pan.baidu.com/s/1QYoQUNraPMUmFW6rp5PDFA) or [Google Drive](https://drive.google.com/open?id=1P0nshF9OjEJwuY9bScuLhPyA2CXSNB5f), and put the file under `pretrained/siamrpn`.

2. Create a symbolic link `data` to your datasets folder (e.g., `data/OTB`, `data/UAV123`, `data/GOT-10k`).

3. Run:

```
python run_tracking.py
```

By default, the tracking experiments will be executed and evaluated over all 7 datasets. Comment lines in `run_tracker.py` as you wish if you need to skip some experiments.
