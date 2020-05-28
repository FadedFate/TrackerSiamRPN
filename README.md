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
![Val_loss trend during training (60 epoches)](https://github.com/FadedFate/TrackerSiamRPN/blob/master/traing.JPG)

### Compared with pretrained model
#### OTB Dataset
| Dataset       |  Model  | Success Score    | Precision Score |
|:-----------   |:----------------:|:----------------:|:----------------:|
| OTB2015       | [Provided](https://github.com/huanglianghua/siamrpn-pytorch)      | 0.630            | 0.837 |
| OTB2015       | Epoch_1   | 0.133            | 0.183 |
| OTB2015       | Epoch_45   | 0.325            | 0.633 |
| OTB2015       | Epoch_60   | 0.329            | 0.642 |

#### VOT Dataset

| Dataset       |  Model  | Accuracy    | Robustness (unnormalized) |
|:-----------   |:----------------: |:-----------:|:-------------------------:|
| VOT2019       | [Provided](https://github.com/huanglianghua/siamrpn-pytorch) | 0.571            | 36.576            |
| VOT2019       | Epoch_1 | 0.213            | 139.880            |
| VOT2019       | Epoch_45 | 0.341            | 57.056            |
| VOT2019       | Epoch_60 | 0.343          | 54.855           |

## Update

 - [ ] Accuracy seems unreasonable low compared with baseline, I will check it again later.
 - [ ] Provide the pretrained model for convenient.
