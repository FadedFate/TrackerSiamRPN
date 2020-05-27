from __future__ import absolute_import

import os
import sys
sys.path.append('..')

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from got10k.datasets import *
from got10k.experiments import *

from siamrpn import TrackerSiamRPN

# fitlog.commit(__file__ , fit_msg="这里是SiamRPN测试过程, 师兄改版(siamrpn_master)")

if __name__ == '__main__':
    net_path = 'pretrained/siamrpn_e'
    iters = 1
    tracker = TrackerSiamRPN(net_path=net_path + str(iters) + ".pth" , fit_log = True)

    experiments = [
        ExperimentGOT10k('/data2/data_zyp/GOT-10k', subset='test'),
        ExperimentOTB('/data2/data_zyp/OTB', version=2015),
        ExperimentVOT('/data/zyp/data/VOT2019', version=2019),
        # ExperimentGOT10k('data/GOT-10k', subset='test'),
        # ExperimentOTB('data/OTB', version=2015),
        # ExperimentOTB('data/OTB', version=2013),
        # ExperimentVOT('data/vot2018', version=2018),
        # ExperimentUAV123('data/UAV123', version='UAV123'),
        # ExperimentUAV123('data/UAV123', version='UAV20L'),
        # ExperimentDTB70('data/DTB70'),
        # ExperimentTColor128('data/Temple-color-128'),
        # ExperimentNfS('data/nfs', fps=30),
        # ExperimentNfS('data/nfs', fps=240),
    ]

    # run experiments
    for e in experiments:
        e.run(tracker, visualize=True)
        e.report([tracker.name])
