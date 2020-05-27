from __future__ import absolute_import, division

import os
import sys
sys.path.append('..')

from got10k.datasets import *

from siamrpn.transforms import SiamRPNTransforms
from siamrpn.datasets import Pair

if __name__ == "__main__":
    root_dir = '/data2/data_zyp/GOT-10k'
    seq_dataset = GOT10k(root_dir, subset='train')
    transforms = SiamRPNTransforms(
            exemplar_sz = 127,
            instance_sz = 271 ,
            context = 0.5)
    dataset_example = Pair(
            seqs=seq_dataset,
            transforms=transforms)
    dataset_example.__getitem__(180)
    dataset_example.__getitem__(280)