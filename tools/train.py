from __future__ import absolute_import

# import fitlog
import os
import sys
sys.path.append('..')

from got10k.datasets import *

from siamrpn import TrackerSiamRPN

# fitlog.commit(__file__ , fit_msg="这里是SiamRPN训练过程, 师兄改版(siamrpn_master)")

root_dir = '/data2/data_zyp/GOT-10k'
# root_dir = os.path.expanduser('~/data/GOT-10k')
train_seqs = GOT10k(root_dir, subset='train', return_meta=True)
# using val 
val_seqs = GOT10k(root_dir, subset='val', return_meta=True)
tracker = TrackerSiamRPN(fit_log = True)
tracker.train_over(seqs = train_seqs , val_seqs = val_seqs)

# fitlog.finish()