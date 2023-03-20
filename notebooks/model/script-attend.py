#!/usr/bin/env python

#SBATCH --job-name=AugFramework

#SBATCH --error=%x.%j.err
#SBATCH --output=%x.%j.out

#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kurnaz@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil
#SBATCH --gres=gpu:1


import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("/pfs/data5/home/kit/tm/px3192/Time-Series-Data-Augmentation-Framework/")
sys.path.append('/pfs/data5/home/kit/tm/px3192/Time-Series-Data-Augmentation-Framework/notebooks/model/')
sys.path.append("../../")

from experiment import Exp

from dataloaders import data_set,data_dict
import torch
import yaml
import os

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = dotdict()   
# TODO change the path as relative path
args.to_save_path     = "../../saved_models/attend/Run_logs"
args.freq_save_path   = "../../saved_models/attend/Freq_data"
args.window_save_path = "../../saved_models/attend/Sliding_window"
args.root_path        = "../../../datasets"


args.drop_transition  = False
args.datanorm_type    = "standardization" # None ,"standardization", "minmax"


args.batch_size       = 256                                                     
args.shuffle          = True
args.drop_last        = False
args.train_vali_quote = 0.90                                           


# training setting 
args.train_epochs            = 150

args.learning_rate           = 0.001  
args.learning_rate_patience  = 5
args.learning_rate_factor    = 0.1


args.early_stop_patience     = 15

args.use_gpu                 = True if torch.cuda.is_available() else False
args.gpu                     = 0
args.use_multi_gpu           = False

args.optimizer               = "Adam"
args.criterion               = "CrossEntropy"

args.seed                             = 1


args.data_name                        =  sys.argv[1]

args.wavelet_filtering                = False
args.wavelet_filtering_regularization = False
args.wavelet_filtering_finetuning     = False
args.wavelet_filtering_finetuning_percent = 0.3

args.regulatization_tradeoff          = 0.0001
args.number_wavelet_filtering         = 10


args.difference       = sys.argv[2] == "True"
args.filtering        =  sys.argv[2] == "True"
args.magnitude        =  sys.argv[2] == "True"
args.weighted_sampler = False




args.pos_select       = None
args.sensor_select    = None


args.representation_type = "time"
args.exp_mode            = "LOCV"

config_file = open('../../configs/data.yaml', mode='r')
data_config = yaml.load(config_file, Loader=yaml.FullLoader)
config = data_config[args.data_name]

args.root_path       = os.path.join(args.root_path,config["filename"])
args.sampling_freq   = config["sampling_freq"]
args.num_classes     =  config["num_classes"]
window_seconds       = config["window_seconds"]
args.windowsize      =   int(window_seconds * args.sampling_freq) 
args.input_length    =  args.windowsize
# input information
args.c_in            = config["num_channels"] * 3 if args.difference else config["num_channels"]
args.p = {
            'jitter': float(sys.argv[3]),
            'exponential_smoothing': float(sys.argv[3]),
            'moving_average': float(sys.argv[3]),
            'magnitude_scaling': float(sys.argv[3]),
            'magnitude_warp': float(sys.argv[3]),
            'magnitude_shift': float(sys.argv[3]),
            'time_warp': float(sys.argv[3]),
            'window_warp': float(sys.argv[3]),
            'window_slice': float(sys.argv[3]),
            'random_sampling': float(sys.argv[3]),
            'slope_adding': float(sys.argv[3]),
        }
args.mixup_lambda = float(sys.argv[4])
args.max_randaug_cnt = int(sys.argv[5])
args.mixup_p = float(sys.argv[6])

print('DataSet={}-ChannelAug={}-RndAugP={}-Mixup_Alpha={}-MaxRndAugCnt={}-MixupP={}'.format(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]))

if args.wavelet_filtering :
    
    if args.windowsize%2==1:
        N_ds = int(torch.log2(torch.tensor(args.windowsize-1)).floor()) - 2
    else:
        N_ds = int(torch.log2(torch.tensor(args.windowsize)).floor()) - 2

    args.f_in            =  args.number_wavelet_filtering*N_ds+1
else:
    args.f_in            =  1


args.filter_scaling_factor = 0.25
args.model_type            = "attend"


exp = Exp(args)

exp.train()

