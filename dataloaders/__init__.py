from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from dataloaders.augmentation import mixup
from dataloaders.augmentation import RandomAugment
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

#from dataloaders.utils import PrepareWavelets,FiltersExtention
# ----------------- har -----------------
"""
from .dataloader_uci_har import UCI_HAR_DATA
from .dataloader_daphnet_har import Daphnet_HAR_DATA
from .dataloader_pamap2_har import PAMAP2_HAR_DATA
from .dataloader_opportunity_har import Opportunity_HAR_DATA
from .dataloader_wisdm_har import WISDM_HAR_DATA
from .dataloader_dsads_har import DSADS_HAR_DATA
from .dataloader_sho_har import SHO_HAR_DATA
from .dataloader_complexHA_har import ComplexHA_HAR_DATA
from .dataloader_mhealth_har import Mhealth_HAR_DATA
from .dataloader_motion_sense_har import MotionSense_HAR_DATA
from .dataloader_usc_had_har import USC_HAD_HAR_DATA
from .dataloader_skoda_r_har import SkodaR_HAR_DATA
from .dataloader_skoda_l_har import SkodaL_HAR_DATA
from .dataloader_single_chest_har import Single_Chest_HAR_DATA
from .dataloader_HuGaDBV2_har import HuGaDBV2_HAR_DATA
from .dataloader_utd_mhad_w_har import UTD_MHAD_W_HAR_DATA
from .dataloader_utd_mhad_t_har import UTD_MHAD_T_HAR_DATA
from .dataloader_synthetic_har import SYNTHETIC_HAR_DATA
from .dataloader_unimib_har import UNIMIB_HAR_DATA
from .dataloader_hhar_har import HHAR_HAR_DATA
from .dataloader_ward_har import WARD_HAR_DATA
data_dict = {"ucihar" : UCI_HAR_DATA,
             "daphnet" : Daphnet_HAR_DATA,
             "pamap2" : PAMAP2_HAR_DATA,
             "oppo"  : Opportunity_HAR_DATA,
             "wisdm" : WISDM_HAR_DATA,
             "dsads" : DSADS_HAR_DATA,
             "sho" : SHO_HAR_DATA,
             "complexha" : ComplexHA_HAR_DATA,
             "mhealth" : Mhealth_HAR_DATA,
             "motionsense" : MotionSense_HAR_DATA,
             "uschad" : USC_HAD_HAR_DATA,
             "skodar" : SkodaR_HAR_DATA,
             "skodal" : SkodaL_HAR_DATA,
             "singlechest" : Single_Chest_HAR_DATA,
             "hugadbv2" : HuGaDBV2_HAR_DATA,
             "utdmhadw" : UTD_MHAD_W_HAR_DATA,
             "utdmhadt" : UTD_MHAD_T_HAR_DATA,
             "synthetic_1" : SYNTHETIC_HAR_DATA,
             "synthetic_2" : SYNTHETIC_HAR_DATA,
             "synthetic_3" : SYNTHETIC_HAR_DATA,
             "synthetic_4" : SYNTHETIC_HAR_DATA,
             "unimib"   : UNIMIB_HAR_DATA,
             "hhar"     : HHAR_HAR_DATA,
             "ward"     : WARD_HAR_DATA}

"""
from .dataloader_HAPT_har import HAPT_HAR_DATA
from .dataloader_EAR_har  import EAR_HAR_DATA
from .dataloader_RW_har import REAL_WORLD_HAR_DATA
from .dataloader_OPPO_har import Opportunity_HAR_DATA
from .dataloader_PAMAP_har import PAMAP2_HAR_DATA
from .dataloader_SKODAR_har import SkodaR_HAR_DATA

data_dict = {"hapt"  : HAPT_HAR_DATA,
             "ear"   : EAR_HAR_DATA,
			 "oppo"  : Opportunity_HAR_DATA,
             "rw"    : REAL_WORLD_HAR_DATA,
             "pamap2": PAMAP2_HAR_DATA,
             "skodar": SkodaR_HAR_DATA}

class data_set(Dataset):
    def __init__(self, args, dataset, flag):
        """
        args : a dict , In addition to the parameters for building the model, the parameters for reading the data are also in here
        dataset : It should be implmented dataset object, it contarins train_x, train_y, vali_x,vali_y,test_x,test_y
        flag : (str) "train","test","vali"
        """
        self.args = args
        self.flag = flag
        self.load_all = args.load_all
        self.data_x = dataset.normalized_data_x
        self.data_y = dataset.data_y
        # print(self.args)

        if flag in ["train","vali"]:
            self.slidingwindows = dataset.train_slidingwindows
        else:
            self.slidingwindows = dataset.test_slidingwindows
        self.act_weights = dataset.act_weights

        if self.args.representation_type in ["freq","time_freq"]:
            if flag in  ["train","vali"]:
                self.freq_path      = dataset.train_freq_path
                self.freq_file_name = dataset.train_freq_file_name
                if self.load_all :
                    self.data_freq   = dataset.data_freq
            else:
                self.freq_path      = dataset.test_freq_path
                self.freq_file_name = dataset.test_freq_file_name
                self.load_all = False

        if self.flag == "train":
            # load train
            self.window_index =  dataset.train_window_index
            print("Train data number : ", len(self.window_index))


        elif self.flag == "vali":
            # load vali

            self.window_index =  dataset.vali_window_index
            print("Validation data number : ",  len(self.window_index))  


        else:
            # load test
            self.window_index = dataset.test_window_index
            print("Test data number : ", len(self.window_index))  
            
            
        all_labels = list(np.unique(dataset.data_y))
        to_drop = list(dataset.drop_activities)
        label = [item for item in all_labels if item not in to_drop]
        self.nb_classes = len(label)
        assert self.nb_classes==len(dataset.no_drop_activites)

        classes = dataset.no_drop_activites
        self.class_transform = {x: i for i, x in enumerate(classes)}
        self.class_back_transform = {i: x for i, x in enumerate(classes)}
        self.one_hot_encoding = {x: np.asarray([1.0 if i == k else 0 for k in range(len(classes))]) for i, x in enumerate(classes)}
        self.inverse_hot_encoding = lambda c : np.argmax(c)
    
        self.input_length = self.slidingwindows[0][2]-self.slidingwindows[0][1]
        self.channel_in = self.data_x.shape[1]-2


        #if self.args.wavelet_filtering:
        #    SelectedWavelet = PrepareWavelets(K=self.args.number_wavelet_filtering, length=self.args.windowsize)
        #    self.ScaledFilter = FiltersExtention(SelectedWavelet)
        #    if self.args.windowsize%2==1:
        #        self.Filter_ReplicationPad1d = torch.nn.ReplicationPad1d(int((self.args.windowsize-1)/2))
        #    else:
        #        self.Filter_ReplicationPad1d = torch.nn.ReplicationPad1d(int(self.args.windowsize/2))

        if self.flag == "train":
            print("The number of classes is : ", self.nb_classes)
            print("The input_length  is : ", self.input_length)
            print("The channel_in is : ", self.channel_in)


    def __getitem__(self, index):
        #print(index)
        index = self.window_index[index]
        start_index = self.slidingwindows[index][1]
        end_index = self.slidingwindows[index][2]

        rand_idx = self.window_index[np.random.randint(0, len(self.window_index))]
        other_start = self.slidingwindows[rand_idx][1]
        other_end = self.slidingwindows[rand_idx][2]

        if self.args.representation_type == "time":

            if self.args.sample_wise ==True:
                sample_x = np.array(self.data_x.iloc[start_index:end_index, 1:-1].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))))
                other_x = np.array(self.data_x.iloc[other_start:other_end, 1:-1].apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))))
            else:
                sample_x = self.data_x.iloc[start_index:end_index, 1:-1].values
                other_x = self.data_x.iloc[other_start:other_end, 1:-1].values

            sample_y = self.class_transform[self.data_y.iloc[start_index:end_index].mode().loc[0]]
            encoded_y = self.one_hot_encoding[self.data_y.iloc[start_index:end_index].mode().loc[0]]
            
            sample_x = np.expand_dims(sample_x,0)
            # print('shape x', sample_x.shape)
            # print('shape y', sample_y)

            # acc_cols = ['acc_x', 'acc_y', 'acc_z']
            acc_cols = ['acc_x']
            colors = {'acc_x': 'red', 'acc_y': 'green', 'acc_z': 'blue'}
            # augcolor = {'acc_x':}
            wsliced_x = RandomAugment.window_slice(sample_x[0])
            jitter_x = RandomAugment.jitter(sample_x[0], 0.5)
            expsmo_x = RandomAugment.exponential_smoothing(sample_x[0])
            movavg_x = RandomAugment.moving_average(sample_x[0])
            magscl_x = RandomAugment.magnitude_scaling(sample_x[0])
            magwrp_x = RandomAugment.magnitude_warp(sample_x[0]) 
            magshf_x = RandomAugment.magnitude_shift(sample_x[0])
            tmewrp_x = RandomAugment.time_warp(sample_x[0]) 
            wndwrp_x = RandomAugment.window_warp(sample_x[0])
            wndslc_x = RandomAugment.window_slice(sample_x[0]) 
            rndsmp_x = RandomAugment.random_sampling(sample_x[0]) 
            slpadd_x = RandomAugment.slope_adding(sample_x[0])
            for idx,col in enumerate(acc_cols):
                plt.figure(figsize=(10,5))
                plt.plot(sample_x[0][:,idx],c = 'blue', linewidth=2, label=col,)
                # plt.plot(jitter_x[:,idx], c = 'red', linewidth=2, label='jitter_' + col, )
                # plt.plot(expsmo_x[:,idx], c = 'red', linewidth=2, label='expsmo_' + col, )
                # plt.plot(movavg_x[:,idx], c = 'red', linewidth=2, label='movavg_' + col, )
                # plt.plot(magscl_x[:,idx], c = 'red', linewidth=2, label='magscl_' + col, )
                # plt.plot(magwrp_x[:,idx], c = 'red', linewidth=2, label='magwrp_' + col, )
                # plt.plot(magshf_x[:,idx], c = 'red', linewidth=2, label='magshf_' + col, )
                # plt.plot(tmewrp_x[:,idx], c = 'red', linewidth=2, label='tmewrp_' + col, )
                # plt.plot(wndwrp_x[:,idx], c = 'red', linewidth=2, label='wndwrp_' + col, )
                # plt.plot(wndslc_x[:,idx], c = 'red', linewidth=2, label='wndscl_' + col, )
                # plt.plot(rndsmp_x[:,idx], c = 'red', linewidth=2, label='rndsmp_' + col, )
                # plt.plot(slpadd_x[:,idx], c = 'red', linewidth=2, label='slpadd_' + col, )
                plt.yticks(np.arange(0, 2, step=0.5)) 
                plt.legend(fontsize=14)
                plt.tight_layout()
                plt.xticks(fontsize=12)
                plt.yticks(fontsize=12)
               
                # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.5))
                # plt.show()
                plt.savefig(r'C:\Users\Murat\Desktop\Bachelor\augplots\original.svg', dpi=500)
                # plt.savefig(r'C:\Users\Murat\Desktop\Bachelor\augplots\expsmo.svg', dpi=500)

                
            return (-1, -1), -1, -1

            if self.flag != 'train':
                return sample_x, encoded_y, encoded_y
            
            other_encoded_y = self.one_hot_encoding[self.data_y.iloc[other_start:other_end].mode().loc[0]]
            
            mixup_x = mixup(sample_x, other_x, self.args.mixup_lambda)
            mixup_x = np.expand_dims(mixup_x, 0)
            mixup_y = mixup(encoded_y, other_encoded_y, self.args.mixup_lambda)

            aug_count = np.random.randint(0, self.args.max_randaug_cnt)
            # print('aug_count', aug_count)
            randaug = RandomAugment(aug_count, self.args.p)
            aug_sample_x = randaug(sample_x[0])
            
            # return (sample_x, aug_sample_x), sample_y, sample_y
            print('aug', aug_sample_x.shape)
            return aug_sample_x, mixup_y, mixup_y
            

        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.window_index)

