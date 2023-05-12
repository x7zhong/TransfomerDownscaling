import os
import cv2
import json
from numpy.core.fromnumeric import var
import torch
import numbers
import random
import datetime as dt
import numpy as np
from PIL import Image
import pickle as pkl
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as T
import torch.nn.functional as F
from .transforms import augment, paired_random_crop, paired_fixed_crop
from basicsr.utils import get_root_logger
from basicsr.utils import img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.dist_util import get_dist_info

def add_time(time, hours=0, fmt="%Y%m%d%H"):
    time = dt.datetime.strptime(time, fmt)
    time = time + dt.timedelta(hours=hours)
    return time.strftime(fmt)

def sub_time(time, hours=0, fmt="%Y%m%d%H"):
    time = dt.datetime.strptime(time, fmt)
    time = time - dt.timedelta(hours=hours)
    return time.strftime(fmt)

def read_file(filename, in_memory):
    lines = {}
    idx = 0
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if len(line) < 1:
                break
            if not in_memory:
                lines[idx] = line.strip()
            else:
                radar_data = np.load(line.strip())
                lines[idx] = radar_data
            idx += 1
    return lines

class RadarDataset(Dataset):
    def __init__(self, opt):
        super().__init__()
        radar_file = opt['radar_file']
        train_list_file = opt['train_file']
        self.radar_map = read_file(radar_file, False)
        with open(train_list_file, 'r') as f:
            self.seqs = f.readlines()
            f.close()

@DATASET_REGISTRY.register()
class MergeDataset(RadarDataset):
    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt
        self.dirName_data = opt['dirName_data']
        self.ListofVar = opt['listofVar']
        self.varName_gt = opt['varName_gt']
        self.dirName_stat = opt['dirName_stat']
        self.index_gt = opt['index_gt']
        self.normalize = opt['normalize']
        #high res hgt
        self.hgt_obs = opt.get('hgt_obs', None)
        # TODO, pass param
        self.seq_length = opt.get('seq_length', 1)
        self.interval = 1
        self.logger = get_root_logger()
        self.var_stats_dict = self.init_stat()

    def init_stat(self):
        var_stats_dict = {}
        all_var = self.ListofVar.copy()
        all_var.extend(self.varName_gt.copy())
        if self.hgt_obs:
            all_var.append(self.hgt_obs)
        for varName in all_var:
            var_stats_dict[varName] = {'mean': None, 'var': None, 'min': None, 'max': None}
            pathName_stat = os.path.join(self.dirName_stat, varName.replace("_cut", "").replace('_fix', "") + '_stat.pkl')

            if os.path.isfile(pathName_stat) != 0:
                self.logger.info('Load stat from {}\n'.format(pathName_stat))

                pkl_file = open(pathName_stat, 'rb')
                var_stats_dict[varName] = pkl.load(pkl_file)
                pkl_file.close()

            else:
                self.logger.info('Stat file {} does not exist!\n'.format(pathName_stat))

        return var_stats_dict

    def __len__(self):
        return len(self.seqs)

    def do_normalize(self, data, varName):
        if self.normalize == 'minmax':
            max_temp = self.var_stats_dict[varName]['max'].numpy()
            min_temp = self.var_stats_dict[varName]['min'].numpy()
            data = (data - min_temp) / (max_temp - min_temp)
        elif self.normalize == 'zscore':
            mean_tmp = self.var_stats_dict[varName]['mean'].numpy()
            std_tmp = np.sqrt(self.var_stats_dict[varName]['var'].numpy())
            data = (data - mean_tmp) / std_tmp
        else:
            raise ValueError(f'the normalization operator {self.normalize} is not implemented')
        return data

    def get_var(self, varName, img_f):
        area, base_time, lead_hour = img_f[:-4].split('_')
        if 'fix' in varName:
            var = varName.split('_')[0]
            pathName = os.path.join(self.dirName_data, varName, f'{area}_{var}.npy')
        elif 'obs' in varName:
            gt_file = area + '_' + add_time(base_time, int(lead_hour)) + '.npy'
            pathName = os.path.join(self.dirName_data, varName, gt_file)
        else:
            pathName = os.path.join(self.dirName_data, varName, img_f)
        if not os.path.exists(pathName):
            return np.array([])
        if 'obs' in varName:
            radar_img = np.load(pathName)[:1920, :1920]
        else:
            radar_img = np.load(pathName)[:192, :192]
        radar_img = self.do_normalize(radar_img, varName)
        return radar_img

    def __getitem__(self, ind):
        radar_seq = []
        target_seq = []

        idx = int(self.seqs[ind])
        hgt_obs_data = []
        scale = self.opt['scale']

        img_f = self.radar_map[idx]

        for varName in self.ListofVar:
            if 'fix' in varName:
                continue
            radar_img = self.get_var(varName, img_f)
            radar_seq.append(radar_img)

        time = img_f[:-4]
        for varName in self.varName_gt:
            target_img = self.get_var(varName, img_f)
            target_seq.append(target_img)
        for varName in self.ListofVar:
            if 'fix' in varName:
                cache_name = varName
                if not hasattr(self, cache_name):
                    radar_img = self.get_var(varName, img_f)
                    setattr(self, cache_name, radar_img)
                else:
                    radar_img = getattr(self, cache_name)
                radar_seq.append(radar_img)
        if self.hgt_obs:
            cache_name = self.hgt_obs
            if not hasattr(self, cache_name):
                hgt_obs_data = self.get_var(self.hgt_obs, img_f)
                setattr(self, cache_name, hgt_obs_data)
            else:
                hgt_obs_data = getattr(self, cache_name)

            hgt_obs_data = [hgt_obs_data]

        target_data = np.array(target_seq)
        radar_data = np.array(radar_seq)
        hgt_obs_data = np.array(hgt_obs_data)
        if self.opt['phase'] == 'train' and self.opt.get('randomcrop', False):
            gt_size = self.opt['gt_size']
            #random crop
            img_gt, img_lq = paired_random_crop([target_data, hgt_obs_data], [radar_data], gt_size, scale)
            target_data, hgt_obs_data = img_gt
            radar_data = img_lq[0]
        if 'val' in self.opt['phase'] or 'test' in self.opt['phase']:
            gt_size = self.opt.get('gt_size', target_data.shape[-1])
            img_gt, img_lq = paired_fixed_crop([target_data, hgt_obs_data], [radar_data], gt_size, scale)
            target_data, hgt_obs_data = img_gt
            radar_data = img_lq[0]

        if self.opt['phase'] == 'train' and self.opt.get('use_flip', False) and self.opt.get('use_rot', False):
            target_data, radar_data, hgt_obs_data = \
                augment([target_data, radar_data, hgt_obs_data], self.opt['use_flip'], self.opt['use_rot'])

        target_data = torch.from_numpy(np.ascontiguousarray(target_data))
        radar_data = torch.from_numpy(np.ascontiguousarray(radar_data))
        hgt_obs_data = torch.from_numpy(np.ascontiguousarray(hgt_obs_data))

        info = [time]
        return {'lq': radar_data, 'gt': target_data, 'hgt': hgt_obs_data, 'info': info}