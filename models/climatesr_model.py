import os
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import pickle as pkl
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.metrics import calculate_metric
from Plot.pcolor_map_one import pcolor_map_one_python as pcolor_map_one
from basicsr.utils.dist_util import get_dist_info
from torch import distributed as dist

@MODEL_REGISTRY.register()
class ClimateSRModel(SRModel):
    """Example model based on the SRModel class.

    In this example model, we want to implement a new model that trains with both L1 and L2 loss.

    New defined functions:
        init_training_settings(self)
        feed_data(self, data)
        optimize_parameters(self, current_iter)

    Inherited functions:
        __init__(self, opt)
        setup_optimizers(self)
        test(self)
        dist_validation(self, dataloader, current_iter, tb_logger, save_img)
        nondist_validation(self, dataloader, current_iter, tb_logger, save_img)
        _log_validation_metric_values(self, current_iter, dataset_name, tb_logger)
        get_current_visuals(self)
        save(self, epoch, current_iter)
    """

    def __init__(self, opt):
        super(ClimateSRModel, self).__init__(opt)
        self.common_stat = opt['datasets']['common']
        self.var_stats_dict = self.init_stat()
        if 'psd' in self.opt['val']['metrics']:
            self.psd_handle = calculate_metric({}, self.opt['val']['metrics']['psd'])
        self.amp_training = self.opt.get('fp16', False)
        logger = get_root_logger()
        logger.info(f'Enable amp training: {self.amp_training}')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_training)
        #self.init_aditional_loss()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        if train_opt.get('pixel_multiscale_opt'):
            self.cri_pix_multiscale = build_loss(train_opt['pixel_multiscale_opt']).to(self.device)
        else:
            self.cri_pix_multiscale = None
        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('psd_opt'):
            self.cri_psd = build_loss(train_opt['psd_opt']).to(self.device)
        else:
            self.cri_psd = None
        if train_opt.get('pairwise_opt'):
            self.cri_pairwise = build_loss(train_opt['pairwise_opt']).to(self.device)
        else:
            self.cri_pairwise = None
        if train_opt.get('obs_opt'):
            self.cri_obs = build_loss(train_opt['obs_opt']).to(self.device)
        else:
            self.cri_obs = None
        if train_opt.get('wuv_opt'):
            self.cri_wuv = build_loss(train_opt['wuv_opt']).to(self.device)
        else:
            self.cri_wuv = None

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def init_aditional_loss(self):
        train_opt = self.opt['train']
        if train_opt.get('psd_opt'):
            self.cri_psd = build_loss(train_opt['psd_opt']).to(self.device)
        else:
            self.cri_psd = None
        if train_opt.get('pairwise_opt'):
            self.cri_pairwise = build_loss(train_opt['pairwise_opt']).to(self.device)
        else:
            self.cri_pairwise = None
        if train_opt.get('obs_opt'):
            self.cri_obs = build_loss(train_opt['obs_opt']).to(self.device)
        else:
            self.cri_obs = None
        if train_opt.get('wuv_opt'):
            self.cri_wuv = build_loss(train_opt['wuv_opt']).to(self.device)
        else:
            self.cri_wuv = None

    def init_stat(self):
        logger = get_root_logger()
        var_stats_dict = {}
        all_var = self.common_stat['listofVar'].copy()
        all_var.extend(self.common_stat['varName_gt'].copy())
        for varName in all_var:
            var_stats_dict[varName] = {'mean': None, 'var': None, 'min': None, 'max': None}

            pathName_stat = os.path.join(self.common_stat['dirName_stat'], varName.replace("_cut", "").replace("_fix", "") + '_stat.pkl')

            if os.path.isfile(pathName_stat) != 0:
                logger.info('Load stat from {}\n'.format(pathName_stat))

                pkl_file = open(pathName_stat, 'rb')
                var_stats_dict[varName] = pkl.load(pkl_file)

            else:
                logger.info('Stat file {} does not exist!\n'.format(pathName_stat))

        return var_stats_dict

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        elif optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.amp_training):
            self.output = self.net_g(self.lq)
        self.output = self.output.float()
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        if self.cri_psd:
            l_psd = self.cri_psd(self.output, self.gt)
            l_total += l_psd
            loss_dict['l_psd'] = l_psd

        self.scaler.scale(l_total).backward()
        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def feed_data(self, data):
        images = data['lq']
        self.lq = images.to(self.device)
        if 'gt' in data:
            targets = data['gt']
            self.gt = targets.to(self.device)
        self.info = data['info']

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def _initialize_best_metric_results(self, dataset_name, metric_results):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric in metric_results.keys():
            content = self.opt['val']['metrics'][metric.split('_')[0]]
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                for idx in range(len(self.common_stat['index_gt'])):
                    self.metric_results.update({metric+'_'+self.common_stat['listofVar'][idx]: 0 for metric in self.opt['val']['metrics'].keys()})
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name, self.metric_results)
        rank, world_size = get_dist_info()
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar and rank == 0:
            pbar = tqdm(total=len(dataset), unit='image')

        for didx in range(rank, len(dataset), world_size):
            val_data = dataset[didx]
            for key in val_data.keys():
                if key != 'info':
                    val_data[key].unsqueeze_(0)
            self.feed_data(val_data)
            self.test()

            c, h, w = self.gt.shape[-3:]
            output = self.output.reshape(-1, c, h ,w)
            target = self.gt.reshape(-1, c, h ,w)
            if isinstance(self.lq, dict):
                if 'lq' in self.lq:
                    images = self.lq['lq']
                elif 'lq_seq' in self.lq:
                    images = self.lq['lq_seq'][:, -1]
            else:
                images = self.lq

            image = images[:, self.common_stat['index_gt']]
            image = F.interpolate(
                image, size=output.shape[-2:], mode="bicubic", align_corners=False)

            if self.common_stat['normalize'] == 'minmax':
                for idx in range(len(self.common_stat['index_gt'])):
                    max_temp = self.var_stats_dict[self.common_stat['varName_gt'][idx]]['max']
                    min_temp = self.var_stats_dict[self.common_stat['varName_gt'][idx]]['min']
                    output[:, idx] = output[:, idx] * (max_temp - min_temp) + min_temp
                    target[:, idx] = target[:, idx] * (max_temp - min_temp) + min_temp
                    image[:, idx] = image[:, idx] * (max_temp - min_temp) + min_temp
            elif self.common_stat['normalize'] == 'zscore':
                for idx in range(len(self.common_stat['index_gt'])):
                    mean_tmp = self.var_stats_dict[self.common_stat['varName_gt'][idx]]['mean']
                    std_tmp = torch.sqrt(self.var_stats_dict[self.common_stat['varName_gt'][idx]]['var'])
                    output[:, idx] = output[:, idx] * std_tmp + mean_tmp
                    target[:, idx] = target[:, idx] * std_tmp + mean_tmp
                    image[:, idx] = image[:, idx] * std_tmp + mean_tmp
            else:
                raise ValueError(f'the normalization operator {self.normalize} is not implemented')

            metric_data['img'] = output
            if hasattr(self, 'gt'):
                metric_data['img2'] = target
            # tentative for out of GPU memory
            save_npy = self.opt['val'].get('save_npy', False)
            if save_img or self.opt['val'].get('save_img_train', 0) or save_npy:
                if save_img  or (save_npy and didx < 50) or (current_iter >= self.opt['val'].get('save_img_train', 0) and didx < 10):
                    swh = self.opt['val'].get('save_img_wh', 1000)
                    images = image.squeeze(0).cpu().numpy()[..., :swh,:swh]
                    targets = target[[-1]].squeeze(0).cpu().numpy()[...,:swh,:swh]
                    outputs = output[[-1]].squeeze(0).cpu().numpy()[...,:swh,:swh]
                    file_name = self.info[0]
                    area, base_time, lead_time = file_name.split('_')

                    if save_img or (not self.opt['val'].get('save_npy_onlyoutput', False)):
                        lat_out = np.load(os.path.join(self.common_stat['dirName_data'], 'LAT_fix_cut_obs', f'{area}_LAT.npy'))[:swh,:swh]
                        lon_out = np.load(os.path.join(self.common_stat['dirName_data'], 'LON_fix_cut_obs', f'{area}_LON.npy'))[:swh,:swh]

                    if save_npy:
                        if self.opt['val'].get('save_npy_onlyoutput', False):
                            save_res = outputs
                            if len(save_res) == 1:
                                save_res = save_res.squeeze(0)
                        else:
                            save_res = np.concatenate([lat_out[np.newaxis], lon_out[np.newaxis], images, outputs, targets], axis=0)
                        save_res_path = os.path.join(self.opt['path']['visualization'].replace('visualization', 'npy'), file_name + '.npy')
                        os.makedirs(os.path.dirname(save_res_path), exist_ok=True)
                        np.save(save_res_path, save_res)
                        
                    if save_img:
                        X = []
                        Y = []
                        title_temp = []

                        X.append(lon_out)
                        X.append(lon_out)
                        X.append(lon_out)
                        Y.append(lat_out)
                        Y.append(lat_out)
                        Y.append(lat_out)
                        title_temp.append('Bicubic')
                        title_temp.append('Model')
                        title_temp.append('GT')

                        for idx in range(len(self.common_stat['index_gt'])):
                            Var_plot = []
                            Var_plot.append(images[idx])
                            Var_plot.append(outputs[idx])
                            Var_plot.append(targets[idx])

                            Xlabel = ''
                            Ylabel = ''
                            if save_img:
                                pathName = os.path.join(self.opt['path']['visualization'], self.info[0] + '_' + self.common_stat['varName_gt'][idx] + '.png')
                            elif self.opt['val'].get('save_img_train', 0):
                                pathName = os.path.join(self.opt['path']['visualization'], self.info[0] + '_' + self.common_stat['varName_gt'][idx] + '_' + str(current_iter) + '.png')

                            var_tick = {'WS_10_m_cut_obs': np.arange(0, 14, 2), 'T_2_m_cut_obs': np.arange(280, 325, 5), \
                                        'SP_cut_obs': np.arange(85000, 10**5 + 5000, 5000)}
                            var_label = {'WS_10_m_cut_obs': 'Wind Speed at 10 m', 'T_2_m_cut_obs': 'Temperature at 2 m', \
                                        'SP_cut_obs': 'Surface Pressure'}

                            options = {'proj': 'Mercator', 'Colorbar': True, 'Colorbar_Tick': var_tick[self.common_stat['varName_gt'][idx]], \
                                    'Colorbar_label': var_label[self.common_stat['varName_gt'][idx]]}
                            #options = {'proj': 'Mercator'}
                            pcolor_map_one(X, Y, Var_plot, title_temp, Xlabel, Ylabel, pathName, options)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    for idx in range(len(self.common_stat['index_gt'])):
                        metric_res = calculate_metric(metric_data, opt_)
                        self.metric_results[name+'_'+self.common_stat['listofVar'][idx]] += metric_res[idx]

            if rank == 0 and use_pbar:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {idx+world_size}')
        if use_pbar and rank == 0:
            pbar.close()
        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for name, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()
            else:
                pass  # assume use one gpu in non-dist testing
            if rank == 0:
                for metric in self.metric_results.keys():
                    self.metric_results[metric] /= len(dataset)
                    # update the best metric result
                    self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

@MODEL_REGISTRY.register()
class ClimateSRAddHGTModel(ClimateSRModel):
    def feed_data(self, data):
        hgt = data['hgt'].to(self.device)
        lq = data['lq'].to(self.device)
        self.lq = {'hgt': hgt, 'lq': lq}
        if 'gt' in data:
            targets = data['gt']
            self.gt = targets.to(self.device)
        self.info = data['info']
