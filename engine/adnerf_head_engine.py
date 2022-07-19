import torch
import torch.nn.functional as F
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
from typing import List
from visdom import Visdom
from data.utils import trivial_collate
import os 
from .utils import Stats, visualize_nerf_outputs
import pickle
from PIL import Image
import importlib
from .basic_engine import NerfEngine as BasicEngine
log = logging.getLogger(__name__)

class NerfEngine(BasicEngine):
    def __init__(self, cfg):
        self.cfg = cfg
        
        # specifying device
        if not torch.cuda.is_available() and cfg.device == 'cuda':
            log.info('Specifying expreriment on GPU while GPU is not available')
            raise Exception
        
        self.device = cfg.device
        log.info('Specify {} as the device.'.format(self.device))

    def _load_dataset_test(self,
                           datasets:object):
        self.test_datasets = datasets.test
        if self.cfg.test.mode == 'evaluation' :
            log.info('Loading test datasets for evaluations..')
                
        if self.cfg.test.mode == 'export_video':
            log.info('Loading test datasets for visualization..')

            # store the video in director
            self.export_dir = 'visualization'
            os.makedirs(self.export_dir, exist_ok = True)
        self.test_dataloaders = []
        for dataset in self.test_datasets:
            self.test_dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=getattr(importlib.import_module('data.utils'), self.cfg.data.dataloader.collate_fn)))

    def forward_audio(self, 
                      audio: torch.Tensor, 
                      idx: int, 
                      dataset: torch.utils.data.Dataset):
        """
        A warper of audio feature extraction process
        
        Args:
            audio (torch.Tensor): the Deepspeech feature of the audio on the specific time stamp
            idx (int): the time idx of the dataframe on the whole dataset
            dataset (Dataset): the dataset where dataframes are fetch from

        Returns:
            aud_para (torch.Tensor): the extracted tensor features, 
                either of the timestamp only or collected from neighboring timestamp
        """
        max_len = len(dataset)        
        # Prepare audio features
        if self.curr_epoch > self.cfg.nosmo_epoches:
            # the trick in AD-Nerf 
            smo_half_win = int(self.cfg.smo_size / 2)
            left_idx = idx - smo_half_win
            right_idx = idx + smo_half_win
            pad_left, pad_right = 0, 0
            if left_idx < 0:
                pad_left = - left_idx
                left_idx = 0
            if right_idx > max_len:
                pad_right =  right_idx - max_len
                right_idx = max_len
            auds_win = torch.cat(
                [dataset.__getitem__(idx)['audio'][None] for idx in range(left_idx, right_idx)],
                dim = 0)
            if pad_left > 0:
                auds_win = torch.cat(
                    [torch.zeros_like(auds_win)[:pad_left], auds_win],
                    dim = 0)
            if pad_right > 0: 
                auds_win = torch.cat(
                    [auds_win, torch.zeros_like(auds_win)[:pad_right]])
            auds_win = self.audionet(auds_win.to(self.device))
            aud_para = self.audioattnet(auds_win)
        else:
            aud_para = self.audionet(audio.unsqueeze(0).to(self.device))
        return aud_para

    def train_epoch(self, ):
        self.model.train()
        self.audionet.train()
        self.audioattnet.train()
        
        self.stats.new_epoch()
        
        for num_dataset, dataloader in enumerate(self.train_dataloaders):
            # choose to fetch data via Dataset rather than Dataloader, because of the following operations.
            dataset = dataloader.dataset
            bg_image = dataset.meta['bg_image'].to(self.device)
            max_len = len(dataset)        
            for iteration in range(max_len):
                idx = np.random.choice(max_len)
                
                self.optim.zero_grad()
                self.optim_aud.zero_grad()
                self.optim_audatt.zero_grad()
                
                # Unpack values
                batch = dataset.__getitem__(idx)                

                image, parsing, camera, audio, sample_rect, camera_idx = batch.values()
                
                if self.cfg.white_bg:
                    face_mask = ((parsing[..., 0] == 0) * (parsing[..., 1] == 0) * (parsing[...,2] == 1) * 1.0).unsqueeze(-1)
                    image = image * face_mask + torch.ones_like(image) * ( 1 - face_mask)
                    bg_image = torch.ones_like(bg_image)

                # extract audio feature
                aud_para = self.forward_audio(audio, idx, dataset)
                # Run the forward pass of the model
                other_params = {
                    'camera_hash':  camera_idx if self.cfg.precache_rays else None, 
                    'camera': camera.to(self.device),
                    'image': image.to(self.device)}  # arguments that original RadianceFieldRenderer in vanillar Nerf used
                nerf_out, metrics = self.model(
                    aud_para = aud_para,
                    rect = sample_rect.to(self.device),
                    bg_image = bg_image,
                    **other_params)

                # The loss is a sum of coarse and fine MSEs
                loss = metrics["mse_coarse"] + metrics["mse_fine"]

                # Take the training step
                loss.backward()
                self.optim.step()
                self.optim_aud.step()
                if self.curr_epoch > self.cfg.nosmo_epoches:
                    self.optim_audatt.step()
                
                # Update stats with the current metrics.
                self.stats.update(
                    {'loss': float(loss), **metrics},
                    stat_set= 'train')
            
                if iteration % self.cfg.stats_print_interval == 0:
                    self.stats.print(stat_set="train")
            
                # Update the visualization cache.
                if self.viz is not None:
                    self.visuals_cache.append({
                        "camera": camera.cpu(),
                        "camera_idx":camera_idx,
                        "image":image.cpu().detach(),
                        "rgb_fine": nerf_out["rgb_fine"].cpu().detach(),
                        "rgb_coarse": nerf_out["rgb_coarse"].cpu().detach(),
                        "rgb_gt": nerf_out["rgb_gt"].cpu().detach(),
                        "coarse_ray_bundle": nerf_out["coarse_ray_bundle"]})

            log.info('Training done on {} datasets'.format(num_dataset))

    def videos_synthesis(self, ):
        self.curr_epoch = self.stats.epoch
        self.model.eval()
        self.audionet.eval()
        self.audioattnet.eval()

        frame_paths = []
        for num_dataset, test_dataloader in enumerate(self.test_dataloaders):
            
            # choose to fetch data via Dataset rather than Dataloader, because of the following operations.
            dataset = test_dataloader.dataset
            bg_image = dataset.meta['bg_image'].to(self.device)
            max_len = len(dataset)

            for iteration in range(max_len):
                idx = iteration
                
                # Unpack values
                test_batch = dataset.__getitem__(idx)
                test_image, parsing, test_camera, test_audio, test_sample_rect, test_camera_idx = test_batch.values()
                
                if self.cfg.white_bg:
                    face_mask = ((parsing[..., 0] == 0) * (parsing[..., 1] == 0) * (parsing[...,2] == 1) * 1.0).unsqueeze(-1)
                    test_image = test_image * face_mask + torch.ones_like(test_image) * ( 1 - face_mask)
                    bg_image = torch.ones_like(bg_image)

                # Activate eval mode of the model (lets us do a full rendering pass).
                with torch.no_grad():
                    
                    if test_image is not None:
                        test_image = test_image.to(self.device)

                    # extract audio feature
                    aud_para = self.forward_audio(test_audio, idx, dataset)
                    # Run the foward pass of the model
                    other_params = {
                        'camera_hash':  test_camera_idx if self.cfg.precache_rays else None, 
                        'camera': test_camera.to(self.device),
                        'image': test_image}  # arguments that original RadianceFieldRenderer in vanillar Nerf used
                    
                    test_nerf_out, test_metrics = self.model(
                        aud_para = aud_para,
                        rect = test_sample_rect.to(self.device),
                        bg_image = bg_image, 
                        **other_params)
                
                # Writing images
                frame = test_nerf_out["rgb_fine"][0].detach().cpu()
                frame_path = os.path.join(self.export_dir, f"scene_{num_dataset:01d}_frame_{iteration:05d}.png")
                log.info(f"Writing {frame_path}")
                tensor2np = lambda x: (x.detach().cpu().numpy() * 255.0).astype(np.uint8)
                if self.cfg.test.with_gt:
                    Image.fromarray(
                        np.hstack([
                            tensor2np(test_image),
                            tensor2np(frame)])
                        ).save(frame_path)
                else:
                    Image.fromarray(tensor2np(frame)).save(frame_path)
                frame_paths.append(frame_path)
                                
        # Convert the exported frames to a video
        video_path = os.path.join(os.getcwd(), "video.mp4")
        ffmpeg_bin = "ffmpeg"
        frame_regexp = os.path.join(os.getcwd(), self.export_dir,"*.png" )
        ffmcmd = (
            "%s -r %d -pattern_type glob -i '%s' -f mp4 -y -b:v 2000k -pix_fmt yuv420p %s"
            %(ffmpeg_bin, self.cfg.test.fps, frame_regexp, video_path))
        log.info('Video gnerated via {} \n {}'.format(ffmpeg_bin, ffmcmd))
        ret = os.system(ffmcmd)
        if ret != 0:
            raise RuntimeError("ffmpeg failed!")   

    def evaluate_full(self, ):
        stats = Stats(["mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "ssim_coarse", "ssim_fine", "lpips_coarse", "lpips_fine", "sec/it"])
        stats.new_epoch()
        self.curr_epoch = self.stats.epoch
        self.model.eval()
        self.audionet.eval()
        self.audioattnet.eval()
        
        for num_dataset, test_dataloader in enumerate(self.test_dataloaders):

            # choose to fetch data via Dataset rather than Dataloader, because of the following operations.
            dataset = test_dataloader.dataset
            bg_image = dataset.meta['bg_image'].to(self.device)
            max_len = len(dataset)
            
            for iteration in range(max_len):
                idx = iteration
                
                # Unpack values
                test_batch = dataset.__getitem__(idx)
                test_image, parsing, test_camera, test_audio, test_sample_rect, test_camera_idx = test_batch.values()

                if self.cfg.white_bg:
                    face_mask = ((parsing[..., 0] == 0) * (parsing[..., 1] == 0) * (parsing[...,2] == 1) * 1.0).unsqueeze(-1)
                    test_image = test_image * face_mask + torch.ones_like(test_image) * ( 1 - face_mask)
                    bg_image = torch.ones_like(bg_image)

                # Activate eval mode of the model (lets us do a full rendering pass).
                with torch.no_grad():
                    
                    if test_image is not None:
                        test_image = test_image.to(self.device)

                    # extract audio feature
                    aud_para = self.forward_audio(test_audio, idx, dataset)
                    # Run the foward pass of the model
                    other_params = {
                        'camera_hash':  test_camera_idx if self.cfg.precache_rays else None, 
                        'camera': test_camera.to(self.device),
                        'image': test_image}  # arguments that original RadianceFieldRenderer in vanillar Nerf used
                    
                    test_nerf_out, test_metrics = self.model(
                        aud_para = aud_para,
                        rect = test_sample_rect.to(self.device),
                        bg_image = bg_image, 
                        **other_params)
                
                # Update other time consuming evaluation results
                extra_metrics = self.image_metrics(
                    image = test_image, 
                    rgb_fine = test_nerf_out['rgb_fine'], 
                    rgb_coarse = test_nerf_out['rgb_coarse'])
                test_metrics.update(extra_metrics)
                
                stats.update(test_metrics, stat_set='test')
                stats.print(stat_set='test')
            
        log.info("Final evaluation metrics on '{}'".format(self.cfg.data.dataset_name))
        for stat in ["mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "ssim_coarse", "ssim_fine", "lpips_coarse", "lpips_fine", ]:
            stat_value = stats.stats['test'][stat].get_epoch_averages()[0]
            log.info(f"{stat:15s}:{stat_value:1.4f}")
    
    def val_epoch(self, ):

        self.model.eval()
        self.audionet.eval()
        self.audioattnet.eval()
        
        for num_dataset, dataloader in enumerate(self.val_dataloaders):

            # choose to fetch data via Dataset rather than Dataloader, because of the following operations.
            dataset = dataloader.dataset
            bg_image = dataset.meta['bg_image'].to(self.device)
            max_len = len(dataset)   

            for iteration in range(max_len):
                idx = iteration
                
                # Unpack values
                batch = dataset.__getitem__(idx)                
                image, parsing, camera, audio, sample_rect, camera_idx = batch.values()               

                if self.cfg.white_bg:
                    face_mask = ((parsing[..., 0] == 0) * (parsing[..., 1] == 0) * (parsing[...,2] == 1) * 1.0).unsqueeze(-1)
                    image = image * face_mask + torch.ones_like(image) * ( 1 - face_mask)
                    bg_image = torch.ones_like(bg_image)

                # Activate eval mode of the model (lets us do a full rendering pass).
                with torch.no_grad():
                    
                    if image is not None:
                        image = image.to(self.device)
                    
                    # extract audio feature
                    aud_para = self.forward_audio(audio, idx, dataset)
                    # Run the forward pass of the model
                    other_params = {
                        'camera_hash':  camera_idx if self.cfg.precache_rays else None, 
                        'camera': camera.to(self.device),
                        'image': image}  # arguments that original RadianceFieldRenderer in vanillar Nerf used
                    val_nerf_out, val_metrics = self.model(
                        aud_para = aud_para,
                        rect = sample_rect.to(self.device),
                        bg_image = bg_image, 
                        **other_params)

                # Update stats with the validation metrics.  
                self.stats.update(val_metrics, stat_set="val")
                
                # if validate only on one batch of each dataloader
                if self.cfg.validation.one_iter: 
                    break

            log.info('Validation done on {} datasets'.format(num_dataset + 1))
            
        self.stats.print(stat_set='val')
        
        if self.viz is not None:
            # Plot that loss curves into visdom.
            self.stats.plot_stats(
                viz = self.viz, 
                visdom_env = self.cfg.visualization.visdom_env,
                plot_file = None)
            log.info('Loss curve ploted in visdom')
            
            # Visualize the intermediate results.
            visualize_nerf_outputs(
                nerf_out = val_nerf_out, 
                output_cache = self.visuals_cache,
                viz = self.viz, 
                visdom_env = self.cfg.visualization.visdom_env)
            log.info('Visualization saved in visdom')

    def train(self, ):
        for epoch in range(self.start_epoch, self.cfg.optimizer.max_epochs):
            self.curr_epoch = epoch
            self.train_epoch()
            if self.sched is not None:
                self.sched.step()
                self.sched_aud.step()
                self.sched_audatt.step()
            if epoch % self.cfg.validation_epoch_interval == 0 and epoch > 0:
                self.val_epoch()
            if epoch % self.cfg.checkpoint_epoch_interval == 0 and epoch > 0:
                self.save_checkpoint()
    
    def build_networks(self,):
        # merge configs in cfg.render, cfg. raysampler, implicit_function
        # since they all correspond to Nerf
        log.info('Initializing nerf model..')
        OmegaConf.set_struct(self.cfg.renderer, False)
        renderer_cfg = OmegaConf.merge(
            self.cfg.renderer, 
            self.cfg.implicit_function, 
            self.cfg.raysampler,)
        self.model = instantiate(
            renderer_cfg, 
            image_size = self.cfg.data.dataset.image_size, 
            ).to(self.device)
        
        components = instantiate(self.cfg.components)
        self.audionet = components['AudioNet'].to(self.device)
        self.audioattnet = components['AudioAttNet'].to(self.device)
        
        if self.cfg.precache_rays:
            log.info('Pre-caching Rays..')
            self.model.eval()
            with torch.no_grad():
                for datasets in (self.train_datasets, self.val_datasets):
                    for dataset in datasets:
                        cache_cameras = [e["camera"].to(self.device) for e in dataset]
                        cache_camera_hashes = [e["camera_idx"] for e in dataset]
                        self.model.precache_rays(cache_cameras, cache_camera_hashes)
                
    def setup_optimizer(self, ): 
        log.info('Setting up optimizers..')
        optim_cfg = self.cfg.optimizer
        
        # fixing name typo
        if optim_cfg.algo == 'adam': optim_cfg.algo = 'Adam'
        if optim_cfg.algo == 'sgd': optim_cfg.algo = 'SGD'
        
        optim = getattr(torch.optim, optim_cfg.algo)
        self.optim = optim([
            dict(
                params=self.model.parameters(), 
                lr=optim_cfg.lr,
                betas=(0.9, 0.999))])
        self.optim_aud = optim([
            dict(
                params=self.audionet.parameters(),
                lr=optim_cfg.lr,
                betas=(0.9, 0.999))])
        self.optim_audatt = optim([
            dict(
                params=self.audioattnet.parameters(),
                lr=optim_cfg.lr * 5, # because AD-Nerf did this :>
                betas=(0.9, 0.999))])
        
    def setup_scheduler(self, ):
        sched_cfg = self.cfg.optimizer.schedule
        if sched_cfg:
            if sched_cfg.type == 'LambdaLR':
                lr_lambda = lambda epoch: sched_cfg.gamma ** (epoch / sched_cfg.step_size)
                other_kwargs = {
                    'last_epoch':self.start_epoch - 1,
                    'lr_lambda':lr_lambda,
                    'verbose':sched_cfg.verbose}
                self.sched = torch.optim.lr_scheduler.LambdaLR(
                    self.optim,
                    **other_kwargs)
                self.sched_aud = torch.optim.lr_scheduler.LambdaLR(
                    self.optim_aud,
                    **other_kwargs)
                other_kwargs.update({
                    'lr_lambda': lambda epoch: 5 * sched_cfg.gamma ** (epoch / sched_cfg.step_size)})
                self.sched_audatt = torch.optim.lr_scheduler.LambdaLR(
                    self.optim_audatt,
                    **other_kwargs)
            else:
                # TODO
                raise NotImplementedError
            log.info('Scheduler {0} starts from epoch {1}'.format(sched_cfg.type, self.start_epoch))
        else:
            self.sched = None
            log.info('Not scheduler specified')
            
    def save_checkpoint(self, ):
        checkpoint_name = 'epoch{}_weights.pth'.format(self.curr_epoch)
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, checkpoint_name)
        log.info('Storing checkpoint in {}..'.format(checkpoint_path))
        data_to_store = {
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'audionet': self.audionet.state_dict(),
            'optim_aud': self.optim_aud.state_dict(),
            'audioattnet': self.audioattnet.state_dict(),
            'optim_audatt': self.optim_audatt.state_dict(),
            'stats': pickle.dumps(self.stats)}  
        torch.save(data_to_store, checkpoint_path)
        
    def restore_checkpoint(self, ):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        if self.cfg.resume and os.path.isfile(self.cfg.resume_from):
            # base_dir, _ = os.path.split(self.cfg.checkpoint_dir)
            log.info('Resuming weights from checkpoint {}..'.format(self.cfg.resume_from))
            loaded_data = torch.load(self.cfg.resume_from)
            # other states
            self.stats = pickle.loads(loaded_data['stats'])
            self.start_epoch = self.stats.epoch
            # model related
            self.model.load_state_dict(loaded_data['model'])
            self.audionet.load_state_dict(loaded_data['audionet'])
            self.audioattnet.load_state_dict(loaded_data['audioattnet'])
            # optimizer related
            if hasattr(self, 'optim'):
                self.optim.load_state_dict(loaded_data['optim'])
                self.optim.last_epoch =  self.stats.epoch
            if hasattr(self, 'optim_aud'):
                self.optim_aud.load_state_dict(loaded_data['optim_aud'])
                self.optim_aud.last_epoch =  self.stats.epoch
            if hasattr(self, 'optim_audatt'):
                self.optim_audatt.load_state_dict(loaded_data['optim_audatt'])
                self.optim_audatt.last_epoch =  self.stats.epoch
        elif self.cfg.resume and not os.path.isfile(self.cfg.resume_from):
            log.error('Checkpint {} not exists'.format(self.cfg.checkpoint_dir))
            raise Exception
        else:
            log.info('Starting new checkpoint')
            self.stats = Stats(["loss", "mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "sec/it"])
            self.start_epoch = 0
            