import torch
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
import importlib
from visdom import Visdom
from data.utils import trivial_collate
import os 
from PIL import Image
from .utils import Stats, visualize_nerf_outputs
import pickle
import collections
from typing import Optional

log = logging.getLogger(__name__)

class NerfEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # specifying device
        if not torch.cuda.is_available() and cfg.device == 'cuda':
            log.info('Specifying expreriment on GPU while GPU is not available')
            raise Exception
        
        self.device = cfg.device
        log.info('Specify {} as the device.'.format(self.device))

        
    def load_dataset(self,):
        datasets = instantiate(self.cfg.data.dataset)
        if self.cfg.train:
            self._load_dataset_train(datasets)

        if self.cfg.test:
            self._load_dataset_test(datasets)
            
    def _load_dataset_train(self,
                            datasets:object):
        log.info('Loading training datasets..')
        self.train_datasets = datasets.train
        self.train_dataloaders = []
        for dataset in datasets.train:
            self.train_dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.cfg.data.dataloader.batch_size,
                    shuffle=True,
                    num_workers=self.cfg.data.dataloader.num_workers,
                    collate_fn=getattr(importlib.import_module('data.utils'), self.cfg.data.dataloader.collate_fn)))
        
        log.info('Loading validation datasets..')
        self.val_datasets = datasets.val
        self.val_dataloaders = []
        for dataset in datasets.val:
            self.val_dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=getattr(importlib.import_module('data.utils'), self.cfg.data.dataloader.collate_fn),
                    sampler=torch.utils.data.RandomSampler(
                        datasets.val, 
                        replacement=self.cfg.validation.replacement,
                        num_samples=self.cfg.validation.num_samples)))
            
    def _load_dataset_test(self,
                           datasets:object):
        self.test_datasets = datasets.test
        if self.cfg.test.mode == 'evaluation':
            log.info('Loading test datasets for evaluations..')
                
        if self.cfg.test.mode == 'export_video':
            from nerf_utils.utils import generate_eval_video_cameras
            log.info('Loading training datasets with new camera pose for visualization..')
            self.test_datasets = [generate_eval_video_cameras(
                dataset._entries, 
                trajectory_type=self.cfg.test.trajectory_type,
                up=self.cfg.test.up,
                scene_center=self.cfg.test.scene_center,
                n_eval_cams=self.cfg.test.n_frames,
                trajectory_scale=self.cfg.test.trajectory_scale) 
                                    for dataset in datasets.train]

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
                    collate_fn=trivial_collate))

            
            
    def build_networks(self,):
        # merge configs in cfg.render, cfg. raysampler, implicit_function
        # since they all correspond to Nerf
        log.info('Initializing nerf model..')
        OmegaConf.set_struct(self.cfg.renderer, False)
        renderer_cfg = OmegaConf.merge(
            self.cfg.renderer, 
            self.cfg.implicit_function, 
            self.cfg.raysampler,)
        self.model = instantiate(renderer_cfg, image_size = self.cfg.data.dataset.image_size, )
        self.model = self.model.to(self.device)
        
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
                lr=optim_cfg.lr)])
        
    def setup_scheduler(self, ):
        sched_cfg = self.cfg.optimizer.schedule
        if sched_cfg:
            if sched_cfg.type == 'LambdaLR':
                lr_lambda = lambda epoch: sched_cfg.gamma ** (epoch / sched_cfg.step_size)
                self.sched = torch.optim.lr_scheduler.LambdaLR(
                    self.optim,
                    lr_lambda,
                    last_epoch = self.start_epoch - 1,
                    verbose = sched_cfg.verbose)
            else:
                # TODO
                raise NotImplementedError
            log.info('Scheduler {0} starts from epoch {1}'.format(sched_cfg.type, self.start_epoch))
        else:
            self.sched = None
            log.info('Not scheduler specified')
    
    def setup_visualizer(self, ):
        log.info('Initializing Visdom...')
        vis_cfg = self.cfg.visualization
        self.visuals_cache = collections.deque(maxlen = vis_cfg.history_size)
        if vis_cfg.visdom:
            self.viz = Visdom(
                server=vis_cfg.visdom_server,
                port=vis_cfg.visdom_port,
                use_incoming_socket=False)
        else:
            self.viz = None 
               
    def train(self, ):
        for epoch in range(self.start_epoch, self.cfg.optimizer.max_epochs):
            self.curr_epoch = epoch
            self.train_epoch()
            if self.sched is not None:
                self.sched.step()
            if epoch % self.cfg.validation_epoch_interval == 0 and epoch > 0:
                self.val_epoch()
            if epoch % self.cfg.checkpoint_epoch_interval == 0 and epoch > 0:
                self.save_checkpoint()
            
    
    def train_epoch(self, ):
        self.model.train()
        self.stats.new_epoch()
        for num_dataset, dataloader in enumerate(self.train_dataloaders):
            for iteration, batch in enumerate(dataloader):
                # Unpack values
                image, camera, camera_idx = batch[0].values()
                self.optim.zero_grad()

                # Run the forward pass of the model 
                nerf_out, metrics = self.model(
                    camera_idx if self.cfg.precache_rays else None, 
                    camera.to(self.device),
                    image.to(self.device))

                # The loss is a sum of coarse and fine MSEs
                loss = metrics["mse_coarse"] + metrics["mse_fine"]

                # Take the training step
                loss.backward()
                self.optim.step()

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
                    
            log.info('Training done on {} datasets'.format(num_dataset + 1))

    def videos_synthesis(self, ):
        self.model.eval()
        frame_paths = []
        for num_dataset, dataloader in enumerate(self.test_dataloaders):
            for iteration, test_batch in enumerate(dataloader):
                test_image, test_camera, camera_idx = test_batch[0].values()
                if test_image is not None:
                    test_image = test_image.to(self.device)
                test_camera = test_camera.to(self.device)
                
                with torch.no_grad():
                    test_nerf_out, test_metrics = self.model(
                        None,
                        test_camera,
                        test_image)
                
                # Writing images
                frame = test_nerf_out["rgb_fine"][0].detach().cpu()
                frame_path = os.path.join(self.export_dir, f"scene_{num_dataset:01d}_frame_{iteration:05d}.png")
                log.info(f"Writing {frame_path}")
                Image.fromarray((frame.numpy() * 255.0).astype(np.uint8)).save(frame_path)
                frame_paths.append(frame_path)
                
        # Convert the exported frames to a video
        video_path = os.path.join(os.getcwd(), "video.mp4")
        ffmpeg_bin = "ffmpeg"
        frame_regexp = os.path.join(os.getcwd(), self.export_dir,"*.png" )
        ffmcmd = (
            "%s -r %d -i %s -f mp4 -y -b:v 2000k -pix_fmt yuv420p %s"
            %(ffmpeg_bin, self.cfg.test.fps, frame_regexp, video_path))
        log.info('Video gnerated via {} \n {}'.format(ffmpeg_bin, ffmcmd))
        ret = os.system(ffmcmd)
        if ret != 0:
            raise RuntimeError("ffmpeg failed!")    
            
                    
                    
    def image_metrics(self, 
                      image:torch.Tensor, 
                      rgb_fine:Optional[torch.Tensor]=None, 
                      rgb_coarse:Optional[torch.Tensor]=None):
        """
        Generate extra metrics for model evaluation
        Args:
            image (torch.Tensor): ground truth image
            rgb_fine (Optinal[torch.Tensor], optional): fine result. Defaults to None.
            rgb_coarse (Optional[torch.Tensor], optional): coarse reulst. Defaults to None.

        Returns:
            metrics_dict (Dict): generation evaluation metrics.
        """
        
        metrics_dict = {}
        # LIPIS pretrained model initialization
        if not hasattr(self, 'lpips_func'):
            import lpips
            self.lpips_func = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_func.eval()
        
        # LIPIS scores
        from nerf_utils.utils import calc_lpips
        norm_image = lambda x: x * 2 - 1
        channel_first = lambda x: x.permute(0,3,1,2)
        with torch.no_grad():
            if rgb_fine is not None:
                metrics_dict['lpips_fine'] = calc_lpips(
                    norm_image(channel_first(image[None, ])), 
                    norm_image(channel_first(rgb_fine)), 
                    self.lpips_func
                    ).squeeze()
            if rgb_coarse is not None:
                metrics_dict['lpips_coarse'] = calc_lpips(
                    norm_image(channel_first(image[None, ])), 
                    norm_image(channel_first(rgb_coarse)), 
                    self.lpips_func
                    ).squeeze()     
        
        # SSIM scores
        from nerf_utils.utils import calc_ssim
        norm_image = lambda x: x * 255.
        to_numpy = lambda x: x.detach().cpu().numpy()
        if rgb_fine is not None:
            metrics_dict['ssim_fine'] = calc_ssim(
                norm_image(to_numpy(image)), 
                norm_image(to_numpy(rgb_fine[0])))
        if rgb_coarse is not None:
            metrics_dict['ssim_coarse'] = calc_ssim(
                norm_image(to_numpy(image)), 
                norm_image(to_numpy(rgb_coarse[0])))   
        
        return metrics_dict  
                
                   
    def evaluate_full(self, ):
        stats = Stats(["mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "ssim_coarse", "ssim_fine", "lpips_coarse", "lpips_fine", "sec/it"])
        stats.new_epoch()
        self.model.eval()
        for num_dataset, test_dataloader in enumerate(self.test_dataloaders):
            for iteration, test_batch in enumerate(test_dataloader):
                test_image, test_camera, camera_idx = test_batch[0].values()
                if test_image is not None:
                    test_image = test_image.to(self.device)
                test_camera = test_camera.to(self.device)
                
                with torch.no_grad():
                    test_nerf_out, test_metrics = self.model(
                        None, 
                        test_camera,
                        test_image)
                    
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
        for num_dataset, dataloader in enumerate(self.val_dataloaders):
            for iteration, val_batch in enumerate(dataloader):
                val_image, val_camera, camera_idx = val_batch[0].values()
                
                # Activate eval mode of the model (lets us do a full rendering pass).
                with torch.no_grad():
                    val_nerf_out, val_metrics = self.model(
                        camera_idx if self.cfg.precache_rays else None, 
                        val_camera.to(self.device),
                        val_image.to(self.device))
                  
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
    
    def save_checkpoint(self, ):
        checkpoint_name = 'epoch{}_weights.pth'.format(self.curr_epoch)
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, checkpoint_name)
        log.info('Storing checkpoint in {}..'.format(checkpoint_path))
        data_to_store = {
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
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
            # optimizer related
            if hasattr(self, 'optim'):
                self.optim.load_state_dict(loaded_data['optimizer'])
                self.optim.last_epoch = self.stats.epoch

        elif self.cfg.resume and not os.path.isfile(self.cfg.resume_from):
            log.error('Checkpint {} not exists'.format(self.cfg.checkpoint_dir))
            raise Exception
        else:
            log.info('Starting new checkpoint')
            self.stats = Stats(["loss", "mse_coarse", "mse_fine", "psnr_coarse", "psnr_fine", "sec/it"])
            self.start_epoch = 0
            
            
            
        
        
        
    

    
