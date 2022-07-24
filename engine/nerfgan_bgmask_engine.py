import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
import logging
from typing import List
from visdom import Visdom
from data.utils import trivial_collate
import os 
import pickle
from PIL import Image
from .basic_engine import NerfEngine as BasicEngine
from .utils import Stats, visualize_nerfgan_bgmask_outputs
from gan_utils.losses import lpips
from gan_utils.losses.id.id_loss import IDLoss
import importlib

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

    # gan function
    def requires_grad(self, 
                      model, 
                      flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train(self, ):
        for epoch in range(self.start_epoch, self.cfg.optimizer.max_epochs):
            self.curr_epoch = epoch
            self.train_epoch()
            if self.sched_gen is not None:
                self.sched_gen.step()
            if self.sched_dis is not None:
                self.sched_dis.step()
            if epoch % self.cfg.validation_epoch_interval == 0: 
                self.val_epoch()
            if epoch % self.cfg.checkpoint_epoch_interval == 0:
                self.save_checkpoint()
                
    def train_epoch(self, ):
        self.gen.train()
        self.stats.new_epoch()
        
        
        for num_dataset, dataloader in enumerate(self.train_dataloaders):
            # choose to fetch data via Dataset rather than Dataloader, because of the following operations.
            dataset = dataloader.dataset
            bg_image = dataset.meta['bg_image'].permute(2,0,1).unsqueeze(0).to(self.device)
            for iteration, batch in enumerate(dataloader):
                
                self.optim_gen.zero_grad()
                self.optim_dis.zero_grad()

                image, parsing, camera, audio, _, camera_idx = batch.values()

                gt_image = image.permute(0, 3, 1, 2).to(self.device)
                parsing = parsing.permute(0, 3, 1, 2).to(self.device)
                
                face_mask = ((parsing[:,0,:,:] == 0) * (parsing[:,1,:,:] == 0) * (parsing[:,2,:,:] == 1) * 1.0).unsqueeze(1)
                bg_mask = ((parsing[:,0,:,:] == 1) * (parsing[:,1,:,:] == 1) * (parsing[:,2,:,:] == 1) * 1.0).unsqueeze(1)
                torso_mask = (1 - face_mask)*(1- bg_mask)

                gt_image = face_mask * gt_image + (1- face_mask) * torch.ones_like(gt_image)

                loss_dict = {}                
                #####################################################
                if self.cfg.losses.gan > 0:
                    # Train discriminator
                    self.requires_grad(self.gen, False)
                    self.requires_grad(self.dis, True)
                    
                    # Run the forward pass of the model
                    other_params = {
                        'camera_hash':  camera_idx if self.cfg.precache_rays else None, 
                        'camera': camera.to(self.device),
                        'ref_image': bg_image}  # arguments that original RadianceFieldRenderer in vanillar Nerf used
                    image_list, mask_list = self.gen(
                        audio = audio.to(self.device),
                        needs_multi_res = False,
                        needs_mask = False,
                        **other_params)

                    # Discriminator prediction on fake images
                    fake_pred = F.softplus(   self.dis(image_list[-1])).mean()
                    real_pred = F.softplus( - self.dis(gt_image      )).mean()
                    d_loss = self.cfg.losses.gan * (fake_pred + real_pred)
                    
                    # Optimize discriminator
                    d_loss.backward()
                    self.optim_dis.step()
                    
                    # Record loss terms
                    loss_dict['loss_dis'] = d_loss
                    loss_dict['dis_real'] = real_pred
                    loss_dict['dis_fake'] = fake_pred
                else:
                    # Record loss terms
                    loss_dict['loss_dis'] = torch.tensor(0)
                    loss_dict['dis_real'] = torch.tensor(0)
                    loss_dict['dis_fake'] = torch.tensor(0)

                # Train generator 
                self.requires_grad(self.gen, True)
                self.requires_grad(self.dis, False)

                # Run the forward pass of the model
                other_params = {
                    'camera_hash':  camera_idx if self.cfg.precache_rays else None, 
                    'camera': camera.to(self.device),
                    'ref_image': bg_image}  # arguments that original RadianceFieldRenderer in vanillar Nerf used
                image_list, mask_list = self.gen(
                    audio = audio.to(self.device),
                    needs_multi_res = True,
                    needs_mask = True,
                    **other_params)
                
                nerf_out = F.interpolate(image_list[-1], gt_image.shape[-2:])

                if self.cfg.losses.gan > 0:
                    # Discriminator predictions on fake images
                    fake_pred = self.dis(nerf_out)
                    
                    # Collect losses
                    gan_loss = F.softplus(-fake_pred).mean()
                else:
                    gan_loss = torch.tensor(0)

                if self.cfg.losses.percept > 0:
                    percept_loss = self.lpips_func(
                        nerf_out, 
                        gt_image).mean()
                else:
                    percept_loss = torch.tensor(0)

                if self.cfg.losses.id > 0:
                    id_loss, _, _ = self.id_func(
                        nerf_out, 
                        gt_image, 
                        bg_image); 
                    id_loss = torch.ones_like(id_loss) - id_loss
                else:
                    id_loss = torch.tensor(0)
                
                
                # # for debug
                # from PIL import Image; import numpy as np
                # Image.fromarray((face_mask[0,0]*255).detach().cpu().numpy()).convert('RGB').save('/home/15288906612/codes/talknerf/wasted/face_mask.png')
                # Image.fromarray((bg_mask[0,0]*255).detach().cpu().numpy()).convert('RGB').save('/home/15288906612/codes/talknerf/wasted/bg_mask.png')
                # Image.fromarray((torso_mask[0,0]*255).detach().cpu().numpy()).convert('RGB').save('/home/15288906612/codes/talknerf/wasted/torso_mask.png')
                # Image.fromarray(((gt_image * torso_mask)[0].permute(1,2,0)*255).detach().cpu().numpy().astype(np.uint8)).save('/home/15288906612/codes/talknerf/wasted/gt_torso.png')
                # Image.fromarray(((gt_image * face_mask)[0].permute(1,2,0)*255).detach().cpu().numpy().astype(np.uint8)).save('/home/15288906612/codes/talknerf/wasted/gt_face.png')

                # #####################################################
                resize = lambda image, h, w : F.interpolate(image, (h, w)) if image.shape[-2] != h or image.shape[-1] != w else image
                multi_res = lambda image: [resize(image, *res_out.shape[-2:]) for res_out in image_list]

                gt_image_list = multi_res(gt_image * face_mask + (1 - face_mask))
                face_mask_list = multi_res(face_mask)

                dumy_sumup = lambda preds, gts: sum([self.l1_func(pred, gt) for pred, gt in zip(preds, gts)])
                mask_sumup = lambda preds, gts, masks: sum([self.l1_func(mask * pred, mask * gt) for pred, gt, mask in zip(preds, gts, masks)])

                l1_loss = dumy_sumup(image_list, gt_image_list)
                l1_mask_loss = mask_sumup(mask_list, face_mask_list, face_mask_list)
                l1_face_loss = mask_sumup(image_list, gt_image_list, face_mask_list)
                


                nerf_out = image_list[-1]
                gt_image = gt_image_list[-1]

                g_loss = l1_loss + \
                    self.cfg.losses.l1_face * l1_face_loss + \
                    self.cfg.losses.mask * l1_mask_loss + \
                    self.cfg.losses.gan * gan_loss + \
                    self.cfg.losses.percept * percept_loss

                loss_dict['loss_gen'] = g_loss
                loss_dict['gen_l1_face'] = l1_face_loss
                loss_dict['gen_l1_mask'] = l1_mask_loss
                loss_dict['gen_psnr'] =  -10.0 * torch.log10( torch.mean((nerf_out - gt_image) ** 2)) 
                if self.cfg.losses.gan > 0:
                    loss_dict['gen_gan'] = gan_loss
                if self.cfg.losses.percept > 0:
                    loss_dict['gen_percept'] = percept_loss

                # #####################################################
                # # generate multi-resolution images and masks
                # resize = lambda image, h, w : F.interpolate(image, (h, w)) if image.shape[-2] != h or image.shape[-1] != w else image
                # multi_res = lambda image: [resize(image, *res_out.shape[-2:]) for res_out in image_list]

                # face_mask_list = multi_res(face_mask)                          
                # torso_mask_list = multi_res(torso_mask)                             
                # bg_mask_list = multi_res(bg_mask)  
                # gt_image_list =  multi_res(gt_image)

                # # put mask on defferent results, and calculate loss
                # mask_sumup = lambda preds, gts, masks: sum([self.l1_func(mask * pred, mask * gt) for pred, gt, mask in zip(preds, gts, masks)])
                
                # l1_face_loss = mask_sumup(image_list, gt_image_list, face_mask_list)
                # l1_torso_loss = mask_sumup(image_list, gt_image_list, torso_mask_list)  
                # l1_bg_loss =  mask_sumup(image_list, g_image_list, bg_mask_list)  
                # mask_loss = mask_sumup(mask_list, face_mask_list, face_mask_list) + mask_sumup(mask_list, face_mask_list, bg_mask_list)                
                    
                # # sum up all losses multiplied by their wights
                # g_loss = self.cfg.losses.gan * gan_loss + \
                #          self.cfg.losses.percept * percept_loss + \
                #          self.cfg.losses.id * id_loss + \
                #          self.cfg.losses.l1_face * l1_face_loss + \
                #          self.cfg.losses.l1_torso * l1_torso_loss + \
                #          self.cfg.losses.l1_bg * l1_bg_loss + \
                #          self.cfg.losses.mask * mask_loss
                
                # # Record loss terms
                # loss_dict['loss_gen'] = g_loss
                # loss_dict['gen_gan'] = gan_loss
                # loss_dict['gen_l1_face'] = l1_face_loss if type(l1_face_loss) is not int else torch.tensor(0)
                # loss_dict['gen_l1_torso'] = l1_torso_loss if type(l1_torso_loss) is not int else torch.tensor(0)
                # loss_dict['gen_l1_bg'] = l1_bg_loss if type(l1_bg_loss) is not int else torch.tensor(0)
                # loss_dict['gen_l1_mask'] = mask_loss if type(mask_loss) is not int else torch.tensor(0)
                # loss_dict['gen_percept'] = percept_loss
                # loss_dict['gen_id'] = id_loss     
                # loss_dict['gen_psnr'] =   -10.0 * torch.log10( torch.mean((nerf_out - gt_image) ** 2))     
                # #####################################################

                # Optimizer generator
                g_loss.backward()
                self.optim_gen.step()

                # Update stats with the current metrics.
                self.stats.update(
                    {**loss_dict},
                    stat_set= 'train')
            
                if iteration % self.cfg.stats_print_interval == 0:
                    self.stats.print(stat_set="train")
            
                # Update the visualization cache.
                if self.viz is not None:
                    self.visuals_cache.append({
                        "camera": camera.cpu(),
                        "camera_idx":camera_idx,
                        "first_frame":image.cpu().detach(),
                        "pred_frame": nerf_out.cpu().detach(),
                        "gt_frame": gt_image.cpu().detach(),
                        })

            log.info('Training done on {} datasets'.format(num_dataset))

    def evaluate_full(self, ):
        self.gen.eval()
    
        for num_dataset, dataloader in enumerate(self.val_dataloaders):

            # choose to fetch data via Dataset rather than Dataloader, because of the following operations.
            dataset = dataloader.dataset
            bg_image = dataset.meta['bg_image'].permute(2,0,1).unsqueeze(0).to(self.device)

            for iteration, batch in enumerate(dataloader):
                
                # Unpack values
                image, parsing, camera, audio, _, camera_idx = batch.values()               
                
                gt_image = image.permute(0, 3, 1, 2).to(self.device)
                parsing = parsing.permute(0, 3, 1, 2).to(self.device)

                # Activate eval mode of the model (lets us do a full rendering pass).
                with torch.no_grad():
                    
                    # Run the forward pass of the model
                    other_params = {
                        'camera_hash':  camera_idx if self.cfg.precache_rays else None, 
                        'camera': camera.to(self.device),
                        'ref_image': bg_image}  # arguments that original RadianceFieldRenderer in vanillar Nerf used
                    val_image_list, val_mask_list = self.gen(
                        audio = audio.to(self.device),
                        needs_multi_res = False,
                        needs_mask = False,
                        **other_params)

                face_mask = ((parsing[:,0,:,:] == 0) * (parsing[:,1,:,:] == 0) * (parsing[:,2,:,:] == 1) * 1.0).unsqueeze(1)
                bg_mask = ((parsing[:,0,:,:] == 1) * (parsing[:,1,:,:] == 1) * (parsing[:,2,:,:] == 1) * 1.0).unsqueeze(1)
                torso_mask = (1 - face_mask)*(1- bg_mask)

                resize = lambda image, h, w : F.interpolate(image, (h, w)) if image.shape[-2] != h or image.shape[-1] != w else image
                multi_res = lambda image: [resize(image, *res_out.shape[-2:]) for res_out in val_image_list]

                gt_image_list = multi_res(gt_image * face_mask)
                
                val_nerf_out = val_image_list[-1]
                gt_image = gt_image_list[-1]

                val_metrics = {}
                val_nerf_out = F.interpolate(val_image_list[-1], gt_image.shape[-2:])
                if self.cfg.losses.percept > 0:
                    val_metrics["gen_percept"] = self.lpips_func(val_nerf_out, gt_image).mean() 
                if self.cfg.losses.id > 0:
                    id_loss, _, _ = self.id_func(val_nerf_out, gt_image, bg_image); id_loss = torch.ones_like(id_loss) - id_loss 
                    val_metrics["gen_id"] = id_loss
                val_metrics["gen_psnr"] = -10.0 * torch.log10( torch.mean((val_nerf_out - gt_image) ** 2))                  




                            


    def videos_synthesis(self, ):
        self.gen.eval()

        frame_paths = []
        for num_dataset, test_dataloader in enumerate(self.test_dataloaders):
            
            # choose to fetch data via Dataset rather than Dataloader, because of the following operations.
            dataset = test_dataloader.dataset
            bg_image = dataset.meta['bg_image'].permute(2,0,1).unsqueeze(0).to(self.device)

            for iteration, test_batch in enumerate(test_dataloader):
                
                # Unpack values
                test_image, test_parsing, test_camera, test_audio, _, test_camera_idx = test_batch.values()
                
                # Activate eval mode of the model (lets us do a full rendering pass).
                with torch.no_grad():

                    # Run the foward pass of the model
                    other_params = {
                        'camera_hash':  test_camera_idx if self.cfg.precache_rays else None, 
                        'camera': test_camera.to(self.device),
                        'ref_image': bg_image}  # arguments that original RadianceFieldRenderer in vanillar Nerf used
                    
                    test_image_list, tset_mask_list = self.gen(
                        audio = test_audio.to(self.device),
                        needs_multi_res = False,
                        needs_mask = False,
                        **other_params)


                # TODO
                # Writing images
                frame = test_image_list[-1].detach().cpu()
                frame_path = os.path.join(self.export_dir, f"scene_{num_dataset:01d}_frame_{iteration:05d}.png")
                log.info(f"Writing {frame_path}")
                tensor2np = lambda x: ((x if x.shape[-1] <= 3 else x.permute(0,2,3,1)).detach().cpu().numpy() * 255.0).astype(np.uint8)
                if self.cfg.test.with_gt:
                    Image.fromarray(
                        np.hstack([
                            tensor2np(test_image)[0],
                            tensor2np(frame)[0]])
                        ).save(frame_path)
                else:
                    Image.fromarray(tensor2np(frame)[0]).save(frame_path)
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
        self.val_epoch()
                
    def val_epoch(self, ):
        self.gen.eval()
        
        # Prepare evaluation metrics
        if not hasattr(self, 'lpips_func'):
            self.lpips_func = lpips.LPIPS(net='alex', version='0.1').to(self.device)
        if not hasattr(self, 'id_func'):
            self.id_func = IDLoss(device = self.device)
    
        for num_dataset, dataloader in enumerate(self.val_dataloaders):

            # choose to fetch data via Dataset rather than Dataloader, because of the following operations.
            dataset = dataloader.dataset
            bg_image = dataset.meta['bg_image'].permute(2,0,1).unsqueeze(0).to(self.device)

            for iteration, batch in enumerate(dataloader):
                
                # Unpack values
                image, parsing, camera, audio, _, camera_idx = batch.values()
                
                gt_image = image.permute(0, 3, 1, 2).to(self.device)
                parsing = parsing.permute(0, 3, 1, 2).to(self.device)

                # Activate eval mode of the model (lets us do a full rendering pass).
                with torch.no_grad():
                    
                    # Run the forward pass of the model
                    other_params = {
                        'camera_hash':  camera_idx if self.cfg.precache_rays else None, 
                        'camera': camera.to(self.device),
                        'ref_image': bg_image}  # arguments that original RadianceFieldRenderer in vanillar Nerf used
                    
                    val_image_list, val_mask_list = self.gen(
                        audio = audio.to(self.device),
                        needs_multi_res = True,
                        needs_mask = True,
                        **other_params)

                face_mask = ((parsing[:,0,:,:] == 0) * (parsing[:,1,:,:] == 0) * (parsing[:,2,:,:] == 1) * 1.0).unsqueeze(1)
                bg_mask = ((parsing[:,0,:,:] == 1) * (parsing[:,1,:,:] == 1) * (parsing[:,2,:,:] == 1) * 1.0).unsqueeze(1)
                torso_mask = (1 - face_mask)*(1- bg_mask)

                resize = lambda image, h, w : F.interpolate(image, (h, w)) if image.shape[-2] != h or image.shape[-1] != w else image
                multi_res = lambda image: [resize(image, *res_out.shape[-2:]) for res_out in val_image_list]

                gt_image_list = multi_res(gt_image * face_mask + (1 - face_mask))
                
                val_nerf_out = val_image_list[-1]
                gt_image = gt_image_list[-1]

                val_metrics = {}
                if self.cfg.losses.percept > 0:
                    val_metrics["gen_percept"] = self.lpips_func(val_nerf_out, gt_image).mean()
                if self.cfg.losses.id > 0:
                    id_loss, _, _ = self.id_func(val_nerf_out, gt_image, bg_image); id_loss = torch.ones_like(id_loss) - id_loss
                    val_metrics["gen_id"] = id_loss
                val_metrics["gen_psnr"] = -10.0 * torch.log10( torch.mean((val_nerf_out - gt_image) ** 2))                  
                
                # Update stats with the validation metrics.  
                self.stats.update(val_metrics, stat_set="val")

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
            visualize_nerfgan_bgmask_outputs(
                nerf_out = {'rgb_gt': gt_image , 'rgb_pred': val_nerf_out, }, #'rgb_ref': bg_image
                multi_images = {'res_{}'.format(str(image.shape[-2])):image for image in val_image_list},
                multi_masks = {'res_{}'.format(str(mask.shape[-2])):mask for mask in val_mask_list},
                output_cache = self.visuals_cache,
                viz = self.viz, 
                visdom_env = self.cfg.visualization.visdom_env)
            log.info('Visualization saved in visdom')

    
    def build_networks(self,):
        # merge configs in cfg.render, cfg. raysampler, implicit_function
        # since they all correspond to Nerf
        log.info('Initializing nerf model..')
        OmegaConf.set_struct(self.cfg.renderer, False)
        renderer_cfg = OmegaConf.merge(
            self.cfg.renderer, 
            self.cfg.raysampler,
            self.cfg.implicit_function, )
        self.gen = instantiate(
            renderer_cfg,
            image_size = self.cfg.data.dataset.image_size
        ).to(self.device)
        components = instantiate(self.cfg.components)
        if self.cfg.train:
            self.dis = components['discriminator'].to(self.device)
            self.l1_func = nn.SmoothL1Loss().to(self.device)
            self.lpips_func = lpips.LPIPS(net='alex', version='0.1').to(self.device)
            self.id_func = IDLoss(device = self.device)
        
    
    def setup_optimizer(self,):
        log.info('Setting up optimizers..')
        optim_cfg = self.cfg.optimizer
        
        # fixing name typo
        if optim_cfg.algo == 'adam': optim_cfg.algo = 'Adam'
        if optim_cfg.algo == 'sgd': optim_cfg.algo = 'SGD'
        
        optim = getattr(torch.optim, optim_cfg.algo)
        assert self.cfg.train
        self.optim_gen = optim([
            dict(
                params=self.gen.parameters(), 
                lr=optim_cfg.lr,
                betas=(0.9, 0.999))])
        self.optim_dis = optim([
            dict(
                params=self.dis.parameters(),
                lr=optim_cfg.lr,
                betas=(0.9, 0.999))])
        
    def setup_scheduler(self):
        sched_cfg = self.cfg.optimizer.schedule
        if sched_cfg:
            if sched_cfg.type == 'LambdaLR':
                lr_lambda = lambda epoch: sched_cfg.gamma ** (epoch / sched_cfg.step_size)
                other_kwargs = {
                    'last_epoch':self.start_epoch - 1,
                    'lr_lambda':lr_lambda,
                    'verbose':sched_cfg.verbose}
                self.sched_gen = torch.optim.lr_scheduler.LambdaLR(
                    self.optim_gen,
                    **other_kwargs)
                self.sched_dis = torch.optim.lr_scheduler.LambdaLR(
                    self.optim_dis,
                    **other_kwargs)
            else:
                # TODO
                raise NotImplementedError
            log.info('Scheduler {0} starts from epoch {1}'.format(sched_cfg.type, self.start_epoch))
        else:
            self.sched = None
            log.info('Not scheduler specified')
            
    def save_checkpoint(self ):
        checkpoint_name = 'epoch{}_weights.pth'.format(self.curr_epoch)
        checkpoint_path = os.path.join(self.cfg.checkpoint_dir, checkpoint_name)
        log.info('Storing checkpoint in {}..'.format(checkpoint_path))
        data_to_store = {
            'gen': self.gen.state_dict(),
            'optim_gen': self.optim_gen.state_dict(),
            'dis': self.dis.state_dict(),
            'optim_dis': self.optim_dis.state_dict(),
            'state': pickle.dumps(self.stats)}
        torch.save(data_to_store, checkpoint_path)
        
    def restore_checkpoint(self):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        if self.cfg.resume and os.path.isfile(self.cfg.resume_from):
            log.info('Resuming weights from checkpoint {}..'.format(self.cfg.resume_from))
            loaded_data = torch.load(self.cfg.resume_from)
            # other states
            self.stats = pickle.loads(loaded_data['state'])
            self.start_epoch = self.stats.epoch
            # model related
            self.gen.load_state_dict(loaded_data['gen'])
            if hasattr(self, 'dis'):
                self.dis.load_state_dict(loaded_data['dis'])
            # optimizer related
            if hasattr(self, 'optim_gen'):
                self.optim_gen.load_state_dict(loaded_data['optim_gen'])
                self.optim_gen.last_epoch = self.stats.epoch
            if hasattr(self, 'optim_dis'):
                self.optim_dis.load_state_dict(loaded_data['optim_dis'])
                self.optim_dis.last_epoch = self.stats.epoch
        elif self.cfg.resume and not os.path.isfile(self.cfg.resume_from):
            log.error('Checkpint {} not exists'.format(self.cfg.checkpoint_dir))
            raise Exception
        else:
            log.info('Starting new checkpoint')
            self.stats = Stats(["gen_psnr", "gen_l1_face", "gen_l1_bg", "gen_l1_torso", "gen_l1_mask", "gen_percept", "gen_id", "gen_gan", "dis_real", "dis_gan", "dis_fake", "loss_gen", "loss_dis", "sec/it"])
            self.start_epoch = 0
