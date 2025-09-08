import torch
import os
import copy
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from BSD500 import BSD500
from parseval_cnn import *
from utils import utilities, spline_utils
import matplotlib.pyplot as plt
from layers.averaged import AveragedDenoiser
torch.manual_seed(0)

class Trainer:
    """
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.sigma = config['sigma']

        # Prepare dataset classes
        train_dataset = BSD500(config['training_options']['train_data_file'])
        val_dataset = BSD500(config['training_options']['val_data_file'])
        print(train_dataset.__len__())

        print('Preparing the dataloaders')
        # Prepare dataloaders 
        self.train_dataloader = DataLoader(train_dataset, batch_size=config["training_options"]["batch_size"], shuffle=True, num_workers=config["training_options"]["num_workers"], drop_last=True)
        self.batch_size = config["training_options"]["batch_size"]
        self.val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)

        print('Building the model')
        
        # Build the model

        # self.model = ParsevalCNN(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN2(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN3(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN4(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN5(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN6(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN7(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN8(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN9(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN10(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN11(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN12(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN13(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN14(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN15(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN16(config['net_params'], config['activation_params']) 
        # self.model = ParsevalCNN17(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN18(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN19(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN20(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN21(config['net_params'], config['activation_params'])
        self.model = ParsevalCNN22(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN23(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN24(config['net_params'], config['activation_params'])
        # self.model = ParsevalCNN25(config['net_params'], config['activation_params'])

        # self.model = AveragedDenoiser(self.model, config['net_params']['beta'])
    
        self.model = self.model.to(device)

        print(self.model)
        
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
                
        # Set up the optimizer
        self.set_optimization()
        self.epochs = config["training_options"]['epochs']
        
        self.criterion = torch.nn.L1Loss(reduction='mean')

        # CHECKPOINTS & TENSOBOARD
        run_name = config['exp_name']
        self.checkpoint_dir = os.path.join(config['log_dir'], config["exp_name"], 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        config_save_path = os.path.join(config['log_dir'], config["exp_name"], 'config.json')
        with open(config_save_path, 'w') as handle:
            json.dump(self.config, handle, indent=4, sort_keys=True)

        writer_dir = os.path.join(config['log_dir'], config["exp_name"], 'tensorboard_logs')
        self.writer = tensorboard.SummaryWriter(writer_dir)

        self.total_training_step = 0

       
    def set_optimization(self):
        """Initialize the optmizer"""
        
        params_list = [{'params': spline_utils.get_no_spline_params(self.model), 'lr': self.config["optimizer"]["lr_weights"]}, \
            {'params': spline_utils.get_spline_coefficients(self.model), 'lr': self.config["optimizer"]["lr_spline_coeffs"]}, \
            {'params': spline_utils.get_spline_scaling_factors(self.model), 'lr': self.config["optimizer"]["lr_spline_scaling_coeffs"]}]
        if hasattr(self.model, 'direction_params'):
            dir_params = self.model.direction_params()
            no_dir_params = list(set(params_list[0]['params']) - set(dir_params))
            params_list[0]['params'] = no_dir_params
            params_list.append({'params': dir_params, 'lr': self.config["optimizer"]["lr_directions"]})
        
        # self.optimizer = torch.optim.Adam(params_list)
        self.optimizer = torch.optim.Adam(params_list)

    def train(self):
        
        best_psnr = 0
        for epoch in range(self.epochs+1):
            self.train_epoch(epoch)
            self.valid_epoch(epoch)
            self.save_checkpoint()
        
        self.writer.flush()
        self.writer.close()
        

    def train_epoch(self, epoch):
        """
        """
        self.model.train()

        tbar = tqdm(self.train_dataloader, ncols=135, position=0, leave=True)
        log = {}
        for batch_idx, data in enumerate(tbar):
            data = data.to(self.device) 
            noisy_data = data + self.sigma*torch.randn_like(data)/255.0

            self.optimizer.zero_grad()
            output = self.model(noisy_data)

            # data fidelity
            data_fidelity = self.criterion(output, data)
                
            # regularization
            regularization = torch.zeros_like(data_fidelity)
            if self.config['activation_params']['lmbda'] > 0:
                regularization = self.config['activation_params']['lmbda'] * spline_utils.get_total_tv2(self.model)

            # entropic
            if hasattr(self.model, 'entropic_loss') and callable(getattr(self.model, 'entropic_loss')):
                entropic_loss = self.config['activation_params']['entro'] * self.model.entropic_loss()
            else:
                entropic_loss = torch.tensor(0.0)

            total_loss = data_fidelity + regularization + entropic_loss
            # total_loss = entropic_loss
            total_loss.backward()
            self.optimizer.step()
                
            log['true_train_loss'] = total_loss.detach().cpu().item()
            log['train_loss']      = data_fidelity.detach().cpu().item()
            # log['lipschitz']  = self.model.network[2].lipschitz_constant()

            # We report the metrics after the network has seen a certain amount of data
            # This was done to compare the training loss with different gradient steps
            if self.total_training_step % (10 * 128 // self.batch_size)  == 0:
                # the step actually represents the amount of data that has been in the model
                self.wrt_step = self.total_training_step * self.batch_size
                self.write_scalars_tb(log)

            tbar.set_description('T ({}) | TotalLoss {:.5f} |'.format(epoch, log['train_loss'])) 
            self.total_training_step += 1

    def valid_epoch(self, epoch):
        
        self.model.eval()
        loss_val = 0.0
        psnr_val = 0.0
        ssim_val = 0.0

        tbar_val = tqdm(self.val_dataloader, ncols=130, position=0, leave=True)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tbar_val):
                data = data.to(self.device) 
                data = data[:, :, 1:, 1:] # TODO: Remove
                noisy_data = data + self.sigma*torch.randn_like(data)/255.0
                output = self.model(noisy_data)

                loss = self.criterion(output, data)

                loss_val += loss.cpu().item() / len(self.val_dataloader)
                out_val = torch.clamp(output, 0., 1.)
                psnr_val += utilities.batch_PSNR(out_val, data, 1.) / len(self.val_dataloader)
                ssim_val += utilities.batch_SSIM(out_val, data, 1.) / len(self.val_dataloader)
            
            # PRINT INFO
            tbar_val.set_description('EVAL ({}) | MSELoss: {:.5f} |'.format(epoch, loss_val))

            # METRICS TO TENSORBOARD
            self.wrt_mode = 'val'
            self.writer.add_scalar(f'{self.wrt_mode}/loss', loss_val, epoch)
            self.writer.add_scalar(f'{self.wrt_mode}/Test PSNR Mean', psnr_val, epoch)
            self.writer.add_scalar(f'{self.wrt_mode}/Test SSIM Mean', ssim_val, epoch)


    def write_scalars_tb(self, logs):
        for k, v in logs.items():
            self.writer.add_scalar(f'train/{k}', v, self.wrt_step)

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }

        print('Saving a checkpoint:')
        filename = self.checkpoint_dir + '/checkpoint.pth'
        torch.save(state, filename)