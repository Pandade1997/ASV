import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
import math, copy, time
import torch.nn as nn
from torch.nn import init
import numpy as np
from model.scheduler import get_scheduler
import codecs
from show import show_params, show_model

supported_rnns = {'lstm': nn.LSTM, 'rnn': nn.RNN, 'gru': nn.GRU}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())
supported_acts = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(), 'tanh': nn.Tanh(),'leakyrelu': nn.LeakyReLU(), 'prelu': nn.PReLU(), 'softplus': nn.Softplus()}
supported_acts_inv = dict((v, k) for k, v in supported_acts.items())
supported_loss = {'mseloss': nn.MSELoss(), 'kldivloss': nn.KLDivLoss(), 'smoothl1loss': nn.SmoothL1Loss()}
supported_loss_inv = dict((v, k) for k, v in supported_loss.items())
supported_norms = {'BatchNorm1d': nn.BatchNorm1d, 'BatchNorm2d': nn.BatchNorm2d, 'GroupNorm': nn.GroupNorm, 'LayerNorm': nn.LayerNorm,'InstanceNorm1d': nn.InstanceNorm1d, 'InstanceNorm2d': nn.InstanceNorm2d}
supported_norms_inv = dict((v, k) for k, v in supported_norms.items())

def ChooseQuantizationQParams(vmax, qmax):
    vmax = np.abs(vmax)
    Q = 0
    if vmax < qmax:
        while vmax * 2 <= qmax:
            Q = Q + 1.0
            vmax = vmax * 2.0
    else:
        while  vmax >= qmax:
            Q = Q - 1
            vmax = vmax * 0.5
    return Q

def QQuantize(qparams, fdata, qBit):
    if len(fdata.shape) < 2:
        fdata = fdata[np.newaxis, :]
    row, col = fdata.shape
    if qBit == 8:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int8)
    elif qBit == 16:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int16)
    elif qBit == 32:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int32)
    elif qBit == 64:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int64)
    else:
        fixed_data = np.zeros(shape = (row, col), dtype=np.int8)
    
    for i in range(row):
        for j in range(col):
            real_val = fdata[i, j]
            transformed_val = real_val * 2**qparams
            clamped_val = max( -2**(qBit-1),  min( 2**(qBit-1) - 1, transformed_val ))

            if qBit == 8:
                fixed_data[i, j]  = np.int8(np.round(clamped_val))
            elif qBit == 16:
                fixed_data[i, j]  = np.int16(np.round(clamped_val))
            elif qBit == 32:
                fixed_data[i, j]  = np.int32(np.round(clamped_val))
            elif qBit == 64:
                fixed_data[i, j]  = np.int64(np.round(clamped_val))
            else:
                fixed_data[i, j]  = np.int8(np.round(clamped_val))
    return fixed_data

def aQQuantize(qparams, fdata, bits):

    transformed_val = fdata * 2 ** qparams
    clamped_val = max( -2**(bits-1),  min( 2**(bits-1) - 1, transformed_val ))
    if bits == 8:
        fixed_data  = np.int8(np.round(clamped_val))
    elif bits == 16:
        fixed_data  = np.int16(np.round(clamped_val))
    elif bits == 32:
        fixed_data  = np.int32(np.round(clamped_val))
    elif bits == 64:
        fixed_data  = np.int64(np.round(clamped_val))
    else:
        fixed_data  = np.int8(np.round(clamped_val))
    return fixed_data

def get_non_pad_mask(padded_input, input_lengths=None, pad_idx=None):
    """padding position is set to 0, either use input_lengths or pad_idx
    """
    assert input_lengths is not None or pad_idx is not None
    if input_lengths is not None:
        # padded_input: N x T x ..
        N = padded_input.size(0)
        non_pad_mask = padded_input.new_ones(padded_input.size()[:-1])  # N x T
        for i in range(N):
            non_pad_mask[i, input_lengths[i]:] = 0
    if pad_idx is not None:
        # padded_input: N x T
        assert padded_input.dim() == 2
        non_pad_mask = padded_input.ne(pad_idx).float()
    # unsqueeze(-1) for broadcast
    return non_pad_mask.unsqueeze(-1)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def get_attn_key_pad_mask(seq_k, seq_q, pad_idx):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_idx)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask

def get_attn_pad_mask(padded_input, input_lengths, expand_length):
    """mask position is set to 1"""
    # N x Ti x 1
    non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths) # num_block, num_frame
    # N x Ti, lt(1) like not operation
    #pad_mask = non_pad_mask.squeeze(-1).lt(1)
    pad_mask = non_pad_mask.squeeze(-1).gt(0) #num_block, num_frame
    attn_mask = pad_mask.unsqueeze(1).expand(-1, expand_length, -1) # num_block, 1, num_frame
    return attn_mask


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1 or classname.find('Filter') != -1):
            if (hasattr(m, 'intialized') and m.intialized is True) or (hasattr(m, 'fix') and m.fix is True):
                print("%s has been intialized" % (classname))
            elif init_type == 'normal':
                init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def init_net(net, init_type='normal', init_gain=0.02, device = None):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    
    '''if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
        net.to(device)
        #net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    '''
    net.to(device)
    init_weights(net, init_type, init_gain=init_gain)
    return net

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.continue_from_name = opt.continue_from_name
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = opt.model_dir      # save all the checkpoints to save_dir
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        #self.setup(opt)

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        #if self.continue_from_name:
        #    self.load_networks(self.continue_from_name)

        self.print_networks(opt.verbose)

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()
            
        lr = self.optimizers[0].param_groups[0]['lr']
        #print('learning rate = %.7f' % lr)
        
    def set_lr_factor(self, lr_factor):
        """Update learning rates for all the networks; called at the end of every epoch"""
        if self.opt.lr_policy == 'warmup':
            for scheduler in self.schedulers:
                scheduler.set_lr_factor(lr_factor)
        
    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.eval()

    def train(self):
        """Make models train mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                net.train()

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        #visual_ret = {}
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def save_networks(self, suffix_name):
        """Save all the networks to the disk.
        Parameters:
            suffix_name (int) -- current epoch; used in the file name '%s_net_%s.pth' % (suffix_name, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s.pth' % (name, suffix_name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    #torch.save(net.modules.cpu().state_dict(), save_path)
                    torch.save(net.cpu().state_dict(), save_path)
                    net.to(self.device)
                else:
                    torch.save(net.cpu().state_dict(), save_path)
    
    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)
    
    def load_networks(self, suffix_names):
        suffix_names = suffix_names.split('-')
        num_model = len(suffix_names)
        if len(suffix_names) == 1:
            suffix_names = suffix_names[0]
        
        print(suffix_names)
        print("num_model = %d" % (num_model))
        
        if isinstance(suffix_names, list):
            nets_dict = {}
            for suffix_name in suffix_names:
                for name in self.model_names:
                    if isinstance(name, str):
                        load_filename = '%s_%s.pth' % (name, suffix_name)
                        load_path = os.path.join(self.save_dir, load_filename)
                        if not os.path.exists(load_path):
                            print("%s is not existed" % (load_path))
                            continue
                        print('average_model[%s]: loading the model from %s' % (name, load_path))
                        
                        cur_state_dict = torch.load(load_path, map_location=str(self.device))
                        if hasattr(cur_state_dict, '_metadata'):
                            del cur_state_dict._metadata
                        
                        if name in nets_dict.keys():
                            state_dict = nets_dict[name]
                            for key in state_dict.keys():
                                if key in cur_state_dict.keys():
                                    print("--> + %s" % (key))
                                    state_dict[key] += cur_state_dict[key]
                            nets_dict[name] = state_dict
                        else:
                            nets_dict[name] = cur_state_dict
                            
            for name in nets_dict.keys():
                state_dict = nets_dict[name]
                for key in state_dict.keys():
                    state_dict[key] /= num_model
                
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.modules
                    
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                
                net.load_state_dict(state_dict, strict=True)
                net = net.to(self.device)
        else:
            for name in self.model_names:
                if isinstance(name, str):
                    load_filename = '%s_%s.pth' % (name, suffix_names)
                    load_path = os.path.join(self.save_dir, load_filename)
                    if not os.path.exists(load_path):
                        print("%s is not existed" % (load_path))
                        continue
                    
                    net = getattr(self, name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.modules
                    print('loading the model from %s' % load_path)
                   
                    state_dict = torch.load(load_path, map_location=str(self.device))
                    
                    print(state_dict.keys())
                    print(net.cpu().state_dict().keys())
                    
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata
                        
                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                    net.load_state_dict(state_dict, strict=True)
                    net = net.to(self.device)
    
    '''
    def load_networks(self, suffix_name):
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (name, suffix_name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.modules
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict, strict=False)
    '''
    
    def init_model(self, model_path, continue_from_name = 'best'):
        if continue_from_name is None:
            print("ERROR: continue_from_model is None")
            return False
        
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_%s.pth' % (name, continue_from_name)
                load_path = os.path.join(model_path, load_filename)
                
                if os.path.exists(load_path):
                    print("initlizing %s with %s" % (name, load_path))
                    
                    state_dict = torch.load(load_path, map_location = self.device)
                    if hasattr(state_dict, '_metadata'):
                        del state_dict._metadata

                    net = getattr(self, name)
                    if isinstance(net, torch.nn.DataParallel):
                        net = net.modules
                    
                    # patch InstanceNorm checkpoints prior to 0.4
                    for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                        
                    net.load_state_dict(state_dict, strict=False)
                    
    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                if net is None:
                    continue
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    #print(net)
                    #show_params(net, name)
                    show_model(net, name)
                #print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                print('\n\n')
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    