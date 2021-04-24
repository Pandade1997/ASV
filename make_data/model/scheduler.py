from torch.optim import lr_scheduler
import torch
import numpy as np

class WarmupLR(object):
    """A simple wrapper class for learning rate scheduling"""
    def __init__(self, optimizer, factor, d_model, warmup_steps=4000, init_steps = 0, visdom_freq_steps = 1):
        '''
        Parameter
            optimizer: the optimizer for learning rate scheduler
            factor: tunable scalar multiply to learning rate
            d_model: Dimension of model, such as the size of one layer
            warmup_steps: warmup steps
        '''
        self.optimizer = optimizer
        self.factor = factor
        self.init_lr = d_model ** (-0.5)
        self.warmup_steps = warmup_steps
        self.step_num = init_steps
        self.visdom_lr = None
        self.visdom_freq_steps = visdom_freq_steps

    def step(self):
        self._update_lr()
        if self.step_num % self.visdom_freq_steps == 0:
            self._visdom()

    def _update_lr(self):
        self.step_num += 1
        lr = self.factor * self.init_lr * min(self.step_num ** (-0.5),
                                         self.step_num * (self.warmup_steps ** (-1.5)))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def set_lr_factor(self, factor):
        self.factor = factor

    def set_visdom(self, visdom_lr, vis):
        self.visdom_lr = visdom_lr  # Turn on/off visdom of learning rate
        self.vis = vis              # visdom enviroment
        self.vis_opts = dict(title='Learning Rate', ylabel='Leanring Rate', xlabel='step')
        self.vis_window = None
        self.x_axis = torch.from_numpy(np.array(range(1, self.step_num, self.visdom_freq_steps))).long()
        self.y_axis = torch.from_numpy(np.array([self.factor * self.init_lr * min(step ** (-0.5), step * (self.warmup_steps ** (-1.5))) for step in range(1, self.step_num, self.visdom_freq_steps)])).float()
        ##self.x_axis = torch.LongTensor()
        ##self.y_axis = torch.FloatTensor()

    def _visdom(self):
        if self.visdom_lr is not None:
            self.x_axis = torch.cat(
                [self.x_axis, torch.LongTensor([self.step_num])])
            self.y_axis = torch.cat(
                [self.y_axis, torch.FloatTensor([self.optimizer.param_groups[0]['lr']])])
            if self.vis_window is None:
                self.vis_window = self.vis.line(X=self.x_axis, Y=self.y_axis,
                                                opts=self.vis_opts)
            else:
                self.vis.line(X=self.x_axis, Y=self.y_axis, win=self.vis_window,
                              update='replace')

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions锛庛€€
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(step):
            lr_l = 1.0 - max(0, step - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)

    elif opt.lr_policy == 'plateau':
        #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_reduce_factor, threshold=opt.lr_reduce_threshold, patience=opt.step_patience, min_lr = opt.min_lr)
    
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)

    elif opt.lr_policy == 'warmup':
        # opt.factor = 2
        # opt.warmup = 4000
        # optimizer, factor, d_model, warmup_steps=4000, init_steps = 0
        scheduler = WarmupLR(optimizer = optimizer, factor = opt.lr_factor, d_model = opt.d_model, warmup_steps = opt.warmup_step, init_steps = opt.steps, visdom_freq_steps = opt.visdom_freq_steps)
        from visdom import Visdom
        vis = Visdom(server=opt.display_server, port=opt.display_port, env=opt.visdom_id + 'Learning Rate', raise_exceptions=True)
        scheduler.set_visdom(visdom_lr = opt.visdom_lr, vis = vis)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


'''
class WarmupLR:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size, factor, warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        #self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
'''