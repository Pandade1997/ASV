import torch.nn as nn
import torch 
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(__file__))
from show import show_params
from .regularization import Regularization

class FTB(nn.Module):

    def __init__(self, input_dim=257, in_channel=9, r_channel=5):

        super(FTB, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channel, r_channel, kernel_size=[1,1]),
                        nn.BatchNorm2d(r_channel),
                        nn.ReLU()
            )
        
        self.conv1d = nn.Sequential(
                        nn.Conv1d(r_channel*input_dim, in_channel, kernel_size=9,padding=4),
                        nn.BatchNorm1d(in_channel),
                        nn.ReLU()
            )
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channel*2, in_channel, kernel_size=[1,1]),
                        nn.BatchNorm2d(in_channel),
                        nn.ReLU()
            )

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]
        '''
        # T-F attention        
        conv1_out = self.conv1(inputs)
        B, C, D, T= conv1_out.size()
        reshape1_out = torch.reshape(conv1_out,[B, C*D, T])
        conv1d_out = self.conv1d(reshape1_out)
        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel,1,T])
        
        # now is also [B,C,D,T]
        att_out = conv1d_out*inputs
        
        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, inputs], 1)
        outputs = self.conv2(cat_out)
        return outputs


class InforComu(nn.Module):
        
        def __init__(self, src_channel, tgt_channel):
            
            super(InforComu, self).__init__()
            self.comu_conv = nn.Conv2d(src_channel, tgt_channel, kernel_size=(1,1))
        
        def forward(self, src, tgt):
            
            outputs=tgt*torch.tanh(self.comu_conv(src))
            return outputs


class GLayerNorm2d(nn.Module):
    
    def __init__(self, in_channel, eps=1e-12):
        super(GLayerNorm2d, self).__init__()
        self.eps = eps 
        self.beta = nn.Parameter(torch.ones([1, in_channel,1,1]))
        self.gamma = nn.Parameter(torch.zeros([1, in_channel,1,1]))
    
    def forward(self,inputs):
        mean = torch.mean(inputs,[1,2,3], keepdim=True)
        var = torch.var(inputs,[1,2,3], keepdim=True)
        outputs = (inputs - mean)/ torch.sqrt(var+self.eps)*self.beta+self.gamma
        return outputs

class TSB(nn.Module):

    def __init__(self, input_dim=257, channel_amp=9, channel_phase=8):
        super(TSB, self).__init__()
        
        self.ftb1 = FTB(input_dim=input_dim,
                        in_channel=channel_amp,
                    )
        self.amp_conv1 = nn.Sequential(
                        nn.Conv2d(channel_amp, channel_amp, kernel_size=(5,5), padding=(2,2)),
                        nn.BatchNorm2d(channel_amp),
                        nn.ReLU()
                    )
        self.amp_conv2 = nn.Sequential(
                        nn.Conv2d(channel_amp, channel_amp, kernel_size=(1,25), padding=(0,12)),
                        nn.BatchNorm2d(channel_amp),
                        nn.ReLU()
                    )
        self.amp_conv3 = nn.Sequential(
                        nn.Conv2d(channel_amp, channel_amp, kernel_size=(5,5), padding=(2,2)),
                        nn.BatchNorm2d(channel_amp),
                        nn.ReLU()
                    )
        
        self.ftb2 = FTB(input_dim=input_dim,
                        in_channel=channel_amp,
                    )

        self.phase_conv1 = nn.Sequential(
                        nn.Conv2d(channel_phase, channel_phase, kernel_size=(5,5), padding=(2,2)),
                        GLayerNorm2d(channel_phase),
                    )
        self.phase_conv2 = nn.Sequential(
                        nn.Conv2d(channel_phase, channel_phase, kernel_size=(1,25), padding=(0,12)),
                        GLayerNorm2d(channel_phase),
                    )

        self.p2a_comu = InforComu(channel_phase, channel_amp)
        self.a2p_comu = InforComu(channel_amp, channel_phase)

    def forward(self, amp, phase):
        '''
        amp should be [Batch, Ca, Dim, Time]
        amp should be [Batch, Cr, Dim, Time]
        
        '''
        
        amp_out1 = self.ftb1(amp)
        amp_out2 = self.amp_conv1(amp_out1)
        amp_out3 = self.amp_conv2(amp_out2)
        amp_out4 = self.amp_conv3(amp_out3)
        amp_out5 = self.ftb2(amp_out4)
        
        phase_out1 = self.phase_conv1(phase)
        phase_out2 = self.phase_conv2(phase_out1)
        
        amp_out = self.p2a_comu(phase_out2, amp_out5)
        phase_out = self.a2p_comu(amp_out5, phase_out2)
        
        return amp_out, phase_out

class PHASEN(nn.Module):

    def __init__(self, num_bin = 257, num_tsb = 3, channel_amp = 24, channel_phase = 12, rnn_nums = 300, regularization_weight = 0.0, regularization_p = 2.0, fixed = False):
        super(PHASEN, self).__init__()

        # ( num_block, 2, num_bin, num_frame )
        self.num_tsb    = num_tsb
        self.feat_dim   = num_bin
        self.fixed      = fixed
        
        self.amp_conv1 = nn.Sequential(
                                nn.Conv2d(2, channel_amp, kernel_size=[7,1], padding=(3,0)),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),
                                nn.Conv2d(channel_amp, channel_amp, kernel_size=[1,7], padding=(0,3)),
                                nn.BatchNorm2d(channel_amp),
                                nn.ReLU(),)
        self.phase_conv1 = nn.Sequential(
                                nn.Conv2d(2, channel_phase, kernel_size=[3,5], padding=(1,2)),
                                nn.Conv2d(channel_phase, channel_phase,  kernel_size=[3,25], padding=(1, 12)),)
        self.tsbs = nn.ModuleList()

        for idx in range(self.num_tsb):
            self.tsbs.append( TSB(input_dim=self.feat_dim, channel_amp=channel_amp, channel_phase=channel_phase))

        self.amp_conv2 = nn.Sequential(nn.Conv2d(channel_amp, 8, kernel_size=[1, 1]), nn.BatchNorm2d(8), nn.ReLU(),)
        self.phase_conv2 = nn.Sequential(nn.Conv1d(channel_phase,2,kernel_size=[1,1]))
        self.rnn = nn.GRU(self.feat_dim * 8, rnn_nums, bidirectional=True)
        self.fcs = nn.Sequential(nn.Linear(rnn_nums*2,600), nn.ReLU(), nn.Linear(600,600), nn.ReLU(), nn.Linear(600,514//2), nn.Sigmoid())

        if regularization_weight > 0.0 and not fixed:
            self.regularization = Regularization(weight_decay = regularization_weight, p = regularization_p)
        else:
            self.regularization = None

        #show_params(self)
    
    def get_regularization_loss(self):
        
        reg_loss = 0.0

        if self.regularization is not None:
            reg_loss = reg_loss + self.regularization(self)
        
        return reg_loss

    def get_trainable_params(self, weight_decay = 0.0, print_info = False):
        if print_info:
            print("####### Trainable Parames of PHASEN")
        weights, biases = [], []
        if not self.fixed:
            if print_info:
                print("phasen = ")
            phasen_parem_names = []
            for name, param in self.named_parameters():
                phasen_parem_names.append(name)
                if 'bias' in name:
                    biases += [param]
                else:
                    weights += [param]
            if print_info:
                print(phasen_parem_names)
        if len(weights) < 1:
            return None
        params = [{'params': weights,'weight_decay': weight_decay,}, {'params': biases, 'weight_decay': 0.0,}]
        return params

    def forward_in(self, inputs):
        #cmp_spec = inputs.permute( [0, 2, 3, 1] ) # ( num_block, 2, num_bin, num_frame )
        cmp_spec = inputs
        
        amp_spec = torch.sqrt( torch.abs(cmp_spec[:,0])**2 + torch.abs(cmp_spec[:,1])**2, ) # ( num_block, num_bin, num_frame )
        amp_spec = torch.unsqueeze(amp_spec, 1)  #( num_block, 1, num_bin, num_frame )
        
        spec    = self.amp_conv1(cmp_spec)
        phase   = self.phase_conv1(cmp_spec)
        s_spec  = spec
        s_phase = phase
        for idx, layer in enumerate(self.tsbs):
            if idx != 0:
                spec += s_spec
                phase += s_phase
            spec, phase = layer(spec, phase)
        spec = self.amp_conv2(spec)

        spec=  torch.transpose(spec, 1,3)
        B, T, D, C = spec.size()
        spec = torch.reshape(spec, [B, T, D*C])
        spec = self.rnn(spec)[0]
        spec = self.fcs(spec)
        
        spec = torch.reshape(spec, [B,T,D,1]) 
        spec = torch.transpose(spec, 1,3)
        
        phase = self.phase_conv2(phase)
        # norm to 1
        phase = phase / (torch.sqrt(torch.abs(phase[:,0])**2 + torch.abs(phase[:,1])**2) + 1e-8).unsqueeze(1)
    
        est_spec = amp_spec * spec * phase # ( num_block, 2, num_bin, num_frame )

        return est_spec                    # ( num_block, 2, num_bin, num_frame )

    def forward(self, inputs, out_wav = False):
        # input = ( num_block, num_frame, 2, num_bin )
        if self.fixed:
            with torch.no_grad():
                return self.forward_in(inputs)
        else:
            return self.forward_in(inputs)
    
    def loss(self, est, labels, mode='Mix'):
        '''
        mode == 'Mix'
            est:    [B, F*2, T]
            labels: [B, F*2,T]
        mode == 'SiSNR'
            est:    [B, T]
            labels: [B, T]
        '''
        if  len(est.size()) == 4:
            est = torch.cat([est[:,0], est[:,1]], 1) # ( num_block, num_bin * 2, num_frame )
        if mode == 'SiSNR':
            if labels.dim() == 3:
                labels = torch.squeeze(labels,1)
            if est.dim() == 3:
                est = torch.squeeze(est,1)
            return -si_snr(est, labels)         
        elif mode == 'Mix':
            b, d, t = est.size()
            gth_cspec = self.stft(labels)
            est_cspec = est  
            gth_mag_spec = torch.sqrt(gth_cspec[:, :self.feat_dim, :]**2 +gth_cspec[:, self.feat_dim:, :]**2)
            est_mag_spec = torch.sqrt(est_cspec[:, :self.feat_dim, :]**2 +est_cspec[:, self.feat_dim:, :]**2)
            
            # power compress 
            gth_cprs_mag_spec = gth_mag_spec**0.3
            est_cprs_mag_spec = est_mag_spec**0.3
            amp_loss          = F.mse_loss(gth_cprs_mag_spec, est_cprs_mag_spec)*d
            compress_coff     = (gth_cprs_mag_spec/(1e-8 + gth_mag_spec)).repeat(1,2,1)
            phase_loss        = F.mse_loss( gth_cspec * compress_coff, est_cspec * compress_coff)*d
            
            all_loss = amp_loss*0.5 + phase_loss*0.5
            return all_loss, amp_loss, phase_loss

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True)
    data = data - mean
    return data

def l2_norm(s1, s2):
    #norm = torch.sqrt(torch.sum(s1*s2, 1, keepdim=True))
    #norm = torch.norm(s1*s2, 1, keepdim=True)
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm  = l2_norm(s1, s2)
    s2_s2_norm  = l2_norm(s2, s2)
    s_target    =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie     = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm  = l2_norm(e_nosie, e_nosie)
    snr         = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

def test_ftb():
    torch.manual_seed(20)
    inputs = torch.randn([10,9, 257, 100])
    net = FTB()
    print(net(inputs).shape)

def test_tsb():
    torch.manual_seed(20)
    inputs = torch.randn([10,9, 257, 100])
    phase = torch.randn([10,8,257,100])
    net = TSB()
    out1, out2 = net(inputs, phase)
    print(out1.shape, out2.shape)

def test_PHASEN():
    torch.manual_seed(20)
    inputs = torch.randn([10,1,16000*4])
    wav_label = torch.randn([10, 16000*4])
    net = PHASEN()
    est_spec, est_wav = net(inputs)
    print(est_spec.shape, est_wav.shape)
    sisnr = net.loss(est_wav, wav_label, mode='SiSNR')
    Mix = net.loss(est_spec, wav_label, mode='Mix')
    print('mix:',Mix, 'SNR:', sisnr)

#test_ftb()
#test_tsb()
#test_PHASEN()
if __name__ == '__main__':
    test_PHASEN()

