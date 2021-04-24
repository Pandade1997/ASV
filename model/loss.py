import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
IGNORE_ID = -1
#############################################################################################################
############################################# Focal Loss ####################################################
#############################################################################################################
class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5), 
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, device, ignore_index = -1, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        
        self.alpha = self.alpha.to(device)
        self.gamma = gamma
        self.class_num = class_num
        self.device = device
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)

        targets_for_scatter = targets.ne(IGNORE_ID).long() * targets
        
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        #ids = targets.view(-1, 1)
        ids = targets_for_scatter.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        #print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        alpha = alpha.float()
        
        #print('probs size= {}'.format(probs.size()))
        #print(probs)
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        non_pad_mask = targets.ne(IGNORE_ID)
        batch_loss = batch_loss.masked_select(non_pad_mask)

        #print('-----bacth_loss------')
        #print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def cal_performance(pred, gold, criterion, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    Args:
        pred: N x T x C, score before softmax
        gold: N x T
    """

    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1) # (N*T, 1)
    
    loss = cal_loss(pred, gold, criterion, smoothing)

    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    acc = (100.0 * n_correct) / float(non_pad_mask.sum().item())

    return loss, acc

def cal_loss(pred, gold, criterion, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """

    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        #loss = F.cross_entropy(pred, gold,
        #                       ignore_index=IGNORE_ID,
        #                       reduction='elementwise_mean')
        loss = criterion(pred, gold)
        
    return loss
