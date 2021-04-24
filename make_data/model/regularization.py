import torch

class Regularization(torch.nn.Module):
    def __init__(self, weight_decay, p = 2.0 ):

        super(Regularization, self).__init__()
        assert weight_decay > 0 and p > 0.0, "Unsupport Regularization params: weight = %.2f, p = %.2f" % (weight_decay, p)

        self.weight_decay = weight_decay
        self.p            = p
 
    def to(self, device):
        self.device = device
        super().to(device)
        return self
 
    def forward(self, model):
        reg_loss = self.regularization_loss(weight_list = self.get_weight(model), weight_decay = self.weight_decay, p = self.p)
        return reg_loss
 
    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
 
    def regularization_loss(self, weight_list, weight_decay, p = 2):
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)

        reg_loss=0
        for name, w in weight_list:
            l2_reg   = torch.norm(w, p = p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss = weight_decay * reg_loss
        return reg_loss
 
    def weight_info(self, model):
        print("---------------regularization weight---------------")
        for name, w in self.get_weight(model):
            print(name)
        print("---------------------------------------------------")