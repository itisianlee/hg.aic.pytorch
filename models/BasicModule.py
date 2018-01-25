# coding:utf8
import torch
import time


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path, device=0):
        data = torch.load(path, map_location=lambda storage, loc: storage.cuda(device))
        return self.load_state_dict(data)

    def save(self, name=None):
        prefix = 'checkpoints/'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S')
        path = prefix + name + '.model'
        data = self.state_dict()
        torch.save(data, path)
        return path

    def get_optimizer(self, lr=2.5e-4, weight_decay=0, momentum=0):
        optimizer = torch.optim.RMSprop(self.parameters(),
                                        lr=lr,
                                        momentum=momentum,
                                        weight_decay=weight_decay)
        return optimizer
