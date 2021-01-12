import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
from math import sqrt
from modules_ import CALayer

########################################################################
##########################################################################
class MySCN(nn.Module):
    def __init__(self):
        super(MySCN, self).__init__()
        self.W1 = nn.Conv2d(128, 256, 3, 1, 1, bias=False)
        self.S = nn.Conv2d(256, 256, 3, 1, 1, groups=1, bias=False)
        self.W2 = nn.Conv2d(256, 128, 3, 1, 1, bias=False)
        self.RW1 = CALayer(256,16)
        self.RW2 = CALayer(256,16)
        self.shlu = nn.ReLU(True)
        self.relu = nn.ReLU(True)
      
        self.Thd = Parameter(torch.Tensor(1))
        init.constant_(self.Thd,0)

    def forward(self, input):

        z = self.W1(input)
        tmp = z

        for _ in range(25):
            x = self.RW1(tmp)
            x = self.shlu(x-self.Thd)
            x = self.RW2(x)
            x = self.S(x)
            tmp = x+z
            ctmp = tmp
        c = self.RW2(self.shlu( self.RW1(ctmp) - self.Thd))
        #c = self.shlu(ctmp)
        out = self.W2(c)
        out = self.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.scn = nn.Sequential(MySCN())
        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.advanced = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #print(m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                #print(m.weight)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.relu(self.advanced(out))
        out = self.scn(out)
        out = self.output(out)
        out = torch.add(out,residual)
        return out




if __name__ == '__main__':
    print('running model.py')
    from tensorboardX import SummaryWriter
    import argparse
    parser = argparse.ArgumentParser(description="rlcsc_graph")
    model = Net(parser.parse_args())

    with SummaryWriter(comment='net') as w:
        w.add_graph(model,(torch.Tensor(48,1,33,33),))
