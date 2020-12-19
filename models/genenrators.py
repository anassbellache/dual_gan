
import torch
import torch.nn as nn
from gen_components import Filtration, PositionwiseFeedForward, ResUnet


class GenOne(nn.Module):
    def __init__(self, opt):
        super(GenOne, self).__init__()
        self.filter = Filtration(opt.detecor_size, opt.n_angle, opt.hidden_dim)
        self.fbp = PositionwiseFeedForward(d_len=opt.detector_size, n_angle=opt.n_angle, d_ff= opt.fbp_size)
        self.refine = ResUnet(1)
    
    def forward(self, x):
        x = self.filter(x)
        x = self.fbp(x)
        x = self.refine(x)
        return x


class GenTwo(nn.Module):
    def __init__(self, opt):
        super(GenTwo, self).__init__()
        self.refine = ResUnet(2)
    
    def forward(self, x):
        return self.refine(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), padding=(3,3), stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding=(3,3), stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=(3,3), padding=(3,3), stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=(3,3), padding=(3,3), stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=(3,3), padding=(3,3), stride=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(3,3), stride=2),
            nn.LeakyReLU(0.2),
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1)
        )
    
    def forward(self, x):
        x = self.convolutions(x)
        x = self.fully_connected(x)
        return x
        