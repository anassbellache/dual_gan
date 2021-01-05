import torch
import torch.nn as nn
from torch_radon import Radon, RadonFanbeam
from .gen_components import Filtration, PositionwiseFeedForward, ResUnet
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from .base_model import BaseModel
from .loss import GANLoss, gradient_penalty
import numpy as np

def calculate_feature_output_size(img_size, kernel_size, padding, stride):
    return int((img_size - kernel_size + 2*padding)/stride) + 1


class GenOne(nn.Module):
    def __init__(self, opt):
        super(GenOne, self).__init__()
        self.filter = Filtration(opt.detector_size, opt.n_angle, 3, opt.hidden_dim)
        self.fbp = PositionwiseFeedForward(d_len=opt.detector_size, n_angle=opt.n_angle, d_ff= opt.fbp_size)
        
    
    def forward(self, x):
        x = self.filter(x)
        x = self.fbp(x)
        return x


class GenTwo(nn.Module):
    def __init__(self):
        super(GenTwo, self).__init__()
        self.refine = ResUnet(2)
    
    def forward(self, x):
        return self.refine(x)


class Discriminator(nn.Module):
    def __init__(self, detector_size,):
        super(Discriminator, self).__init__()
        N = calculate_feature_output_size(detector_size, 3, 3, 1)
        N = calculate_feature_output_size(N, 3, 3, 2)
        N = calculate_feature_output_size(N, 3, 3, 1)
        N = calculate_feature_output_size(N, 3, 3, 2)
        N = calculate_feature_output_size(N, 3, 3, 1)

        self.feature_size = calculate_feature_output_size(N, 3, 3, 2) ** 2
        
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
            nn.Linear(256 * self.feature_size, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64,1)
        )
        
    
    def forward(self, x):
        x = self.convolutions(x)
        x = x.contiguous().view((x.shape[0], -1))
        x = self.fully_connected(x)
        return x


class DualGAN(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.netG_1 = GenOne(opt)
        self.netG_2 = GenTwo()
        self.optimizers = []
        self.loss_gan = GANLoss(self.device)
        self.ssim = SSIM(data_range=255, size_average=True, channel=1)
        self.loss_names = ['G1_loss', 'G2_loss', 'D_loss']
        self.visual_names = ['Sino_In', 'Image_In', 'Image_2', 'Image_Real']

        if self.isTrain:
            self.model_names = ['G1', 'G2', 'D']
        else:
            self.model_names = ['G1', 'G2']
        
        self.netG_1.to(self.device)
        self.netG_2.to(self.device)


        if self.isTrain:
            self.netD = Discriminator(opt.detector_size)
            self.netD.to(self.device)
            self.criterionMSE = nn.MSELoss()
            self.optimizer_G1 = torch.optim.Adam(self.netG_1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizer_G2 = torch.optim.Adam(self.netG_2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99))
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_D)
            #angles = np.linspace(0, np.pi, opt.n_angle, endpoint=False)
            """ if opt.parallel:
                self.radon = Radon(opt.detector_size, angles, clip_to_circle=True)
            else:
                self.radon = RadonFanbeam(opt.detector_size, angles, opt.source_distance, 
                                        opt.det_distance, opt.det_count, opt.det_spacing, 
                                        clip_to_circle=True) """
    
    def set_input(self, input):
        sino, img, fbp = input[0], input[1], input[2]
        self.Sino_In = torch.tensor(sino, dtype=torch.float32, requires_grad=True).to(self.device)
        self.Image_Real = torch.tensor(img, dtype=torch.float32, requires_grad=True).to(self.device)
        self.Image_In = torch.tensor(fbp, dtype=torch.float32, requires_grad=True).to(self.device)

    def forward(self):
        self.Image_1 = self.netG_1(self.Sino_In)
    
    def backward_D(self):
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netG_1, False)
        self.set_requires_grad(self.netG_2, False)
        self.optimizer_D.zero_grad()
        img1 = self.netG_1(self.Sino_In)
        in_two = torch.cat([img1, self.Image_In], 1)
        img2 = self.netG_2(in_two)
        pred_true = self.netD(self.Image_Real)
        pred_fake_1 = self.netD(img1.to(self.device))
        pred_fake_2 = self.netD(img2)
        self.loss_D_real =  self.loss_gan(pred_true, True) * self.opt.ladv
        loss_D_G1 = self.loss_gan(pred_fake_1, False) * self.opt.ladv
        loss_D_G2 = self.loss_gan(pred_fake_2, False) * self.opt.ladv
        self.loss_D_fake = loss_D_G1 + loss_D_G2
        self. gradient_penalty = gradient_penalty(self.Image_Real, img2, self.netD, self.device)
        self.D_loss = self.loss_D_real + self.loss_D_fake + gradient_penalty * self.opt.lgrad
        self.D_loss.backward()
        self.optimizer_D.step()
        
    
    def backward_G(self):
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netG_1, True)
        self.set_requires_grad(self.netG_2, True)
        self.optimizer_G1.zero_grad()
        self.optimizer_G2.zero_grad()
        img1 = self.netG_1(self.Sino_In)
        in_two = torch.cat([img1, self.Image_In], 1)
        img2 = self.netG_2(in_two)
        pred_fake_1 = self.netD(img1)
        pred_fake_2 = self.netD(img2)
        #sinoG1 = self.radon.forward(img1)
        #sinoG2 = self.radon.forward(img2)

        self.loss_G1 = self.loss_gan(pred_fake_1, True) * self.opt.ladv
        self.loss_G2 = self.loss_gan(pred_fake_2, True) * self.opt.ladv        
        self.loss_G1_MSE = self.criterionMSE(img1, self.Image_Real) * self.opt.lmse
        self.loss_G2_MSE = self.criterionMSE(img2, self.Image_Real) * self.opt.lmse
        self.loss_G1_SSIM = (1 - self.ssim(img1, self.Image_Real)) * self.opt.lssim
        self.loss_G2_SSIM = (1 - self.ssim(img2, self.Image_Real)) * self.opt.lssim
        #self.loss_sino_G1 = self.criterionMSE(self.Sino_In, sinoG1)
        #self.loss_sino_G2 = self.criterionMSE(self.Sino_In, sinoG2)
        self.loss_G1_tot = self.loss_G1 + self.loss_G1_MSE + self.loss_G1_SSIM #+ self.loss_sino_G1
        self.loss_G2_tot = self.loss_G2 + self.loss_G2_MSE + self.loss_G2_SSIM #+ self.loss_sino_G2

        self.loss_G1_tot.backward(retain_graph=True)
        self.loss_G2_tot.backward()
        
        self.optimizer_G1.step()
        self.optimizer_G2.step()



if __name__ == "__main__":
    batch_size = 3
    sinograms = torch.rand(batch_size, 1,  1024, 1024)
    print(sinograms.shape)
    disc = Discriminator(1024)
    print(disc.feature_size)
    x = disc(sinograms)
    print(x.shape)
    

    


        


    