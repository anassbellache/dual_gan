from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import torch
import torch.nn as nn
import functools


class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
    
    def __call__(self, prediction, target_is_real):
        if target_is_real:
            loss = -prediction.mean()
        else:
            loss = prediction.mean()

        return loss

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
    
    def __call__(self, X, Y):
        return 1 - ssim( X, Y, data_range=255, size_average=True)

