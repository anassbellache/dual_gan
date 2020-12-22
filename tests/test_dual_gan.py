import unittest
from options.train_options import TrainOptions, BaseOptions
from models.dual_gan import DualGAN

import torch

class TestBaseOptions(unittest.TestCase):
    def test_model_inputs(self):
        opt = BaseOptions('./config.yaml')
        self.assertEqual(opt.batch_size, 3)
        self.assertEqual(opt.detector_size, 1024)
        self.assertEqual(opt.n_angle, 180)
        self.assertEqual(opt.hidden_dim, 100)
        self.assertEqual(opt.fbp_size, 64)
        

class TestTrainOptions(unittest.TestCase):
    def test_train_inputs(self):
        opt = TrainOptions('./config.yaml')
        self.assertEqual(opt.ladv, 20)
        self.assertEqual(opt.lmse, 0.5)
        self.assertEqual(opt.lssim, 2.0)
        self.assertEqual(opt.beta1, 0.5)
        self.assertTrue(opt.isTrain)

class TestDualGan(unittest.TestCase):
    def test_instance(self):
        opt = TrainOptions('./config.yaml')
        gan = DualGAN(opt)
        self.assertIsNotNone(gan)
    
    def test_discriminator(self):
        opt = TrainOptions('./config.yaml')
        gan = DualGAN(opt)
        sinograms = torch.rand(opt.batch_size, 1,  opt.detector_size, opt.detector_size)
        x = gan.netD(sinograms)
        self.assertEqual(x.shape, torch.Size([3,1]))
    



if __name__ == "__main__":
    unittest.main()