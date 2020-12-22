import sys
import os


import torch
import torch.nn as nn

from models.gen_components import Filtration, PositionwiseFeedForward, ResUnet
import unittest


class TestFilteringCNN(unittest.TestCase):
    def test_shape(self):
        sino = torch.randn(1, 1, 1024, 180)
        _ , _ , detect_len, n_angles = sino.shape
        conv_filter = Filtration(detect_len, n_angles, 3, 8)
        filtered_sino = conv_filter(sino)
        self.assertEqual(sino.shape, filtered_sino.shape)


class TestFBPNet(unittest.TestCase):
    def test_shape(self):
        sino = torch.randn(1, 180, 1024)
        tomo_slice = torch.randn(1, 1024, 1024) 
        _ ,n_angles, detect_len = sino.shape
        fbp = PositionwiseFeedForward(detect_len, n_angles, 64)
        img = fbp(sino)
        self.assertEqual(img.shape, tomo_slice.shape)
        

class TestRefinement(unittest.TestCase):
    def test_shape(self):
        tomo_slice = torch.randn(1, 1, 1024, 1024) 
        refine = ResUnet(1)
        tomo_slice_enhanced = refine(tomo_slice)
        self.assertEqual(tomo_slice_enhanced.shape, tomo_slice.shape)

if __name__ == '__main__':
    unittest.main()