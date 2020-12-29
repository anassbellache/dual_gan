import sys
import os


import torch
import torch.nn as nn

from models.gen_components import Filtration, PositionwiseFeedForward, ResUnet
import unittest


class TestFilteringCNN(unittest.TestCase):
    def test_shape(self):
        sino = torch.randn(1, 1, 180, 512)
        _ , _ , n_angles, detect_len = sino.shape
        conv_filter = Filtration(detect_len, n_angles, 3, 8)
        filtered_sino = conv_filter(sino)
        self.assertEqual(sino.shape, filtered_sino.shape)


class TestFBPNet(unittest.TestCase):
    def test_shape(self):
        sino = torch.randn(1, 1, 180, 512)
        tomo_slice = torch.randn(1, 1, 512, 512) 
        _ , _, n_angles, detect_len = sino.shape
        fbp = PositionwiseFeedForward(detect_len, n_angles, 64)
        img = fbp(sino)
        self.assertEqual(img.shape, tomo_slice.shape)
        

class TestRefinement(unittest.TestCase):
    def test_shape(self):
        tomo_slice = torch.randn(1, 1, 512, 512) 
        refine = ResUnet(1)
        tomo_slice_enhanced = refine(tomo_slice)
        self.assertEqual(tomo_slice_enhanced.shape, tomo_slice.shape)


class TestFilterFBPCombination(unittest.TestCase):
    def test_shape(self):
        sino = torch.randn(1, 1, 180, 512)
        _, _, n_angles, detect_len = sino.shape
        conv_filter = Filtration(detect_len, n_angles, 3, 8)
        filtered_sino = conv_filter(sino)
        tomo_slice = torch.randn(1, 1, 512, 512) 
        fbp = PositionwiseFeedForward(detect_len, n_angles, 64)
        img = fbp(filtered_sino)
        self.assertEqual(img.shape, tomo_slice.shape)


if __name__ == '__main__':
    unittest.main()