import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys, os, time, argparse, shutil, scipy, h5py, glob
from models.dual_gan import DualGAN
from data_processor import generate_training_batch, bkgdGen
from options.train_options import TrainOptions
#from skimage.measure import compare_ssim

args = TrainOptions('./config.yaml')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

torch.autograd.set_detect_anomaly(True)

batch_size = args.batch_size
image_size = args.detector_size
disc_iters, gene_iters = args.itd, args.itg
lambda_mse, lambda_adv, lambda_ssim = args.lmse, args.ladv, args.lssim
EPOCHS = 1000

dir_path = os.path.dirname(os.path.realpath(__file__))

itr_out_dir = args.name + '-itrOout'
if os.path.isdir(itr_out_dir):
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir)

if not os.path.exists("outputs"):
    os.makedirs("outputs")

sys.stdout = open('%s%s' % (itr_out_dir, 'iter-prints.log'), 'w')
print('X train: {}\nY train: {}\nX test: {}\nY test: {}'.format(args.xtrain, args.ytrain, args.xtest, args.ytest))

mb_data_iter = bkgdGen(data_generator=generate_training_batch(dir_path + args.filename, batch_size))

dna = DualGAN(args)

for epoch in range(EPOCHS):
    time_git_st = time.time()
    for _ge in range(gene_iters):
        X_batch, Y_batch, labels_batch = mb_data_iter.next()
        dna.set_input((X_batch, Y_batch, labels_batch))
        dna.backward_G()
    
    itr_prints_gen = '[Info] Epoch: %05d, gloss: %.2f (mse%.3f, adv%.3f, ssiù:%.3f), gen_elapse: %.2fs/itr' % (\
                     epoch, dna.loss_G1_tot + dna.loss_G2_tot, (dna.loss_G1_MSE + dna.loss_G2_MSE), 
                     (dna.loss_G1 + dna.loss_G2), (dna.loss_G1_SSIM + dna.loss_G2_SSIM)
                     , (time.time() - time_git_st)/gene_iters)
    
    time_dit_st = time.time()

    for _de in range(disc_iters):
        X_batch, Y_batch, labels_batch = mb_data_iter.next()
        dna.set_input((X_batch, Y_batch, labels_batch))
        dna.backward_D()
    
    print('%s; dloss: %.2f (r%.3f, f%.3f), disc_elapse: %.2fs/itr, gan_elapse: %.2fs/itr' % (itr_prints_gen ,dna.D_loss, dna.loss_D_real.detach().cpu().numpy().mean(),dna.loss_D_fake.detach().cpu().numpy().mean(), (time.time()-time_dit_st)/disc_iters, time.time()-time_git_st))
    