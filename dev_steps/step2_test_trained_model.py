#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import datetime
import time
import glob
import random
import argparse

from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed

from src.networks import Generator, Discriminator
from src.losses import generator_loss, discriminator_loss
from src.data_generators import D_data_generator, G_data_generator
from src.discriminator_augmentations import disc_augment
from src.optimizers import get_optimizers

import pandas as pd

# my import
import src.visualize_tools as vt
import voxelmorph as vxm

# ----------------------------------------------------------------------------
# Set up CLI arguments:
# TODO: replace with a config json. CLI is unmanageably large now.
# TODO: add option for type of discriminator augmentation.

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dataset', type=str, default='OASIS3')
parser.add_argument('--name', type=str, default='experiment_name')
parser.add_argument('--d_train_steps', type=int, default=1)
parser.add_argument('--g_train_steps', type=int, default=1)

# TTUR for training GAN, already set the default values in consistent with appendices
parser.add_argument('--lr_g', type=float, default=1e-4)
parser.add_argument('--lr_d', type=float, default=3e-4)
parser.add_argument('--beta1_g', type=float, default=0.0)
parser.add_argument('--beta2_g', type=float, default=0.9)
parser.add_argument('--beta1_d', type=float, default=0.0)
parser.add_argument('--beta2_d', type=float, default=0.9)

parser.add_argument(
    '--unconditional', dest='conditional', default=True, action='store_false',
)
parser.add_argument(
    '--nonorm_reg', dest='norm_reg', default=True, action='store_false',
) # Not used in the paper.
parser.add_argument(
    '--oversample', dest='oversample', default=True, action='store_false',
)
parser.add_argument(
    '--d_snout', dest='d_snout', default=False, action='store_true',
)
parser.add_argument(
    '--noclip', dest='clip_bckgnd', default=True, action='store_false',
) # should be True, updated
parser.add_argument('--reg_loss', type=str, default='NCC') # One of {'NCC', 'NonSquareNCC'}. Not used NonSquareNCC in paper
parser.add_argument('--losswt_reg', type=float, default=1.0)
parser.add_argument('--losswt_gan', type=float, default=0.1)
parser.add_argument('--losswt_tv', type=float, default=0.00) # Not used in the paper.
parser.add_argument('--losswt_gp', type=float, default=1e-3) # TODO: Gradient penalty for discriminator loss. Need to be adjusted according to dataset. Important!!!
parser.add_argument('--gen_config', type=str, default='ours') # One of {'ours', 'voxelmorph'}.
parser.add_argument('--steps_per_epoch', type=int, default=1000)
parser.add_argument('--rng_seed', type=int, default=33)
parser.add_argument('--start_step', type=int, default=0) # Not used in paper. GAN training is active from the first iteration.
parser.add_argument('--resume_ckpt', type=int, default=0) # checkopint
parser.add_argument('--g_ch', type=int, default=32)
parser.add_argument('--d_ch', type=int, default=64)
parser.add_argument('--init', type=str, default='default') # One of {'default', 'orthogonal'}.
parser.add_argument('--lazy_reg', type=int, default=1) # Not used in the paper.

# my arguments
parser.add_argument('--checkpoint_path', type=str, default='/proj/weights/')
parser.add_argument('--save_path', type=str, default='/proj/outputs/')
parser.add_argument('--split_csv', type=str, default='FS_scans_metadata_with_split.csv')

args = parser.parse_args()

# my CLI
checkpoint_path =args.checkpoint_path #None
save_path = args.save_path #None

# Get CLI information:
epochs = args.epochs
batch_size = args.batch_size
dataset = args.dataset
exp_name = args.name
lr_g = args.lr_g
lr_d = args.lr_d
beta1_g = args.beta1_g
beta2_g = args.beta2_g
beta1_d = args.beta1_d
beta2_d = args.beta2_d
conditional = args.conditional
reg_loss = args.reg_loss
norm_reg = args.norm_reg
oversample = args.oversample
atlas_model = args.gen_config
steps = args.steps_per_epoch
lambda_gan = args.losswt_gan
lambda_reg = args.losswt_reg
lambda_tv = args.losswt_tv
lambda_gp = args.losswt_gp
g_loss_wts = [lambda_gan, lambda_reg, lambda_tv]
start_step = args.start_step
rng_seed = args.rng_seed
resume_ckpt = args.resume_ckpt
d_snout = args.d_snout
clip_bckgnd = args.clip_bckgnd
g_ch = args.g_ch
d_ch = args.d_ch
init = args.init
lazy_reg = args.lazy_reg

# Folder name --> save_folder:
save_folder = (
    ('{}_dataset_{}_eps{}_Gconfig_{}_normreg_{}_lrg{}_lrd{}_cond_{}_'
     'regloss_{}_lbdgan_{}_lbdreg_{}_lbdtv_{}_lbdgp_{}_dsnout_{}_start_{}')
    .format(exp_name, dataset, epochs, atlas_model, norm_reg, lr_g, lr_d,
            conditional, reg_loss, lambda_gan, lambda_reg, lambda_tv,
            lambda_gp, d_snout, start_step)
)

# Append to save_folder if using clip or lazy_reg settings:
if clip_bckgnd:
    save_folder = save_folder + '_clip_{}'.format(clip_bckgnd)

if lazy_reg > 1:
    save_folder = save_folder + '_lazy_{}'.format(lazy_reg)

# ----------------------------------------------------------------------------
# Set RNG seeds

seed(rng_seed)
set_random_seed(rng_seed)
random.seed(rng_seed)

# ----------------------------------------------------------------------------
# Initialize data generators

# Change these if working with new dataset:
if dataset == 'dHCP':
    fpath = './data/dHCP2/npz_files/T2/train/*.npz'
    avg_path = (
        './data/dHCP2/npz_files/T2/linearaverage_100T2_train.npz'
    )
    n_condns = 1
elif dataset == 'pHD':
    fpath = './data/predict-hd/npz_files/train_npz/*.npz'
    avg_path = './data/predict-hd/linearaverageof100.npz'
    n_condns = 3
elif dataset == 'OASIS3_single_cohort':
    main_path = '/proj/OASIS3_atlasGAN/'
    fpath = main_path + 'all_npz/'
    avg_path = main_path + 'linearaverageof100.npz'
    n_condns = 1
else:
    raise ValueError('dataset expected to be dHCP, pHD, or OASIS3_single_cohort')

# extract training data from FS_scans_metadata_with_split.csv
def read_csv(set = "train"):
    path = main_path +  args.split_csv
    df = pd.read_csv(path)
    ids_set = df["ID"][df["Partition"]==set]
    ids_set = [fpath+i.split("_")[0]+"_"+i.split("_")[2]+'.npz' for i in list(ids_set)]
    return ids_set
img_paths = read_csv("test")
# img_paths = glob.glob(fpath)

avg_img = np.load(avg_path)['vol']  # TODO: make generic fname in npz

vol_shape = avg_img.shape # calculate [216, 190, 172] for OASIS3 dataset

Dtrain_data_generator = D_data_generator(
    vol_shape=vol_shape,
    img_list=img_paths,
    oversample_age=oversample,
    batch_size=batch_size,
    dataset=dataset,
)

Gtrain_data_generator = G_data_generator(
    vol_shape=vol_shape,
    img_list=img_paths,
    oversample_age=oversample,
    batch_size=batch_size,
    dataset=dataset,
)

avg_batch = np.repeat(
    avg_img[np.newaxis, ...], batch_size, axis=0,
)[..., np.newaxis]


# ----------------------------------------------------------------------------
# Initialize networks

generator = Generator(
    ch=g_ch,
    atlas_model=atlas_model,
    conditional=conditional,
    normreg=norm_reg,
    clip_bckgnd=clip_bckgnd,
    input_resolution=[*vol_shape, 1],
    initialization=init,
    n_condns=n_condns,
)

discriminator = Discriminator(
    ch=d_ch, conditional=conditional, sn_out=d_snout,
    input_resolution=[*vol_shape, 1],
    initialization=init, n_condns=n_condns,
)

# ----------------------------------------------------------------------------
# Set up optimizers

generator_optimizer, discriminator_optimizer = get_optimizers(
    lr_g, beta1_g, beta2_g, lr_d, beta1_d, beta2_d,
)


# ----------------------------------------------------------------------------
# Set up Checkpoints
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

# restore checkpoint from the latest trained model:
if checkpoint_path:
    checkpoint.restore(
        tf.train.latest_checkpoint(checkpoint_path)
    ).assert_consumed()
else:
    checkpoint.restore(
        tf.train.latest_checkpoint('./training_checkpoints/{}/'.format(save_folder))
    ).assert_consumed()

# extract template generation sub-network
# ----------------------------------------------------------------------------
# Set up generator training loop

@tf.function
def get_inputs(unconditional_inputs, conditional_inputs):
    """If conditionally training, append condition tensor to network inputs."""
    if conditional:
        return unconditional_inputs + conditional_inputs
    else:
        return unconditional_inputs

target_images, target_condns = next(iter(Gtrain_data_generator))

input_images = tf.convert_to_tensor(target_images, dtype=tf.float32)
avg_input = tf.convert_to_tensor(avg_batch, dtype=tf.float32)
input_condns = tf.convert_to_tensor(target_condns, dtype=tf.float32)

# override the input_condns
for i, age in enumerate(range(60,81)):# 60,91
    if dataset == 'OASIS3_single_cohort':
        input_condns_HC = np.expand_dims(np.array((age / 97.1726095890411)), axis=0)

    def predict_and_save(input_condns, disease_conditions, save_plot_path):
        os.makedirs(save_plot_path, exist_ok=True)

        moved_atlases, disp_fields_ms, sharp_atlases, disp_fields = generator(
            get_inputs([input_images, avg_input], [input_condns]),
            # training=False,
        )

        atlasmax = tf.reduce_max(sharp_atlases).numpy() # find the max value
        print("atlasmax = {}".format(atlasmax))

        template = tf.nn.relu(sharp_atlases.numpy().squeeze()).numpy()/ atlasmax  # with normalization
        # template = sharp_atlases.numpy().squeeze() # without normalization
        mid_slices_moving = [np.take(template, template.shape[d] // 2, axis=d) for d in range(3)]
        mid_slices_moving[1] = np.rot90(mid_slices_moving[1], 1)
        mid_slices_moving[2] = np.rot90(mid_slices_moving[2], -1)
        vt.slices(mid_slices_moving, cmaps=['gray'], grid=[1, 3], save=True, show=False,
                  suptitle="age = "+ str(age) + ',HC0/AD1 = ' + str(disease_conditions),
                  save_path=save_plot_path + "age_"+ str(age) + 'disease_' + str(disease_conditions))

        # using another image to guide saving
        vxm.py.utils.save_volfile(template,
                                  save_plot_path + "age_" + str(age) + 'disease_' + str(disease_conditions) + '.nii.gz')
        vt.correct_vox2ras_matrix(save_plot_path + "age_" + str(age) + 'disease_' + str(disease_conditions) + '.nii.gz', reference_nifiti = '/proj/OASIS3_atlasGAN/OASIS30113d4437_align_norm.nii.gz')

    # predict and save
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if dataset == 'OASIS3_single_cohort':
            predict_and_save(input_condns_HC, 0, save_path)
    else:
        if dataset == 'OASIS3_single_cohort':
            predict_and_save(input_condns_HC, 0, './my_plot/')