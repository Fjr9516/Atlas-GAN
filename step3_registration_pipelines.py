#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import random
import argparse

from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed

from src.networks import Generator
import pandas as pd
import tensorflow.keras.layers as KL

# my import
import src.visualize_tools as vt
import voxelmorph as vxm
from src.networks import conv_block
from neurite.tf.layers import MeanStream
from voxelmorph.tf.layers import SpatialTransformer, VecInt, RescaleTransform

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
)  # Not used in the paper.
parser.add_argument(
    '--oversample', dest='oversample', default=True, action='store_false',
)
parser.add_argument(
    '--d_snout', dest='d_snout', default=False, action='store_true',
)
parser.add_argument(
    '--noclip', dest='clip_bckgnd', default=True, action='store_false',
)  # should be True, updated
parser.add_argument('--reg_loss', type=str,
                    default='NCC')  # One of {'NCC', 'NonSquareNCC'}. Not used NonSquareNCC in paper
parser.add_argument('--losswt_reg', type=float, default=1.0)
parser.add_argument('--losswt_gan', type=float, default=0.1)
parser.add_argument('--losswt_tv', type=float, default=0.00)  # Not used in the paper.
parser.add_argument('--losswt_gp', type=float,
                    default=1e-3)  # TODO: Gradient penalty for discriminator loss. Need to be adjusted according to dataset. Important!!!
parser.add_argument('--gen_config', type=str, default='ours')  # One of {'ours', 'voxelmorph'}.
parser.add_argument('--steps_per_epoch', type=int, default=1000)
parser.add_argument('--rng_seed', type=int, default=33)
parser.add_argument('--start_step', type=int,
                    default=0)  # Not used in paper. GAN training is active from the first iteration.
parser.add_argument('--resume_ckpt', type=int, default=0)  # checkopint
parser.add_argument('--g_ch', type=int, default=32)
parser.add_argument('--d_ch', type=int, default=64)
parser.add_argument('--init', type=str, default='default')  # One of {'default', 'orthogonal'}.
parser.add_argument('--lazy_reg', type=int, default=1)  # Not used in the paper.

# my arguments
parser.add_argument('--checkpoint_path', type=str,
                    default='/home/fjr/data/trained_models/Atlas-GAN/training_checkpoints/gploss_1e_4_dataset_OASIS3_eps200_Gconfig_ours_normreg_True_lrg0.0001_lrd0.0003_cond_True_regloss_NCC_lbdgan_0.1_lbdreg_1.0_lbdtv_0.0_lbdgp_0.0001_dsnout_False_start_0_clip_True/')
parser.add_argument('--save_path', type=str, default='/home/fjr/data/trained_models/Atlas-GAN/my_plot_1e-4/')

args = parser.parse_args()

# my CLI
checkpoint_path = args.checkpoint_path  # None
save_path = args.save_path  # None

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
elif dataset == 'OASIS3':
    main_path = '/media/fjr/My Passport/data/OASIS3/'  # /data/OASIS3/ or /proj/OASIS3_atlasGAN/ or /media/fjr/My Passport/data/OASIS3/
    fpath = main_path + 'all_npz/'
    avg_path = main_path + 'linearaverageof100.npz'
    n_condns = 3
else:
    raise ValueError('dataset expected to be dHCP, pHD or OASIS3')

# define a registration model
def Registration(
    ch=32,
    normreg=False,
    input_resolution=[160, 192, 160, 1],
    output_vel=False,
):
    image_inputs = tf.keras.layers.Input(shape=input_resolution)
    new_atlas    = tf.keras.layers.Input(shape=input_resolution)

    init = None
    vel_init = tf.keras.initializers.RandomNormal(
        mean=0.0,
        stddev=1e-5,
    )

    # Registration network. Taken from vxm:
    # Encoder:
    inp = KL.concatenate([image_inputs, new_atlas])
    d1 = conv_block(inp, ch, stride=2, instancen=normreg, init=init)
    d2 = conv_block(d1, ch, stride=2, instancen=normreg, init=init)
    d3 = conv_block(d2, ch, stride=2, instancen=normreg, init=init)
    d4 = conv_block(d3, ch, stride=2, instancen=normreg, init=init)

    # Bottleneck:
    dres = conv_block(d4, ch, instancen=normreg, init=init)

    # Decoder:
    d5 = conv_block(dres, ch, mode='up', instancen=normreg, init=init)
    d5 = KL.concatenate([d5, d3])

    d6 = conv_block(d5, ch, mode='up', instancen=normreg, init=init)
    d6 = KL.concatenate([d6, d2])

    d7 = conv_block(d6, ch, mode='up', instancen=normreg, init=init)
    d7 = KL.concatenate([d7, d1])

    d7 = conv_block(
        d7, ch, mode='const', instancen=normreg, init=init,
    )
    d7 = conv_block(
        d7, ch, mode='const', instancen=normreg, init=init,
    )
    d7 = conv_block(d7, ch//2, mode='const', activation=False, init=init)

    # Get velocity field:
    d7 = tf.pad(d7, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")

    vel = KL.Conv3D(
        filters=3,
        kernel_size=3,
        padding='valid',
        use_bias=True,
        kernel_initializer=vel_init,
        name='vel_field',
    )(d7)

    # Get diffeomorphic displacement field:
    diff_field = VecInt(method='ss', int_steps=5, name='def_field')(vel)

    # # Get moving average of deformations:
    # diff_field_ms = MeanStream(name='mean_stream', cap=100)(diff_field)
    #
    # # compute regularizers on diff_field_half for efficiency:
    # diff_field_half = 1.0 * diff_field
    vel_field  = RescaleTransform(2.0, name='flowup_vel_field')(vel)
    diff_field = RescaleTransform(2.0, name='flowup')(diff_field)
    moved_atlas = SpatialTransformer()([new_atlas, diff_field])

    if output_vel:
        ops = [moved_atlas, diff_field, vel_field, vel]
    else:
        ops = [moved_atlas, diff_field, vel_field]

    return tf.keras.Model(
        inputs=[image_inputs, new_atlas],
        outputs=ops,
    )

def WeightLoading_Registration_Block(main_path, checkpoint_path, n_condns, output_vel=False):
    # extract training data from FS_scans_metadata_with_split.csv
    # def read_csv(set="train"):
    #     path = f'{main_path}FS_scans_metadata_with_split.csv'
    #     df = pd.read_csv(path)
    #     ids_set = df["ID"][df["Partition"] == set]
    #     ids_set = [f'{main_path}all_npz/' + i.split("_")[0] + "_" + i.split("_")[2] + '.npz' for i in list(ids_set)]
    #     return ids_set
    #
    # img_paths = read_csv("test")
    avg_path = main_path + 'linearaverageof100.npz'

    avg_img = np.load(avg_path)['vol']  # TODO: make generic fname in npz

    vol_shape = avg_img.shape  # calculate [208, 176, 160] for OASIS3 dataset

    # avg_batch = np.repeat(
    #     avg_img[np.newaxis, ...], batch_size, axis=0,
    # )[..., np.newaxis]

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

    # ----------------------------------------------------------------------------
    # Set up Checkpoints
    checkpoint = tf.train.Checkpoint(
        generator=generator,
    )

    # restore checkpoint from the latest trained model:
    if checkpoint_path:
        checkpoint.restore(
            tf.train.latest_checkpoint(checkpoint_path)
        ).expect_partial()
    else:
        raise ValueError('Testing phase, please provide checkpoint path!')

    registration_model = Registration(
        ch=g_ch,
        normreg=norm_reg,
        input_resolution=[*vol_shape, 1],
        output_vel=output_vel
    )
    # weights_list = generator.get_weights() # 117 long

    # construct weight layer names
    def get_layers_name_with_weights(generator):
        weights_layers = []
        for layer_id, layer in enumerate(generator.layers):
            if len(layer.trainable_weights) > 0 or len(layer.non_trainable_weights) > 0:
                # print(f'{layer_id}th layer, name = {layer.name}, '
                #       f'trainable_weights = {len(layer.trainable_weights)}, non_trainable_weights = {len(layer.non_trainable_weights)}')
                # repeat_times = len(layer.trainable_weights) + len(layer.non_trainable_weights)
                # for i in range(repeat_times):
                #     weights_layers.append(layer.name)
                weights_layers.append(layer.name)
        return weights_layers

    weights_layers_generator = get_layers_name_with_weights(generator)
    weights_layers_registration=get_layers_name_with_weights(registration_model)
    # load weight layer by layer, references: https://www.gcptutorials.com/post/how-to-get-weights-of-layers-in-tensorflow
    # https://stackoverflow.com/questions/43702323/how-to-load-only-specific-weights-on-keras
    start_generator = weights_layers_generator.index('conv3d_12')
    for i, layer in enumerate(weights_layers_registration):
        generator_layer = weights_layers_generator[start_generator+i]
        print(f'Loading weights for layer {layer} from generator layer {generator_layer}')
        registration_model.get_layer(layer).set_weights(generator.get_layer(generator_layer).get_weights())

    print("loading end")
    return registration_model


def template_generator(main_path,  input_age, checkpoint_path):
    input_condns  = np.expand_dims(np.array((input_age / 97.1726095890411)), axis=0)
    n_condns = 1

    avg_path = main_path + 'linearaverageof100.npz'

    avg_img = np.load(avg_path)['vol']  # TODO: make generic fname in npz

    vol_shape = avg_img.shape  # calculate [216, 190, 172] for OASIS3 dataset

    avg_batch = np.repeat(
        avg_img[np.newaxis, ...], batch_size, axis=0,
    )[..., np.newaxis]

    avg_input = tf.convert_to_tensor(avg_batch, dtype=tf.float32)
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

    # ----------------------------------------------------------------------------
    # Set up Checkpoints
    checkpoint = tf.train.Checkpoint(
        generator=generator,
    )

    # restore checkpoint from the latest trained model:
    if checkpoint_path:
        checkpoint.restore(
            tf.train.latest_checkpoint(checkpoint_path)
        ).expect_partial()
    else:
        raise ValueError('Testing phase, please provide checkpoint path!')

    # ----------------------------------------------------------------------------
    # Set up generator training loop

    @tf.function
    def get_inputs(unconditional_inputs, conditional_inputs):
        """If conditionally training, append condition tensor to network inputs."""
        if conditional:
            return unconditional_inputs + conditional_inputs
        else:
            return unconditional_inputs

    _, _, sharp_atlases, _ = generator(
            get_inputs([avg_input, avg_input], [input_condns]),
            training=False,
        )

    atlasmax = tf.reduce_max(sharp_atlases).numpy()  # find the max value
    print("atlasmax = {}".format(atlasmax))

    return tf.nn.relu(sharp_atlases.numpy().squeeze()).numpy() / atlasmax  # with normalization

def registrator(moving_image, fixed_image, checkpoint_path, main_path, n_condns, output_vel=False):
    tf.keras.backend.clear_session()
    fixed_image = fixed_image[np.newaxis, ..., np.newaxis]
    moving_image = moving_image[np.newaxis, ..., np.newaxis]

    fixed_image = tf.convert_to_tensor(fixed_image, dtype=tf.float32)
    moving_image = tf.convert_to_tensor(moving_image, dtype=tf.float32)

    registration_model = WeightLoading_Registration_Block(main_path, checkpoint_path, n_condns,  output_vel=output_vel)

    opts = registration_model([fixed_image, moving_image])

    return opts
def registrator_synthmorph(fixed_image, moving_image):
    tf.keras.backend.clear_session()

    fixed_image = fixed_image[np.newaxis, ..., np.newaxis]
    moving_image = moving_image[np.newaxis, ..., np.newaxis]

    fixed_image = tf.convert_to_tensor(fixed_image, dtype=tf.float32)
    moving_image = tf.convert_to_tensor(moving_image, dtype=tf.float32)

    model = vxm.networks.VxmDense.load('./models/brains-dice-vel-0.5-res-16-256f.h5')
    model.summary()

    moved, vel_half = model.predict([moving_image, fixed_image])
    return moved, vel_half