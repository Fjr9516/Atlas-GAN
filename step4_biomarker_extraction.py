#!/usr/bin/env python
# -*- coding:utf-8 -*-
# import
from pipline_AtlasGAN_registration import registrator
import numpy as np
import os
import visualize_tools as vt
import disentangle_tools as dt
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import voxelmorph as vxm
import seaborn as sns
import helmholtz_decom as hd
import statistics

from scipy.stats import ttest_ind
import itertools

# functions
def calcu_biomarkers(labels, segmentation, wk, u0, age_gap, quantile = 0):
    x, y, z, d = u0.shape

    # calculate tk_3d
    u0_norm = np.sum(u0 ** 2, axis=3, keepdims=True)
    numerator = np.sum(wk * u0, axis=3, keepdims=True)

    if quantile != 0:
        # set up a threshold
        th = np.quantile(u0_norm, quantile)
        print(f'{quantile} quantile for u0_norm is {th}')
    else:
        th = 0
    tk_3d = np.divide(numerator, u0_norm, out=np.zeros_like(numerator), where=u0_norm > 0)

    # calculate w_ad
    tk_3d_broadcasted = np.broadcast_to(tk_3d, (x, y, z, d))
    w_ad = wk - tk_3d_broadcasted * u0

    mask = np.zeros_like(segmentation, dtype=np.uint8)
    if labels == "wholebrain":
        mask = np.ones_like(segmentation, dtype=np.uint8)
        mask[segmentation == 0] = 0
        print(f"# mask = {mask.sum()}")
        mask[u0_norm.squeeze() <= th] = 0
        print(f"# mask = {mask.sum()}")
    elif labels == "diff_ventricles":
        labels_ventricles = [4, 14, 15, 43]
        mask = apply_mask_diff(tk_3d, labels_ventricles, segmentation[0], segmentation[1])[1]
        print(f"# mask = {mask.sum()}")
        mask[u0_norm.squeeze() <= th] = 0
        print(f"# mask = {mask.sum()}")
    elif labels == "diff_hippo":
        labels_hippocampus = [17, 53, 18, 54]
        mask = apply_mask_diff(tk_3d, labels_hippocampus, segmentation[0], segmentation[1])[1]
        print(f"# mask = {mask.sum()}")
        mask[u0_norm.squeeze() <= th] = 0
        print(f"# mask = {mask.sum()}")
    else:
        for label in labels:
            mask[segmentation == label] = 1
        print(f"# mask = {mask.sum()}")
        mask[u0_norm.squeeze() <= th] = 0
        print(f"# mask = {mask.sum()}")

    # calculate masked tk_3d and w_ad
    def apply_mask(input_array, mask):
        # create a new array with the same shape as the input array
        masked_array = np.zeros_like(input_array)
        # set the masked pixels to the input values
        masked_array[mask == 1] = input_array[mask == 1]
        return masked_array

    masked_tk_3d = apply_mask(tk_3d, mask)
    masked_w_ad = apply_mask(w_ad, mask)

    print(f'Min-Max masked_tk_3d = {np.min(masked_tk_3d)}-{np.max(masked_tk_3d)}')
    # print(f'unique values are {np.unique(masked_tk_3d)}, len is {np.unique(masked_tk_3d).shape}')
    print(f'Average unique sum/ len-1 = {np.sum(np.unique(masked_tk_3d))/(np.unique(masked_tk_3d).shape[0]-1)}')

    t_hc = np.sum(np.unique(masked_tk_3d))/(np.unique(masked_tk_3d).shape[0]-1) # way 1
    t_hc = np.sum(masked_tk_3d) / mask.sum() # way 2
    print(f't_hc={t_hc}')
    U, V, W = masked_w_ad[..., 0], masked_w_ad[..., 1], masked_w_ad[..., 2]
    t_ad = (np.sum(np.sqrt(U ** 2 + V ** 2 + W ** 2)) / mask.sum()) / age_gap
    print(f't_ad={t_ad}')
    return t_hc, t_ad

def crop_image_vector(arr):
    arr = np.transpose(arr, (2,1,0,3))
    # Get the center of the image along each axis
    center_x = arr.shape[0] // 2
    center_y = arr.shape[1] // 2
    center_z = arr.shape[2] // 2
    # Calculate the starting and ending indices for the crop along each axis
    start_x = center_x - 40
    end_x = center_x + 40
    start_y = center_y - 40
    end_y = center_y + 40
    start_z = center_z - 48
    end_z = center_z + 48
    # Crop the image
    cropped = arr[start_x:end_x, start_y:end_y, start_z:end_z, :]
    return cropped

def draw_1x3_plot_ADHC(df, save_name = 'ADHC_markers_updated'):
    plt.rcParams.update({'font.size': 14})
    # Create a 2x2 figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    # Plot first subplot
    sns.boxplot(x='disease_condition', y='age_shifts_wholebrain', data=df, ax=axs[0])
    axs[0].set_title('Whole brain')
    axs[0].set_ylabel('Age Shifts Differences')
    axs[0].set_xlabel('Disease Condition')

    # Plot second subplot
    sns.boxplot(x='disease_condition', y='age_shifts_ventri', data=df, ax=axs[1])
    axs[1].set_title('Ventricles')
    axs[1].set_ylabel('Age Shifts Differences')
    axs[1].set_xlabel('Disease Condition')

    # # Plot third subplot
    # sns.boxplot(x='disease_condition', y='age_shifts_hippo', data=df, ax=axs[2])
    # axs[2].set_title('Hippocampi & Amygdala')
    # axs[2].set_ylabel('Age Shifts Differences')
    # axs[2].set_xlabel('Disease Condition')

    # Plot third subplot
    sns.boxplot(x='disease_condition', y='age_shifts_diff_ventri', data=df, ax=axs[2])
    axs[2].set_title('Ventricles with diff map')
    axs[2].set_ylabel('Age Shifts Differences')
    axs[2].set_xlabel('Disease Condition')

    plt.savefig(f'./figs/{save_name}.png', bbox_inches='tight')
    plt.show()

def draw_1x3_plot_ADHC_tad(df):
    plt.rcParams.update({'font.size': 14})
    # Create a 2x2 figure
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))

    # Plot first subplot
    sns.boxplot(x='disease_condition', y='wholebrain_lst_tad', data=df, ax=axs[0])
    axs[0].set_title('Whole brain')
    axs[0].set_ylabel('Age Shifts Differences')
    axs[0].set_xlabel('Disease Condition')

    # Plot second subplot
    sns.boxplot(x='disease_condition', y='ventri_lst_tad', data=df, ax=axs[1])
    axs[1].set_title('Ventricles')
    axs[1].set_ylabel('Age Shifts Differences')
    axs[1].set_xlabel('Disease Condition')

    # Plot third subplot
    sns.boxplot(x='disease_condition', y='hippo_lst_tad', data=df, ax=axs[2])
    axs[2].set_title('Hippocampi & Amygdala')
    axs[2].set_ylabel('Age Shifts Differences')
    axs[2].set_xlabel('Disease Condition')

    # Plot third subplot
    sns.boxplot(x='disease_condition', y='diff_ventri_lst_tad', data=df, ax=axs[3])
    axs[3].set_title('Ventricles with diff map')
    axs[3].set_ylabel('Age Shifts Differences')
    axs[3].set_xlabel('Disease Condition')

    plt.savefig(f'./figs/ADHC_markers_noadjust_tad.png', bbox_inches='tight')
    plt.show()

def draw_1x3_plot_AD(df, save_name = 'AD_markers_updated'):
    plt.rcParams.update({'font.size': 14})
    # Create a 2x2 figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

    # Plot first subplot
    sns.boxplot(x='scan_cdr', y='age_shifts_wholebrain', data=df, ax=axs[0])
    axs[0].set_title('Whole brain')
    axs[0].set_ylabel('Age Shifts Differences')
    axs[0].set_xlabel('CDR')

    # Plot second subplot
    sns.boxplot(x='scan_cdr', y='age_shifts_ventri', data=df, ax=axs[1])
    axs[1].set_title('Ventricles')
    axs[1].set_ylabel('Age Shifts Differences')
    axs[1].set_xlabel('CDR')

    # # Plot third subplot
    # sns.boxplot(x='scan_cdr', y='age_shifts_hippo', data=df, ax=axs[2])
    # axs[2].set_title('Hippocampi & Amygdala')
    # axs[2].set_ylabel('Age Shifts Differences')
    # axs[2].set_xlabel('CDR')

    # Plot third subplot
    sns.boxplot(x='scan_cdr', y='age_shifts_diff_ventri', data=df, ax=axs[2])
    axs[2].set_title('Ventricles with diff map')
    axs[2].set_ylabel('Age Shifts Differences')
    axs[2].set_xlabel('CDR')

    plt.savefig(f'./figs/{save_name}.png', bbox_inches='tight')
    plt.show()

def draw_1x3_plot_AD_tad(df):
    plt.rcParams.update({'font.size': 14})
    # Create a 2x2 figure
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(18, 4))

    # Plot first subplot
    sns.boxplot(x='scan_cdr', y='wholebrain_lst_tad', data=df, ax=axs[0])
    axs[0].set_title('Whole brain')
    axs[0].set_ylabel('Age Shifts Differences')
    axs[0].set_xlabel('CDR')

    # Plot second subplot
    sns.boxplot(x='scan_cdr', y='ventri_lst_tad', data=df, ax=axs[1])
    axs[1].set_title('Ventricles')
    axs[1].set_ylabel('Age Shifts Differences')
    axs[1].set_xlabel('CDR')

    # Plot third subplot
    sns.boxplot(x='scan_cdr', y='hippo_lst_tad', data=df, ax=axs[2])
    axs[2].set_title('Hippocampi & Amygdala')
    axs[2].set_ylabel('Age Shifts Differences')
    axs[2].set_xlabel('CDR')

    # Plot third subplot
    sns.boxplot(x='scan_cdr', y='diff_ventri_lst_tad', data=df, ax=axs[3])
    axs[3].set_title('Ventricle diff')
    axs[3].set_ylabel('Age Shifts Differences')
    axs[3].set_xlabel('CDR')

    plt.savefig(f'./figs/AD_markers_noadjust_tad.png', bbox_inches='tight')
    plt.show()

def calculate_tk_wAD(Sub_svf, SubHC_svf, quantile = 0):
    u0 = SubHC_svf
    wk = Sub_svf
    x, y, z, d = u0.shape

    # calculate tk_3d
    u0_norm = np.sum(u0 ** 2, axis=3, keepdims=True)

    if quantile != 0:
        # set up a threshold
        mask = np.ones_like(crop_image(np.transpose(vt.load_nii(seg_90), (2, 1, 0)))[::2, ::2, ::2], dtype=np.uint8)
        mask[crop_image(np.transpose(vt.load_nii(seg_90), (2, 1, 0)))[::2, ::2, ::2] == 0] = 0
        print(f"# mask = {mask.sum()}")
        # quantile = 1- mask.sum()/(x*y*z*d)
        # th = np.quantile(np.unique(u0_norm), quantile) #  quantile
        th = np.quantile(u0_norm, quantile)
        print(f'{quantile} quantile for u0_norm is {th}')
    else:
        th = 0
    numerator = np.sum(wk * u0, axis = 3, keepdims=True)
    tk_3d = np.divide(numerator, u0_norm, out=np.zeros_like(numerator), where=u0_norm > th)
    # tk_3d = numerator / u0_norm

    # calculate w_ad
    tk_3d_broadcasted = np.broadcast_to(tk_3d, (x, y, z, d))
    w_ad = wk - tk_3d_broadcasted * u0

    return tk_3d.squeeze(), w_ad, th

def calculate_tk_given_mask(mask, tk_3d, w_ad, age_gap = 1):
    # calculate masked tk_3d and w_ad
    def apply_mask(input_array, mask):
        # create a new array with the same shape as the input array
        masked_array = np.zeros_like(input_array)
        # set the masked pixels to the input values
        masked_array[mask == 1] = input_array[mask == 1]
        return masked_array

    masked_tk_3d = apply_mask(tk_3d, mask)
    masked_w_ad = apply_mask(w_ad, mask)

    t_hc = np.sum(masked_tk_3d) / mask.sum()
    print(f't_hc={np.sum(masked_tk_3d) / mask.sum()}')
    U, V, W = masked_w_ad[..., 0], masked_w_ad[..., 1], masked_w_ad[..., 2]
    t_ad = (np.sum(np.sqrt(U ** 2 + V ** 2 + W ** 2)) / mask.sum()) / age_gap
    print(f't_ad={t_ad}')
    return t_hc, t_ad

def calculate_tk_given_labels(labels, segmentation, tk_3d, w_ad, age_gap = 1):
    mask = np.zeros_like(segmentation, dtype=np.uint8)
    if labels == "wholebrain":
        mask = np.ones_like(segmentation, dtype=np.uint8)
        mask[segmentation == 0] = 0
        print(f"# mask = {mask.sum()}")
    else:
        for label in labels:
            mask[segmentation == label] = 1
        print(f"# mask = {mask.sum()}")

    # calculate masked tk_3d and w_ad
    def apply_mask(input_array, mask):
        # create a new array with the same shape as the input array
        masked_array = np.zeros_like(input_array)
        # set the masked pixels to the input values
        masked_array[mask == 1] = input_array[mask == 1]
        return masked_array

    masked_tk_3d = apply_mask(tk_3d, mask)
    masked_w_ad = apply_mask(w_ad, mask)

    t_hc = np.sum(masked_tk_3d) / mask.sum()
    print(f't_hc={np.sum(masked_tk_3d) / mask.sum()}')
    U, V, W = masked_w_ad[..., 0], masked_w_ad[..., 1], masked_w_ad[..., 2]
    t_ad = (np.sum(np.sqrt(U ** 2 + V ** 2 + W ** 2)) / mask.sum()) / age_gap
    print(f't_ad={t_ad}')
    return t_hc, t_ad

def read_csv_set(meta_csv_path, disease=1, set = "test"):
    df = pd.read_csv(meta_csv_path)
    ids_set = df["ID"][df["Partition"]==set]
    disease_conditions = df["HC0/AD1"][df["Partition"] == set]
    ids_set = np.unique([i.split("_")[0] for c, i in enumerate(list(ids_set)) if list(disease_conditions)[c]==disease])
    return list(ids_set)

def read_csv(meta_csv_path, ID_name = 'OAS30001'):
    df = pd.read_csv(meta_csv_path)
    ids_set = [fpath+i.split("_")[0]+"_"+i.split("_")[2]+'.npz' for i in list(df["ID"])]
    names = np.array([i.split("_")[0] for i in list(df["ID"])])
    idxs  = np.where(names == ID_name)
    ids_set = np.array(ids_set)[idxs]
    disease_conditions = np.array(df['HC0/AD1'])[idxs]
    return ids_set, disease_conditions

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

def crop_image(arr):
    arr = np.transpose(arr, (2,1,0))

    # Get the center of the image along each axis
    center_x = arr.shape[0] // 2
    center_y = arr.shape[1] // 2
    center_z = arr.shape[2] // 2

    # Calculate the starting and ending indices for the crop along each axis
    start_x = center_x - 80
    end_x = center_x + 80
    start_y = center_y - 80
    end_y = center_y + 80
    start_z = center_z - 96
    end_z = center_z + 96

    # Crop the image
    cropped = arr[start_x:end_x, start_y:end_y, start_z:end_z]

    return cropped

def apply_mask(input_array, labels, segmentation):
    if labels == "wholebrain":
        mask = np.ones_like(segmentation, dtype=np.uint8)
        mask[segmentation == 0] = 0
        print(f"# mask = {mask.sum()}")
    else:
        mask = np.zeros_like(segmentation, dtype=np.uint8)
        for label in labels:
            mask[segmentation == label] = 1
        print(f"# mask = {mask.sum()}")
    num_mask = mask.sum()
    # create a new array with the same shape as the input array
    masked_array = np.zeros_like(input_array)
    # set the masked pixels to the input values
    masked_array[mask == 1] = input_array[mask == 1]
    return masked_array, num_mask, mask

def apply_mask_diff(input_array, labels, segmentation1, segmentation2):
    '''

    Args:
        input_array:
        labels:
        segmentation1: morphological smaller segmentation
        segmentation2: morphological bigger segmentation

    Returns:

    '''
    def get_mask_from_labels(segmentation):
        mask = np.zeros_like(segmentation, dtype=np.uint8)
        for label in labels:
            mask[segmentation == label] = 1
        print(f"# mask = {mask.sum()}")
        return mask
    if get_mask_from_labels(segmentation2).sum() > get_mask_from_labels(segmentation1).sum():
        dif_mask = get_mask_from_labels(segmentation2) - get_mask_from_labels(segmentation1)
    else:
        dif_mask = get_mask_from_labels(segmentation1) - get_mask_from_labels(segmentation2)

    # create a new array with the same shape as the input array
    masked_array = np.zeros_like(input_array)
    # set the masked pixels to the input values
    masked_array[dif_mask == 1] = input_array[dif_mask == 1]
    return masked_array, dif_mask

def plot_lines(list1, list2, title1='', title2=''):
    # create figure and axes objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # plot first line on the first subplot
    ax1.plot(list1)

    # set title and labels for the first subplot
    ax1.set_title(title1)
    ax1.set_xlabel('X axis')
    ax1.set_ylabel('Y axis')

    # plot second line on the second subplot
    ax2.plot(list2)

    # set title and labels for the second subplot
    ax2.set_title(title2)
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')

    # display the plot
    plt.show()

def show_thc_tad_on_region(labels_ventricles, svf_u0, saved_name, quantile = 0):
    t_hc_lst = []
    t_ad_lst = []
    for hc_svf_id in range(1, 31):
        # print(f'For velocity field vel{hc_svf_id}')
        hc_svf = np.load(f'{saved_svf_path}vel{hc_svf_id}.npz')['vel_half'].squeeze()
        # tk_3d, w_ad, th = calculate_tk_wAD(hc_svf, svf_u0.squeeze(), quantile = quantile)
        # t_hc, t_ad = calculate_tk_given_labels(labels_ventricles,
        #              np.transpose(vt.load_nii(seg_90), (2,1,0))[::2,::2,::2],
        #                                        tk_3d, w_ad, th)
        # refined t_hc and t_ad

        if labels_ventricles == 'diff_ventricles':
            t_hc, t_ad = calcu_biomarkers(labels_ventricles,
                                          [np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2, ::2],
                                           np.transpose(vt.load_nii(seg_90), (2, 1, 0))[::2, ::2, ::2]],
                                          hc_svf, svf_u0.squeeze(), age_gap=hc_svf_id, quantile=quantile)
        else:
            t_hc, t_ad = calcu_biomarkers(labels_ventricles,
                                          np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2, ::2],
                                          hc_svf, svf_u0.squeeze(), age_gap=hc_svf_id, quantile=quantile)

        t_ad_lst.append(t_ad)
        t_hc_lst.append(t_hc)

    def linear_regression(x, y):
        """Calculate the slope and intercept of a linear regression for a set of data."""
        # Calculate the slope and intercept using numpy's polyfit method
        slope, intercept = np.polyfit(x, y, 1)
        return slope, intercept

    # plot fitted line and points
    plt.rcParams.update({'font.size': 14})
    x = [i for i in range(1, len(t_hc_lst) + 1)]
    plot_fitted_line(np.array(x), np.array(t_hc_lst), saved_name)
    # plot_fitted_line(np.array(x), np.array(t_ad_lst), saved_name)

    # # plot_lines(t_hc_lst, t_ad_lst, 't_hc', 't_ad')
    # x = [i for i in range(1, len(t_hc_lst)+1)]
    # print(linear_regression(x, t_hc_lst))
    # plt.figure()
    # plt.plot(x, x)
    # plt.plot(x, t_hc_lst)
    # plt.show()

def plot_fitted_line(x, y, title, saved_name, ylims):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    plt.rcParams.update({'font.size': 14})
    # Create a linear regression object and fit the data
    reg = LinearRegression().fit(x.reshape(-1, 1), y)

    # Get the slope and intercept values
    slope = reg.coef_[0]
    intercept = reg.intercept_

    # Get the predicted y values for the data points
    y_pred = reg.predict(x.reshape(-1, 1))

    # Calculate the R-squared value
    r2 = r2_score(y, y_pred)

    # Plot the data points and the linear line
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(x, y_pred, color='red')

    # Add the legend with the slope, intercept, and R-squared values
    legend_text = f'y = {slope:.2f}x+{intercept:.2f}, R2: {r2:.2f}'
    ax.legend([legend_text])
    plt.ylim(ylims[0], ylims[1])
    plt.xlabel('Biological age')
    plt.ylabel('Marker value')
    plt.savefig(f'./figs/{saved_name}.png', bbox_inches='tight')
    plt.title(title)
    # Show the plot
    plt.show()

def pairwise_ttest(distributions, names):
    # Perform pairwise t-tests and calculate p-values
    p_values = {}
    for (i, dist1), (j, dist2) in itertools.combinations(enumerate(distributions), 2):
        dist1 = np.array(dist1)
        dist2 = np.array(dist2)
        dist1 = [i for i in dist1 if np.isnan(i) == False]
        dist2 = [i for i in dist2 if np.isnan(i) == False]
        t, p = ttest_ind(dist1, dist2, equal_var=False)
        p_values[(names[i], names[j])] = p

    # Print results
    for (name1, name2), p in p_values.items():
        if p < 0.05:
            print(f"{name1} and {name2} have a significant difference (p-value = {p})")
        else:
            print(f"{name1} and {name2} do not have a significant difference (p-value = {p})")

# load templates from best model (using HC_only model)
checkpoint_path = '/home/data/jrfu/data/trained_models/HC_only/training_checkpoints/gploss_1e_4_dataset_OASIS3_single_cohort_eps300_Gconfig_ours_normreg_True_lrg0.0001_lrd0.0003_cond_True_regloss_NCC_lbdgan_0.1_lbdreg_1.0_lbdtv_0.0_lbdgp_0.0001_dsnout_False_start_0_clip_True/'
templates_path = '/home/data/jrfu/data/trained_models/HC_only/plots/my_plot_1e-4/'
main_path  = '/home/data/jrfu/data/OASIS3/'
fpath      = f'{main_path}all_npz/'
# saved_svf_path = '/home/data/jrfu/data/trained_models/HC_only/plots/my_plot_1e-4_svfs/'
temp_60 = f'{templates_path}age_60disease_0.nii.gz'
seg_60 = f'{templates_path}age_60disease_0_SynthSeg.nii.gz'
seg_90 = f'{templates_path}age_90disease_0_SynthSeg.nii.gz'
meta_csv_path = f'{main_path}FS_scans_metadata_with_split.csv'

labels_ventricles = [4, 14, 15, 43]
labels_hippocampus = [17, 53, 18, 54]

# ================= 3. calculate tk and w_specific on Ventricle ================================================
saved_svf_path = '/home/data/jrfu/data/trained_models/HC_only/plots/my_plot_1e-4_svfs_invert/' # AtlasGAN is used for express average svf

# # ---- 3.1 test for HC templates firstly ---
# # --- test difference unit svf ------
svfs = []
for svf_id in range(30,31):
    ith_vel = np.load(f'{saved_svf_path}vel{svf_id}.npz')['vel_half'].squeeze()
    svfs.append(ith_vel/svf_id)

svf_u0 = svfs[0]

# ====== 3.2 test for HC test scans =============
seg_90_img = np.transpose(vt.load_nii(seg_90), (2, 1, 0))
seg_60_img = np.transpose(vt.load_nii(seg_60), (2, 1, 0))

AD_subjects_test = read_csv_set(meta_csv_path, disease=1, set="test")
HC_subjects_test = read_csv_set(meta_csv_path, disease=0, set="test")

# dev
AD_subjects_test.extend(read_csv_set(meta_csv_path, disease=1, set="dev"))
HC_subjects_test.extend(read_csv_set(meta_csv_path, disease=0, set="dev"))

# train
AD_subjects_test.extend(read_csv_set(meta_csv_path, disease=1, set="train"))

AD_subjects_test = np.unique(AD_subjects_test)
HC_subjects_test = np.unique(HC_subjects_test)

cdr_csv_path = f'{main_path}OASIS3_UDSb4_cdr.csv'
def load_cdr_forSubject(cdr_csv_path, subject_id, age):
    df = pd.read_csv(cdr_csv_path)
    sub = {}
    for head in df.head():
        sub[head] = df[head][df["OASISID"]==subject_id]
    index = np.argmin(np.abs(sub["age at visit"] - age))
    output=np.array(sub["CDRTOT"])[index]
    return output

def caculate_csv_given_cohort(HC_subjects_test, cohort = 'HC'):
    saved_HC_path = f'{saved_svf_path}{cohort}/'

    quantiles = [0, 0.5, 0.8, 0.99]

    # create a dict saving tk and tad
    dict_2t = {}
    subject_ids = []
    subject_ages = []

    age_shifts_ventri = []
    age_shifts_hippo = []
    age_shifts_wholebrain = []
    age_shifts_diff_hippo = []

    age_shifts_diff_ventri = []

    disease_condition = []
    scan_cdr = []

    # --- for HC subjects ---
    numb_HC_subj = len(HC_subjects_test)
    for count, subject_id in enumerate(HC_subjects_test):
        img_npzs, disease_conditions = read_csv(meta_csv_path, subject_id)
        imgs_names = [i.split("/")[7] for i in list(img_npzs)]

        imgs = [vt.load_npz(img_path)['vol'] for img_path in img_npzs]
        segs = [vt.load_npz(img_path)['synth_seg'] for img_path in img_npzs]
        ages = [vt.load_npz(img_path)['age'] for img_path in img_npzs]

        print(f'==== {count}/{numb_HC_subj}: HC Subject {subject_id} with {len(imgs)} scans ====')
        cdrs = [load_cdr_forSubject(cdr_csv_path, subject_id, ages[i]) for i in range(len(ages))]

        # # SAVE npz to a nifity
        # HC_subjects_test = ['OAS30614']
        # i = 2
        # img = imgs[i]
        # seg = segs[i]
        #
        # def save_to_nifti(data, save_path_name):
        #     vxm.py.utils.save_volfile(data, f'{save_path_name}.nii.gz')
        #     vt.correct_vox2ras_matrix(f'{save_path_name}.nii.gz', reference_nifiti='./src/align_norm.nii.gz')
        #
        # save_path = './temp_files/'
        # save_to_nifti(seg, f'{save_path}{subject_id}_seg_{ages[i]:.1f}')
        # save_to_nifti(img, f'{save_path}{subject_id}_img_{ages[i]:.1f}')

        # == registration pairwise ==
        for num_img, (img, seg) in enumerate(zip(imgs, segs)):
            print(f"Age is {ages[num_img]}")
            age_gap = ages[num_img] - 60
            if abs(age_gap) >= 0:
                saved_name = f'{saved_HC_path}{subject_id}_{ages[num_img]:.1f}.npz'
                load_npz = np.load(saved_name)
                vel_half = load_npz['vel_half']

                t_hc0 = []
                t_hc  = []
                t_hc1 = []
                t_hc2 = []
                t_hc3 = []
                for quantile in quantiles:
                    t_hc0.append(calcu_biomarkers("wholebrain", seg_60_img[::2, ::2, ::2], vel_half.squeeze(),
                                                    svf_u0.squeeze(), abs(age_gap), quantile))
                    t_hc.append(calcu_biomarkers(labels_ventricles, seg_60_img[::2, ::2, ::2], vel_half.squeeze(),
                                                  svf_u0.squeeze(), abs(age_gap), quantile))
                    t_hc1.append(calcu_biomarkers(labels_hippocampus, seg_60_img[::2, ::2, ::2], vel_half.squeeze(),
                                                    svf_u0.squeeze(), abs(age_gap), quantile))

                    t_hc2.append(calcu_biomarkers('diff_ventricles',
                                                    [seg_60_img[::2, ::2, ::2],
                                                     seg_90_img[::2, ::2, ::2]],  # self seg
                                                    vel_half.squeeze(), svf_u0.squeeze(), abs(age_gap), quantile))

                    t_hc3.append(calcu_biomarkers('diff_hippo',
                                                    [seg_90_img[::2, ::2, ::2],
                                                     seg_60_img[::2, ::2, ::2]],  # self seg
                                                    vel_half.squeeze(), svf_u0.squeeze(), abs(age_gap), quantile))

                # save data to lsts
                subject_ids.append(subject_id)
                subject_ages.append(ages[num_img])
                disease_condition.append(0)
                age_shifts_ventri.append(t_hc)
                age_shifts_hippo.append(t_hc1)
                age_shifts_wholebrain.append(t_hc0)

                age_shifts_diff_ventri.append(t_hc2)
                age_shifts_diff_hippo.append(t_hc3)
                scan_cdr.append(load_cdr_forSubject(cdr_csv_path, subject_id, ages[num_img]))

    dict_2t = {}
    dict_2t["subject_ids"]=subject_ids
    dict_2t["subject_ages"]=subject_ages
    dict_2t["disease_condition"]=disease_condition
    dict_2t["scan_cdr"]=scan_cdr

    def separate_thc_tad(lsts):
        thc_all = []
        tad_all = []
        for lst in lsts:
            thc_quantile = []
            tad_quantile = []
            for thcad in lst:
                thc_quantile.append(thcad[0])
                tad_quantile.append(thcad[1])
            thc_all.append(thc_quantile)
            tad_all.append(tad_quantile)
        return thc_all, tad_all

    ven_thc, ven_tad = separate_thc_tad(age_shifts_ventri)
    hipp_thc, hipp_tad = separate_thc_tad(age_shifts_hippo)
    wb_thc, wb_tad = separate_thc_tad(age_shifts_wholebrain)
    vendiff_thc, vendiff_tad = separate_thc_tad(age_shifts_diff_ventri)
    hippdiff_thc, hippdiff_tad = separate_thc_tad(age_shifts_diff_hippo)

    dict_2t["ven_thc"]=ven_thc
    dict_2t["ven_tad"]=ven_tad
    dict_2t["hipp_thc"]=hipp_thc
    dict_2t["hipp_tad"]=hipp_tad
    dict_2t["wb_thc"]=wb_thc
    dict_2t["wb_tad"]=wb_tad

    dict_2t["vendiff_thc"]=vendiff_thc
    dict_2t["vendiff_tad"]=vendiff_tad
    dict_2t["hippdiff_thc"]=hippdiff_thc
    dict_2t["hippdiff_tad"]=hippdiff_tad

    csv_name = f'./figs/Vel30_invert_{cohort}_age_shifts1.csv'
    print('Saving to {} file'.format(csv_name))
    pd.DataFrame(dict_2t).to_csv(csv_name, index=False)

caculate_csv_given_cohort(HC_subjects_test)
# caculate_csv_given_cohort(AD_subjects_test, cohort = 'AD')

# ============ plot =====================
# Load CSV data into Pandas DataFrame
csv_name = f'./figs/Vel30_invert_HC_age_shifts1.csv'
quantiles = [0, 0.5, 0.8, 0.99]
print(csv_name)

regions = ['Ventricles', 'Hippo', 'WholeBrain', 'Ventricles_diff'] # 'Hippo_diff'
df = pd.read_csv(csv_name)

def create_new_df(df, quantiles = [0, 0.5, 0.8, 0.99], cohort = 'AD'):
    df_new = pd.DataFrame()
    for key, value in df.items():
        if key in ["subject_ids", "subject_ages", "disease_condition", "scan_cdr"]:
            df_new[key] = value
        else:
            for id, quantile in enumerate(quantiles):
                df_new[f'{key}_{quantile}'] = [float(i.split(',')[id].replace('[', '').replace(']', '')) for i in list(value)]
    # new_csv_name = f'./figs/Vel30_invert_{cohort}_age_shifts_seperate.csv'
    # df_new.to_csv(new_csv_name, index=False)
    return df_new

def df_plot_AD(df_new):
    for key, value in df_new.items():
        if key in ["subject_ids", "subject_ages", "disease_condition", "scan_cdr"]:
            pass
        else:
            sns.lmplot(x="subject_ages", y=key, hue="scan_cdr", data=df_new,
                       palette="Set1");
            plt.show()

def df_plot_HC(df_new):
    for key, value in df_new.items():
        if key in ["subject_ids", "subject_ages", "disease_condition", "scan_cdr"]:
            pass
        else:
            sns.jointplot(x="subject_ages", y=key, data=df_new, kind="reg");
            plt.show()

def plot_1x4_fitted_line(df, th, save_name):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    plt.rcParams.update({'font.size': 14})

    # Plot the data points and the linear line
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(25, 4))

    # Plot first subplot
    sns.regplot(x='subject_ages', y=f'wb_thc_{th}', data=df, ax=axs[0],
                scatter_kws={"color": "black", "alpha": 0.5},
                line_kws={"color": "red"},
                ci=99)
    axs[0].set_title('Whole brain')
    axs[0].set_ylabel('Marker Value')
    axs[0].set_xlabel('Biological Age')
    axs[0].set_ylim([-40, 50])

    # Plot second subplot
    sns.regplot(x='subject_ages', y=f'ven_thc_{th}', data=df, ax=axs[1],
                scatter_kws={"color": "black", "alpha": 0.5},
                line_kws={"color": "red"},
                ci=99)
    axs[1].set_title('Ventricles')
    axs[1].set_ylabel('Marker Value')
    axs[1].set_xlabel('Biological Age')
    axs[1].set_ylim([-40, 50])

    # Plot third subplot
    sns.regplot(x='subject_ages', y=f'hipp_thc_{th}', data=df, ax=axs[2],
                scatter_kws={"color": "black", "alpha": 0.5},
                line_kws={"color": "red"},
                ci=99)
    axs[2].set_title('Hippocampi & Amygdala')
    axs[2].set_ylabel('Marker Value')
    axs[2].set_xlabel('Biological Age')
    axs[2].set_ylim([-40, 50])

    # Plot third subplot
    sns.regplot(x='subject_ages', y=f'vendiff_thc_{th}', data=df, ax=axs[3],
                scatter_kws={"color": "black", "alpha": 0.5},
                line_kws={"color": "red"},
                ci=99)
    axs[3].set_title('Ventricle diff')
    axs[3].set_ylabel('Marker Value')
    axs[3].set_xlabel('Biological Age')
    axs[3].set_ylim([-40, 50])

    plt.savefig(f'./figs/{save_name}.png', bbox_inches='tight')
    plt.show()

# df_new = create_new_df(df)
# # df_plot_AD(df_new)
#
# df_new = create_new_df(df, cohort = 'HC')
#
# print_names = ['wb_thc', 'ven_thc',
#                'hipp_thc', 'vendiff_thc']
# thresholds = ['0', '0.99']
#
# for thre in thresholds:
#     plot_1x4_fitted_line(df_new, thre, f'HC_marker_fittedline_at{thre}')

# for print_name in print_names:
#     for thre in thresholds:
#         plot_fitted_line(np.array(df_new.subject_ages), np.array(df_new[f'{print_name}_{thre}']), '', f'{print_name}_{thre}', [-40, 50])

# compare AD and HC
print_names = ['wb_thc', 'ven_thc',
               'hipp_thc', 'vendiff_thc']
thresholds = ['0', '0.99']

def draw_1x4_plot_ADHC(df, th, save_name = 'ADHC_markers_updated'):
    plt.rcParams.update({'font.size': 14})
    # Create a 2x2 figure
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(25, 4))

    # Plot first subplot
    sns.boxplot(x='disease_condition', y=f'wb_thc_{th}', data=df, ax=axs[0])
    axs[0].set_title('Whole brain')
    axs[0].set_ylabel('Marker Value')
    axs[0].set_xlabel('Disease condition')

    # Plot second subplot
    sns.boxplot(x='disease_condition', y=f'ven_thc_{th}', data=df, ax=axs[1])
    axs[1].set_title('Ventricles')
    axs[1].set_ylabel('Marker Value')
    axs[1].set_xlabel('Disease condition')

    # Plot third subplot
    sns.boxplot(x='disease_condition', y=f'hipp_thc_{th}', data=df, ax=axs[2])
    axs[2].set_title('Hippocampi & Amygdala')
    axs[2].set_ylabel('Marker Value')
    axs[2].set_xlabel('Disease condition')

    # Plot third subplot
    sns.boxplot(x='disease_condition', y=f'vendiff_thc_{th}', data=df, ax=axs[3])
    axs[3].set_title('Ventricle diff')
    axs[3].set_ylabel('Marker Value')
    axs[3].set_xlabel('Disease condition')

    plt.savefig(f'./figs/{save_name}.png', bbox_inches='tight')
    plt.show()

AD_csv_name = f'./figs/Vel30_invert_AD_age_shifts.csv'
HC_csv_name = f'./figs/Vel30_invert_HC_age_shifts1.csv'

AD_df = pd.read_csv(AD_csv_name)
AD_df.disease_condition.values[:]=1
HC_df = pd.read_csv(HC_csv_name)

AD_df = create_new_df(AD_df)
HC_df = create_new_df(HC_df)

frames = [AD_df, HC_df]
result = pd.concat(frames)
result.to_csv('./figs/Vel30_invert_AllTest_results.csv', index=False)
for thre in thresholds:
    draw_1x4_plot_ADHC(result, thre, save_name = f'ADHC_markers_boxplots_at{thre}')


# draw cohort level boxplot
def draw_1x4_plot_AD_thc(df, th, save_name):
    plt.rcParams.update({'font.size': 14})
    # Create a 2x2 figure
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(25, 4))

    # Plot first subplot
    sns.boxplot(x='scan_cdr', y=f'wb_thc_{th}', data=df, ax=axs[0])
    axs[0].set_title('Whole brain')
    axs[0].set_ylabel('Marker Value')
    axs[0].set_xlabel('CDR')

    # Plot second subplot
    sns.boxplot(x='scan_cdr', y=f'ven_thc_{th}', data=df, ax=axs[1])
    axs[1].set_title('Ventricles')
    axs[1].set_ylabel('Marker Value')
    axs[1].set_xlabel('CDR')

    # Plot third subplot
    sns.boxplot(x='scan_cdr', y=f'hipp_thc_{th}', data=df, ax=axs[2])
    axs[2].set_title('Hippocampi & Amygdala')
    axs[2].set_ylabel('Marker Value')
    axs[2].set_xlabel('CDR')

    # Plot third subplot
    sns.boxplot(x='scan_cdr', y=f'vendiff_thc_{th}', data=df, ax=axs[3])
    axs[3].set_title('Ventricle diff')
    axs[3].set_ylabel('Marker Value')
    axs[3].set_xlabel('CDR')

    plt.savefig(f'./figs/{save_name}.png', bbox_inches='tight')
    plt.show()

csv_name = f'./figs/Vel30_invert_AD_age_shifts.csv'
quantiles = [0, 0.5, 0.8, 0.99]
print(csv_name)

regions = ['Ventricles', 'Hippo', 'WholeBrain', 'Ventricles_diff'] # 'Hippo_diff'
df = pd.read_csv(csv_name)

df_new = create_new_df(df)

for thre in thresholds:
    draw_1x4_plot_AD_thc(df_new, thre, f'AD_markers_boxplots_at{thre}')

# drop age diff < 5
df_new = df_new.drop(df_new[abs(df_new.subject_ages - 60) < 5].index)
for thre in thresholds:
    draw_1x4_plot_AD_thc(df_new, thre, f'AD_markers_boxplots_at{thre}_drop5')

# # ---- 3.2 test for AD then ---

# for i, svf_u0 in enumerate(svfs):
#     print(i)
#     for quantile in [0.87]: # 0, 0.5, 0.8, 0.9
#        show_thc_tad_on_region('wholebrain', svf_u0, 'wholebrain_healthy_markers', quantile=quantile)
#        # show_thc_tad_on_region(labels_ventricles, svf_u0, 'ventricles_healthy_markers', quantile=quantile)
#        show_thc_tad_on_region(labels_hippocampus, svf_u0, 'hippocampus_healthy_markers', quantile=quantile)
#        # show_thc_tad_on_region('diff_ventricles', svf_u0, 'diff_ventricles_healthy_markers', quantile=quantile)

# COMMONT: TODO: For whole brain and ventricles: 0.995 quantiles give the best fit (1.06x-0.31; 1.07x+0.76);
#  for hipp: 0.87 gives the best fit: 0.99x+3.17
# # plot optimal figures
# show_thc_tad_on_region('wholebrain', svfs[0], 'wholebrain_healthy_markers_optimal', quantile=0.995)
# show_thc_tad_on_region(labels_ventricles, svfs[0], 'ventricles_healthy_markers_optimal', quantile=0.995)
# show_thc_tad_on_region('diff_ventricles', svfs[0], 'diff_ventricles_healthy_markers_optimal', quantile=0.995)


# ---- 3.2 calculate tk tad for each cohort and save them to a csv file ----
cdr_csv_path = f'{main_path}OASIS3_UDSb4_cdr.csv'
def load_cdr_forSubject(cdr_csv_path, subject_id, age):
    df = pd.read_csv(cdr_csv_path)
    sub = {}
    for head in df.head():
        sub[head] = df[head][df["OASISID"]==subject_id]
    index = np.argmin(np.abs(sub["age at visit"] - age))
    output=np.array(sub["CDRTOT"])[index]
    return output

saved_svf_path = '/home/data/jrfu/data/trained_models/HC_only/plots/my_plot_1e-4_svfs_invert/'\
# '/home/data/jrfu/data/trained_models/HC_only/plots/my_plot_1e-4_svfs_synthmorph_invert/' # Synthmorph is used for calculating tk?
# or '/home/data/jrfu/data/trained_models/HC_only/plots/my_plot_1e-4_svfs_invert/'
Flag_synthmorph = False

def construct_csv(saved_svf_path, svf_u0, csv_name, saved = False):
    seg_90_img = np.transpose(vt.load_nii(seg_90), (2, 1, 0))
    seg_60_img = np.transpose(vt.load_nii(seg_60), (2, 1, 0))

    saved_AD_path = f'{saved_svf_path}AD/'
    saved_HC_path = f'{saved_svf_path}HC/'

    AD_subjects_test = read_csv_set(meta_csv_path, disease=1, set = "test")
    HC_subjects_test = read_csv_set(meta_csv_path, disease=0, set = "test")

    # dev
    AD_subjects_test.extend(read_csv_set(meta_csv_path, disease=1, set="dev"))
    HC_subjects_test.extend(read_csv_set(meta_csv_path, disease=0, set="dev"))

    # train
    AD_subjects_test.extend(read_csv_set(meta_csv_path, disease=1, set="train"))

    AD_subjects_test = np.unique(AD_subjects_test)
    HC_subjects_test = np.unique(HC_subjects_test)

    # substract HC 0 avg, so calculate HC 0 avg now
    def avg_SVF():
        vel_halfs = []
        for count, subject_id in enumerate(HC_subjects_test):
            img_npzs, disease_conditions = read_csv(meta_csv_path, subject_id)

            imgs = [vt.load_npz(img_path)['vol'] for img_path in img_npzs]
            segs = [vt.load_npz(img_path)['synth_seg'] for img_path in img_npzs]
            ages = [vt.load_npz(img_path)['age'] for img_path in img_npzs]

            print(f'==== {count}/{len(HC_subjects_test)}: AD Subject {subject_id} with {len(imgs)} scans ====')

            for num_img, (img, seg) in enumerate(zip(imgs, segs)):
                saved_name = f'{saved_HC_path}{subject_id}_{ages[num_img]:.1f}.npz'
                load_npz = np.load(saved_name)
                vel_halfs.append(load_npz['vel_half'])
        return np.mean(np.array(vel_halfs).squeeze(), 0)
    HC0_avg_svf = svf_u0 # np.load(f'{saved_svf_path}HC/HC0_avg_svf.npz')['vel_half']

    # AD_subjects_test = ['OAS30027']
    if Flag_synthmorph:
        seg_60_img = crop_image(seg_60_img)
        seg_90_img = crop_image(seg_90_img)
        svf_u0 = crop_image_vector(svf_u0)

    if saved == False:
        # ---- save ----
        # create a dict saving tk and tad
        dict_2t = {}
        subject_ids = []
        subject_ages = []
        age_shifts_ventri = []
        ventri_lst_tad = []
        age_shifts_hippo = []
        hippo_lst_tad = []
        age_shifts_wholebrain = []
        wholebrain_lst_tad = []

        age_shifts_diff_ventri = []
        diff_ventri_lst_tad = []

        disease_condition = []
        scan_cdr = []

        # --- for AD subjects ---
        numb_AD_subj = len(AD_subjects_test)
        for count, subject_id in enumerate(AD_subjects_test):
            img_npzs, disease_conditions = read_csv(meta_csv_path, subject_id)

            imgs = [vt.load_npz(img_path)['vol'] for img_path in img_npzs]
            segs = [vt.load_npz(img_path)['synth_seg'] for img_path in img_npzs]
            ages = [vt.load_npz(img_path)['age'] for img_path in img_npzs]
            cdrs = [load_cdr_forSubject(cdr_csv_path, subject_id, ages[i]) for i in range(len(ages))]

            print(f'==== {count}/{numb_AD_subj}: AD Subject {subject_id} with {len(imgs)} scans ====')

            for num_img, (img, seg) in enumerate(zip(imgs, segs)):
                print(f"Age is {ages[num_img]}")
                age_gap = ages[num_img] - 60
                if  abs(age_gap) >= 0: # do it for all scans if using averaged u0
                    saved_name = f'{saved_AD_path}{subject_id}_{ages[num_img]:.1f}.npz'
                    load_npz = np.load(saved_name)
                    vel_half = load_npz['vel_half']#.squeeze() - HC0_avg_svf
                    # tk_3d, w_ad = calculate_tk_wAD(vel_half.squeeze(), svf_u0.squeeze())
                    # TODO: think about how to decide seg
                    # t_hc0, t_ad0 = calculate_tk_given_labels("wholebrain", seg_60_img[::2, ::2, ::2], tk_3d, w_ad, age_gap)
                    # t_hc, t_ad = calculate_tk_given_labels(labels_ventricles, seg_60_img[::2, ::2, ::2], tk_3d, w_ad, age_gap)
                    # t_hc1, t_ad1 = calculate_tk_given_labels(labels_hippocampus, seg_60_img[::2, ::2, ::2], tk_3d, w_ad, age_gap)
                    t_hc0, t_ad0 = calcu_biomarkers("wholebrain", seg_60_img[::2, ::2, ::2], vel_half.squeeze(), svf_u0.squeeze(), abs(age_gap), quantile=0.995)
                    t_hc, t_ad = calcu_biomarkers(labels_ventricles, seg_60_img[::2, ::2, ::2], vel_half.squeeze(),
                                                    svf_u0.squeeze(), abs(age_gap), quantile=0.995)
                    t_hc1, t_ad1 = calcu_biomarkers(labels_hippocampus, seg_60_img[::2, ::2, ::2], vel_half.squeeze(),
                                                    svf_u0.squeeze(), abs(age_gap), quantile=0.99)

                    if Flag_synthmorph:
                        seg = crop_image(seg)
                    t_hc2, t_ad2 = calcu_biomarkers('diff_ventricles',
                                                  [seg_60_img[::2, ::2, ::2],
                                                   seg_90_img[::2, ::2, ::2]], # self seg or seg 90
                                                   vel_half.squeeze(), svf_u0.squeeze(), abs(age_gap), quantile=0.995)
                    print(
                        f'Age shift for wholebrain is {t_hc0 - ages[num_img] + 60}, ventricle part is {t_hc - ages[num_img] + 60}, hippo part is {t_hc1 - ages[num_img] + 60}')

                    # dt.show_vectors_in3dimemsions(apply_mask(vel_half.squeeze(), labels_ventricles,
                    #                                          np.transpose(vt.load_nii(seg_90), (2, 1, 0))[::2, ::2, ::2])[0], scale=300)
                    # dt.show_vectors_in3dimemsions(apply_mask(np.load(f'{saved_AD_path}{subject_id}_{ages[2]:.1f}.npz')['vel_half'].squeeze(), labels_ventricles,
                    # np.transpose(vt.load_nii(seg_90), (2, 1, 0))[::2, ::2, ::2])[0], scale=50)
                    # vt.TEST_Ori_3slice(img, title=f'Moving {ages[num_img]}')

                    # def calculate_marker_trend_for_subject(saved_AD_path1, seg_60_img, seg_90_img, seg, svf_u0,
                    #                                        labels='wholebrain'):
                    #     if "synthmorph" in saved_AD_path1:
                    #         seg_60_img = crop_image(seg_60_img)
                    #         seg_90_img = crop_image(seg_90_img)
                    #         svf_u0 = crop_image_vector(svf_u0)
                    #         seg = crop_image(seg)
                    #     for quantile in [0, 0.8, 0.9, 0.999]:
                    #         thc = []
                    #         tad = []
                    #         for i in range(len(imgs)):
                    #             t_hc0, t_ad0 = calcu_biomarkers(labels, seg_60_img[::2, ::2, ::2],
                    #                                             np.load(
                    #                                                 f'{saved_AD_path1}{subject_id}_{ages[i]:.1f}.npz')[
                    #                                                 'vel_half'].squeeze(), svf_u0.squeeze(), age_gap,
                    #                                             quantile=quantile)
                    #             thc.append(t_hc0)
                    #             tad.append(t_ad0)
                    #         plt.plot(ages, thc)
                    #         plt.title(quantile)
                    #         plt.show()

                    # # # plot svf_u0
                    # print(age_gap)
                    # dt.show_vectors_in3dimemsions(apply_mask(vel_half.squeeze() * age_gap, labels_ventricles,
                    #                                          np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2,
                    #                                          ::2])[0], scale=25)
                    # dt.show_vectors_in3dimemsions(apply_mask(vel_half.squeeze(), labels_ventricles,
                    #                                          np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2,
                    #                                          ::2])[0], scale=25)
                    # calculate_tk_given_labels(labels_ventricles, seg_60_img[::2, ::2, ::2], tk_3d, w_ad, age_gap)
                    #
                    # dt.show_vectors_in3dimemsions(apply_mask_diff(svf_u0.squeeze() * age_gap, labels_ventricles,
                    #                                               np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2,
                    #                                               ::2], np.transpose(vt.load_nii(seg_90), (2, 1, 0))[::2, ::2,
                    #                                               ::2])[0], scale=25)
                    # dt.show_vectors_in3dimemsions(apply_mask_diff(svf_u0.squeeze() * age_gap, labels_ventricles,
                    #                                               np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2,
                    #                                               ::2], seg[::2, ::2,::2])[0], scale=25)
                    # # plot subject-2-template
                    # dt.show_vectors_in3dimemsions(apply_mask_diff(vel_half.squeeze(), labels_ventricles,
                    #                                               np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2,
                    #                                               ::2], np.transpose(vt.load_nii(seg_90), (2, 1, 0))[::2, ::2,
                    #                                               ::2])[0], scale=25)
                    # dt.show_vectors_in3dimemsions(apply_mask_diff(vel_half.squeeze(), labels_ventricles,
                    #                                               np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2,
                    #                                               ::2], seg[::2, ::2,::2])[0], scale=25)
                    # # just report ventricles edages
                    # calculate_tk_given_mask(apply_mask_diff(svf_u0.squeeze() * age_gap, labels_ventricles,
                    #                                         np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2, ::2],
                    #                                         np.transpose(vt.load_nii(seg_90), (2, 1, 0))[::2, ::2, ::2])[1], tk_3d, w_ad, age_gap=1)
                    #
                    # calculate_tk_given_mask(apply_mask_diff(svf_u0.squeeze() * age_gap, labels_ventricles,
                    #                                         np.transpose(vt.load_nii(seg_60), (2, 1, 0))[::2, ::2, ::2],
                    #                                         seg[::2, ::2, ::2])[1], tk_3d, w_ad, age_gap=1)


                    # # --- calculate divergence of w_ad and save it ---
                    # W_vec = np.transpose(apply_mask(w_ad, "wholebrain", seg_60_img[::2, ::2, ::2])[0], (3, 0, 1, 2))
                    # x = np.array(np.meshgrid(np.arange(W_vec.shape[2]), np.arange(W_vec.shape[1]), np.arange(W_vec.shape[3])))
                    # W_div_man = hd.diverg(W_vec, x)
                    # W_curl_man = hd.curl_f(W_vec, x)[3]
                    # with np.load(saved_name) as existing_data:
                    #     np.savez_compressed(saved_name, **{**existing_data,
                    #                                        'W_div_man':W_div_man, 'W_curl_man':W_curl_man})

                    # save data to lsts
                    subject_ids.append(subject_id)
                    subject_ages.append(ages[num_img])
                    disease_condition.append(1)
                    age_shifts_ventri.append(t_hc - age_gap)
                    ventri_lst_tad.append(t_ad)
                    age_shifts_hippo.append(t_hc1 - age_gap)
                    hippo_lst_tad.append(t_ad1)
                    age_shifts_wholebrain.append(t_hc0 - age_gap)
                    wholebrain_lst_tad.append(t_ad0)

                    age_shifts_diff_ventri.append(t_hc2 - age_gap)
                    diff_ventri_lst_tad.append(t_ad2)

                    scan_cdr.append(load_cdr_forSubject(cdr_csv_path, subject_id, ages[num_img]))

        # --- for HC subjects ---
        numb_HC_subj = len(HC_subjects_test)
        for count, subject_id in enumerate(HC_subjects_test):
            img_npzs, disease_conditions = read_csv(meta_csv_path, subject_id)
            imgs_names = [i.split("/")[7] for i in list(img_npzs)]

            imgs = [vt.load_npz(img_path)['vol'] for img_path in img_npzs]
            segs = [vt.load_npz(img_path)['synth_seg'] for img_path in img_npzs]
            ages = [vt.load_npz(img_path)['age'] for img_path in img_npzs]
            cdrs = [load_cdr_forSubject(cdr_csv_path, subject_id, ages[i]) for i in range(len(ages))]

            print(f'==== {count}/{numb_HC_subj}: HC Subject {subject_id} with {len(imgs)} scans ====')

            # == registration pairwise ==
            for num_img, (img, seg) in enumerate(zip(imgs, segs)):
                print(f"Age is {ages[num_img]}")
                age_gap = ages[num_img] - 60
                if abs(age_gap) >= 0:
                    saved_name = f'{saved_HC_path}{subject_id}_{ages[num_img]:.1f}.npz'
                    load_npz = np.load(saved_name)
                    vel_half = load_npz['vel_half']#.squeeze() - HC0_avg_svf
                    # tk_3d, w_ad = calculate_tk_wAD(vel_half.squeeze(), svf_u0.squeeze())
                    # TODO: think about how to decide seg
                    # t_hc0, t_ad0 = calculate_tk_given_labels("wholebrain", seg_60_img[::2, ::2, ::2], tk_3d, w_ad, age_gap)
                    # t_hc, t_ad = calculate_tk_given_labels(labels_ventricles, seg_60_img[::2, ::2, ::2], tk_3d, w_ad, age_gap)
                    # t_hc1, t_ad1 = calculate_tk_given_labels(labels_hippocampus, seg_60_img[::2, ::2, ::2], tk_3d, w_ad, age_gap)
                    t_hc0, t_ad0 = calcu_biomarkers("wholebrain", seg_60_img[::2, ::2, ::2], vel_half.squeeze(), svf_u0.squeeze(), abs(age_gap), quantile=0.995)
                    t_hc, t_ad = calcu_biomarkers(labels_ventricles, seg_60_img[::2, ::2, ::2], vel_half.squeeze(),
                                                    svf_u0.squeeze(), abs(age_gap), quantile=0.995)
                    t_hc1, t_ad1 = calcu_biomarkers(labels_hippocampus, seg_60_img[::2, ::2, ::2], vel_half.squeeze(),
                                                    svf_u0.squeeze(), abs(age_gap), quantile=0.99)


                    if Flag_synthmorph:
                        seg = crop_image(seg)
                    t_hc2, t_ad2 = calcu_biomarkers('diff_ventricles',
                                                  [seg_60_img[::2, ::2, ::2],
                                                   seg_90_img[::2, ::2, ::2]], # self seg
                                                   vel_half.squeeze(), svf_u0.squeeze(), abs(age_gap), quantile=0.995)

                    # # --- calculate divergence of w_ad and save it ---
                    # W_vec = np.transpose(apply_mask(w_ad, "wholebrain", seg[::2, ::2, ::2])[0], (3, 0, 1, 2))
                    # x = np.array(np.meshgrid(np.arange(W_vec.shape[2]), np.arange(W_vec.shape[1]), np.arange(W_vec.shape[3])))
                    # W_div_man = hd.diverg(W_vec, x)
                    # W_curl_man = hd.curl_f(W_vec, x)[3]
                    # with np.load(saved_name) as existing_data:
                    #     np.savez_compressed(saved_name, **{**existing_data,
                    #                                        'W_div_man':W_div_man, 'W_curl_man':W_curl_man})

                    print(
                        f'Age shift for wholebrain is {t_hc0 - ages[num_img] + 60}, ventricle part is {t_hc - ages[num_img] + 60}, hippo part is {t_hc1 - ages[num_img] + 60}')

                    # save data to lsts
                    subject_ids.append(subject_id)
                    subject_ages.append(ages[num_img])
                    disease_condition.append(0)
                    age_shifts_ventri.append(t_hc-age_gap)
                    ventri_lst_tad.append(t_ad)
                    age_shifts_hippo.append(t_hc1-age_gap)
                    hippo_lst_tad.append(t_ad1)
                    age_shifts_wholebrain.append(t_hc0-age_gap)
                    wholebrain_lst_tad.append(t_ad0)

                    age_shifts_diff_ventri.append(t_hc2 - age_gap)
                    diff_ventri_lst_tad.append(t_ad2)

                    scan_cdr.append(load_cdr_forSubject(cdr_csv_path, subject_id, ages[num_img]))

        dict_2t["subject_ids"]=subject_ids
        dict_2t["subject_ages"]=subject_ages
        dict_2t["disease_condition"]=disease_condition
        dict_2t["age_shifts_ventri"]=age_shifts_ventri
        dict_2t["ventri_lst_tad"]=ventri_lst_tad
        dict_2t["age_shifts_hippo"]=age_shifts_hippo
        dict_2t["hippo_lst_tad"]=hippo_lst_tad
        dict_2t["age_shifts_wholebrain"]=age_shifts_wholebrain
        dict_2t["wholebrain_lst_tad"]=wholebrain_lst_tad

        dict_2t["age_shifts_diff_ventri"] = age_shifts_diff_ventri
        dict_2t["diff_ventri_lst_tad"] = diff_ventri_lst_tad

        dict_2t["scan_cdr"] = scan_cdr
        print('Saving to {} file'.format(csv_name))
        pd.DataFrame(dict_2t).to_csv(csv_name, index=False)

    # ============ plot =====================
    # Load CSV data into Pandas DataFrame
    print(csv_name)

    df = pd.read_csv(csv_name)

    # ==== show boxplots for different age gaps, on healthy subjects ====
    def boxplot_by_agerange(value = 'age_shifts_diff_ventri'):
        df = pd.read_csv(csv_name)
        df = df.drop(df[(df.disease_condition == 1)].index) # exclude AD cohort
        age_intervals = [(0, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
        age_labels = ['<50', '50-59', '60-69', '70-79', '80-89', '90-99']
        df['age_category'] = pd.cut(df['subject_ages'],
                                    bins=[interval[0] for interval in age_intervals] + [age_intervals[-1][1]],
                                    labels=age_labels)
        fig, ax = plt.subplots()
        # Use boxplot function to plot boxplots for each age category
        sns.boxplot(data=df, x='age_category', y=value, ax=ax)
        # Add title and axis labels
        ax.set_title("Boxplots by Age Category")
        ax.set_xlabel("Age Category")
        ax.set_ylabel(value)
        # Rotate x-axis labels if needed
        plt.xticks(rotation=45)
        # Display the plot
        plt.show()

    # draw for paper
    draw_1x3_plot_ADHC(df)
    draw_1x3_plot_ADHC_tad(df)
    for name in ["age_shifts_wholebrain", "age_shifts_ventri", "age_shifts_hippo", "age_shifts_diff_ventri"]:
        data = df[name][df['disease_condition'] == 0]
        print(statistics.mean(data), statistics.stdev(data))
        data = df[name][df['disease_condition'] == 1]
        print(statistics.mean(data), statistics.stdev(data))

    # # Create a 2x2 figure
    # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    #
    # # Plot first subplot
    # sns.boxplot(x='disease_condition', y='age_shifts_ventri', data=df, ax=axs[0, 0])
    # axs[0, 0].set_title('Age Shifts Ventri')
    #
    # # Plot second subplot
    # sns.boxplot(x='disease_condition', y='age_shifts_wholebrain', data=df, ax=axs[0, 1])
    # axs[0, 1].set_title('age_shifts_wholebrain')
    #
    # # Plot third subplot
    # sns.boxplot(x='disease_condition', y='age_shifts_hippo', data=df, ax=axs[1, 0])
    # axs[1, 0].set_title('Age Shifts Hippo')
    #
    # # Plot fourth subplot
    # sns.boxplot(x='disease_condition', y='wholebrain_lst_tad', data=df, ax=axs[1, 1])
    # axs[1, 1].set_title('wholebrain_lst_tad')
    #
    # # Show the figure
    # plt.suptitle(csv_name)
    # plt.show()

    # t-test
    pairwise_ttest(
        [df["age_shifts_ventri"][df["disease_condition"] == 0], df["age_shifts_ventri"][df["disease_condition"] == 1]],
        ["age_shifts_ventri_healthy", "age_shifts_ventri_AD"])
    pairwise_ttest(
        [df["age_shifts_hippo"][df["disease_condition"] == 0], df["age_shifts_hippo"][df["disease_condition"] == 1]],
        ["age_shifts_hippo_healthy", "age_shifts_hippo_AD"])
    pairwise_ttest(
        [df["age_shifts_wholebrain"][df["disease_condition"] == 0], df["age_shifts_wholebrain"][df["disease_condition"] == 1]],
        ["age_shifts_wholebrain_healthy", "age_shifts_wholebrain_ad"])
    pairwise_ttest(
        [df["age_shifts_diff_ventri"][df["disease_condition"] == 0], df["age_shifts_diff_ventri"][df["disease_condition"] == 1]],
        ["age_shifts_diff_ventri_healthy", "age_shifts_diff_ventri_ad"])

    pairwise_ttest(
        [df["wholebrain_lst_tad"][df["disease_condition"] == 0], df["wholebrain_lst_tad"][df["disease_condition"] == 1]],
        ["wholebrain_lst_tad_hc", "wholebrain_lst_tad_ad"])
    pairwise_ttest(
        [df["hippo_lst_tad"][df["disease_condition"] == 0], df["hippo_lst_tad"][df["disease_condition"] == 1]],
        ["hippo_lst_tad_hc", "hippo_lst_tad_ad"])
    pairwise_ttest(
        [df["hippo_lst_tad"][df["disease_condition"] == 0], df["hippo_lst_tad"][df["disease_condition"] == 1]],
        ["ventri_lst_tad_hc", "ventri_lst_tad_ad"])
    pairwise_ttest(
        [df["diff_ventri_lst_tad"][df["disease_condition"] == 0], df["diff_ventri_lst_tad"][df["disease_condition"] == 1]],
        ["diff_ventri_lst_tad_hc", "diff_ventri_lst_tad_ad"])
    print("================================================================================")

    # draw for paper
    df = df.drop(df[(df.disease_condition == 0)].index) # exclude HC cohort for AD plot
    draw_1x3_plot_AD(df)
    draw_1x3_plot_AD_tad(df)
    for name in ["age_shifts_wholebrain", "age_shifts_ventri", "age_shifts_hippo", "age_shifts_diff_ventri"]:
        data = df[name][df['scan_cdr'] == 0]
        print(statistics.mean(data), statistics.stdev(data))
        data = df[name][df['scan_cdr'] == 0.5]
        print(statistics.mean(data), statistics.stdev(data))
        data = df[name][df['scan_cdr'] == 1]
        print(statistics.mean(data), statistics.stdev(data))
        data = df[name][df['scan_cdr'] == 2]
        print(statistics.mean(data), statistics.stdev(data))

    # t-test
    for region in ['age_shifts_wholebrain', 'age_shifts_ventri', 'age_shifts_hippo', "age_shifts_diff_ventri"]:
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 0.5]],
            [f"{region}_CDR0", f"{region}_CDR0.5"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 1]],
            [f"{region}_CDR0", f"{region}_CDR1"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0.5], df[region][df["scan_cdr"] == 1]],
            [f"{region}_CDR0.5", f"{region}_CDR1"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 2]],
            [f"{region}_CDR0", f"{region}_CDR2"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0.5], df[region][df["scan_cdr"] == 2]],
            [f"{region}_CDR0.5", f"{region}_CDR2"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 1], df[region][df["scan_cdr"] == 2]],
            [f"{region}_CDR1", f"{region}_CDR2"])

    for region in ['wholebrain_lst_tad', 'ventri_lst_tad', 'hippo_lst_tad']:
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 0.5]],
            [f"{region}_CDR0", f"{region}_CDR0.5"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 1]],
            [f"{region}_CDR0", f"{region}_CDR1"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0.5], df[region][df["scan_cdr"] == 1]],
            [f"{region}_CDR0.5", f"{region}_CDR1"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 2]],
            [f"{region}_CDR0", f"{region}_CDR2"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 0.5], df[region][df["scan_cdr"] == 2]],
            [f"{region}_CDR0.5", f"{region}_CDR2"])
        pairwise_ttest(
            [df[region][df["scan_cdr"] == 1], df[region][df["scan_cdr"] == 2]],
            [f"{region}_CDR1", f"{region}_CDR2"])
    print("================================================================================")

csv_names = ['./figs/Vel30_invert_ADHC_wholebrain_ventri_hippo_diff_ven_all_withTem60Seg_atlasGAN_90seg1.csv']
for csv_name, svf_u0 in zip(csv_names, svfs):
    construct_csv(saved_svf_path, svf_u0, csv_name, saved = True)

# # ====================== compare between two cohort with divergence map ========
# def compare_cohort_maps(saved_svf_path, saved = False):
#     if saved == False:
#         saved_AD_path = f'{saved_svf_path}AD/'
#         saved_HC_path = f'{saved_svf_path}HC/'
#
#         AD_subjects_test = read_csv_set(meta_csv_path, disease=1, set = "test")
#         HC_subjects_test = read_csv_set(meta_csv_path, disease=0, set = "test")
#
#         # --- for AD subjects ---
#         W_div_man_AD = []
#         W_curl_man_AD  = []
#         numb_AD_subj = len(AD_subjects_test)
#         for count, subject_id in enumerate(AD_subjects_test):
#             img_npzs, disease_conditions = read_csv(meta_csv_path, subject_id)
#
#             imgs = [vt.load_npz(img_path)['vol'] for img_path in img_npzs]
#             segs = [vt.load_npz(img_path)['synth_seg'] for img_path in img_npzs]
#             ages = [vt.load_npz(img_path)['age'] for img_path in img_npzs]
#
#             print(f'==== {count}/{numb_AD_subj}: AD Subject {subject_id} with {len(imgs)} scans ====')
#
#             for num_img, (img, seg) in enumerate(zip(imgs, segs)):
#                 print(f"Age is {ages[num_img]}")
#                 saved_name = f'{saved_AD_path}{subject_id}_{ages[num_img]:.1f}.npz'
#                 load_npz = np.load(saved_name)
#                 W_div_man_AD.append(load_npz['W_div_man'])
#                 W_curl_man_AD.append(load_npz['W_curl_man'])
#
#         # --- for HC subjects ---
#         W_div_man_HC = []
#         W_curl_man_HC  = []
#         numb_HC_subj = len(HC_subjects_test)
#         for count, subject_id in enumerate(HC_subjects_test):
#             img_npzs, disease_conditions = read_csv(meta_csv_path, subject_id)
#
#             imgs = [vt.load_npz(img_path)['vol'] for img_path in img_npzs]
#             segs = [vt.load_npz(img_path)['synth_seg'] for img_path in img_npzs]
#             ages = [vt.load_npz(img_path)['age'] for img_path in img_npzs]
#
#             print(f'==== {count}/{numb_HC_subj}: HC Subject {subject_id} with {len(imgs)} scans ====')
#
#             # == registration pairwise ==
#             for num_img, (img, seg) in enumerate(zip(imgs, segs)):
#                 print(f"Age is {ages[num_img]}")
#                 saved_name = f'{saved_HC_path}{subject_id}_{ages[num_img]:.1f}.npz'
#                 load_npz = np.load(saved_name)
#                 W_div_man_HC.append(load_npz['W_div_man'])
#                 W_curl_man_HC.append(load_npz['W_curl_man'])
#
#         # save
#         W_div_man_AD = np.array(W_div_man_AD)
#         W_div_man_HC = np.array(W_div_man_HC)
#         W_curl_man_AD = np.array(W_curl_man_AD)
#         W_curl_man_HC = np.array(W_curl_man_HC)
#         np.savez_compressed(
#             f'{saved_svf_path}/W_div_man_AD.npz',
#             vol=W_div_man_AD
#         )
#         np.savez_compressed(
#             f'{saved_svf_path}/W_div_man_HC.npz',
#             vol=W_div_man_HC
#         )
#         np.savez_compressed(
#             f'{saved_svf_path}/W_curl_man_AD.npz',
#             vol=W_curl_man_AD
#         )
#         np.savez_compressed(
#             f'{saved_svf_path}/W_curl_man_HC.npz',
#             vol=W_curl_man_HC
#         )
#
#         # plot
#         dt.plot_3slices(np.mean(W_div_man_AD, axis=0), common_bar=True, cmap='seismic', title='W_div_man_AD')
#         dt.plot_3slices(np.mean(W_div_man_HC, axis=0), common_bar=True, cmap='seismic', title='W_div_man_HC')
#         dt.plot_3slices(np.mean(W_div_man_AD, axis=0) - np.mean(W_div_man_HC, axis=0), common_bar=True, cmap='seismic', title='W_div_man_AD-HC')
#         dt.plot_3slices(np.mean(W_curl_man_AD, axis=0)[0], common_bar=True, cmap='seismic', title='W_curl_man_AD')
#         dt.plot_3slices(np.mean(W_curl_man_HC, axis=0)[0], common_bar=True, cmap='seismic', title='W_curl_man_HC')
#         dt.plot_3slices(np.mean(W_curl_man_AD, axis=0)[0] - np.mean(W_curl_man_HC, axis=0)[0], common_bar=True, cmap='seismic',
#                         title='W_curl_man_AD-HC')
#     else:
#         W_div_man_AD = np.load(f'{saved_svf_path}/HC/W_div_man_AD.npz')['vol']
#         W_div_man_HC = np.load(f'{saved_svf_path}/W_div_man_HC.npz')['vol']
#         W_curl_man_AD = np.load(f'{saved_svf_path}/W_curl_man_AD.npz')['vol']
#         W_curl_man_HC = np.load(f'{saved_svf_path}/W_curl_man_HC.npz')['vol']
#
#         # plot
#         dt.plot_3slices(np.mean(W_div_man_AD, axis=0), common_bar=True, cmap='seismic', title='W_div_man_AD')
#         dt.plot_3slices(np.mean(W_div_man_HC, axis=0), common_bar=True, cmap='seismic', title='W_div_man_HC')
#         dt.plot_3slices(np.mean(W_div_man_AD, axis=0) - np.mean(W_div_man_HC, axis=0), common_bar=True, cmap='seismic', title='W_div_man_AD-HC')
#         dt.plot_3slices(np.mean(W_curl_man_AD, axis=0)[0], common_bar=True, cmap='seismic', title='W_curl_man_AD')
#         dt.plot_3slices(np.mean(W_curl_man_HC, axis=0)[0], common_bar=True, cmap='seismic', title='W_curl_man_HC')
#         dt.plot_3slices(np.mean(W_curl_man_AD, axis=0)[0] - np.mean(W_curl_man_HC, axis=0)[0], common_bar=True, cmap='seismic',
#                         title='W_curl_man_AD-HC')
#
# compare_cohort_maps(saved_svf_path, True)
#
# # ============================= measure AD tk according to CDR scores ================================================
# cdr_csv_path = f'{main_path}OASIS3_UDSb4_cdr.csv'
# def load_cdr_forSubject(cdr_csv_path, subject_id, age):
#     df = pd.read_csv(cdr_csv_path)
#     sub = {}
#     for head in df.head():
#         sub[head] = df[head][df["OASISID"]==subject_id]
#     index = np.argmin(np.abs(sub["age at visit"] - age))
#     output=np.array(sub["CDRTOT"])[index]
#     return output
#
# def plot_AD_tk_according_CDR(saved_svf_path, svf_u0, csv_name, saved = True):
#     seg_90_img = np.transpose(vt.load_nii(seg_90), (2, 1, 0))
#     seg_60_img = np.transpose(vt.load_nii(seg_60), (2, 1, 0))
#
#     if saved == False:
#         saved_AD_path = f'{saved_svf_path}AD/'
#
#         AD_subjects_test = read_csv_set(meta_csv_path, disease=1, set="test")
#
#         AD_tk_with_cdr = {}
#         subject_ids = []
#         subject_ages = []
#         age_shifts_wholebrain = []
#         age_shifts_ventri = []
#         age_shifts_hippo = []
#         scan_cdr = []
#
#         ventri_lst_tad = []
#         hippo_lst_tad = []
#         wholebrain_lst_tad = []
#
#         # --- for AD subjects ---
#         numb_AD_subj = len(AD_subjects_test)
#         for count, subject_id in enumerate(AD_subjects_test):
#             img_npzs, disease_conditions = read_csv(meta_csv_path, subject_id)
#
#             imgs = [vt.load_npz(img_path)['vol'] for img_path in img_npzs]
#             segs = [vt.load_npz(img_path)['synth_seg'] for img_path in img_npzs]
#             ages = [vt.load_npz(img_path)['age'] for img_path in img_npzs]
#
#             print(f'==== {count}/{numb_AD_subj}: AD Subject {subject_id} with {len(imgs)} scans ====')
#
#             for num_img, (img, seg) in enumerate(zip(imgs, segs)):
#                 print(f"Age is {ages[num_img]}")
#                 saved_name = f'{saved_AD_path}{subject_id}_{ages[num_img]:.1f}.npz'
#                 load_npz = np.load(saved_name)
#                 vel_half = load_npz['vel_half']
#                 age_gap = abs(ages[num_img] - 60)
#                 # tk_3d, w_ad = calculate_tk_wAD(vel_half.squeeze(), svf_u0.squeeze())
#                 # # TODO: think about how to decide seg: Now using seg60
#                 # age_gap = ages[num_img] - 60
#                 # t_hc0, t_ad0 = calculate_tk_given_labels("wholebrain", seg[::2, ::2, ::2], tk_3d, w_ad, age_gap)
#                 # t_hc, t_ad = calculate_tk_given_labels(labels_ventricles, seg[::2, ::2, ::2], tk_3d, w_ad, age_gap)
#                 # t_hc1, t_ad1 = calculate_tk_given_labels(labels_hippocampus, seg[::2, ::2, ::2], tk_3d, w_ad, age_gap)
#                 t_hc0, t_ad0 = calcu_biomarkers("wholebrain", seg_90_img[::2, ::2, ::2], vel_half.squeeze(),
#                                                 svf_u0.squeeze(), age_gap, quantile=0.995)
#                 t_hc, t_ad = calcu_biomarkers(labels_ventricles, seg_90_img[::2, ::2, ::2], vel_half.squeeze(),
#                                               svf_u0.squeeze(), age_gap, quantile=0.995)
#                 t_hc1, t_ad1 = calcu_biomarkers(labels_hippocampus, seg_90_img[::2, ::2, ::2], vel_half.squeeze(),
#                                                 svf_u0.squeeze(), age_gap, quantile=0.87)
#
#                 # save data to lsts
#                 subject_ids.append(subject_id)
#                 subject_ages.append(ages[num_img])
#                 age_shifts_ventri.append(t_hc - age_gap)
#                 age_shifts_hippo.append(t_hc1-age_gap)
#                 age_shifts_wholebrain.append(t_hc0-age_gap)
#
#                 ventri_lst_tad.append(t_ad)
#                 hippo_lst_tad.append(t_ad1)
#                 wholebrain_lst_tad.append(t_ad0)
#
#                 scan_cdr.append(load_cdr_forSubject(cdr_csv_path, subject_id, ages[num_img]))
#
#
#         AD_tk_with_cdr["subject_ids"]=subject_ids
#         AD_tk_with_cdr["subject_ages"]=subject_ages
#         AD_tk_with_cdr["age_shifts_wholebrain"]=age_shifts_wholebrain
#         AD_tk_with_cdr["age_shifts_ventri"]=age_shifts_ventri
#         AD_tk_with_cdr["age_shifts_hippo"]=age_shifts_hippo
#         AD_tk_with_cdr["scan_cdr"]=scan_cdr
#
#         AD_tk_with_cdr["ventri_lst_tad"]=ventri_lst_tad
#         AD_tk_with_cdr["hippo_lst_tad"]=hippo_lst_tad
#         AD_tk_with_cdr["wholebrain_lst_tad"]=wholebrain_lst_tad
#         pd.DataFrame(AD_tk_with_cdr).to_csv(csv_name, index=False)
#
#     # ============ plot =====================
#     # Load CSV data into Pandas DataFrame
#     print(csv_name)
#     df = pd.read_csv(csv_name)
#
#     # draw for paper
#     draw_1x3_plot_AD(df)
#     draw_1x3_plot_AD_tad(df)
#     for name in ["age_shifts_wholebrain", "age_shifts_ventri", "age_shifts_hippo"]:
#         data = df[name][df['scan_cdr'] == 0]
#         print(statistics.mean(data), statistics.stdev(data))
#         data = df[name][df['scan_cdr'] == 0.5]
#         print(statistics.mean(data), statistics.stdev(data))
#         data = df[name][df['scan_cdr'] == 1]
#         print(statistics.mean(data), statistics.stdev(data))
#
#     # # plot
#     # plt.rcParams.update({'font.size': 22})
#     # fig, axes = plt.subplots(1, 1, figsize=(8, 8),  dpi=300)
#     # my_pal = {0: "green", 0.5: "gold", 1: "red"}
#     # sns.boxplot(x='scan_cdr', y='age_shifts_ventri', data=df, ax=axes, palette=my_pal)
#     # plt.savefig(f'./figs/CDR_AD.png', bbox_inches='tight')
#     # plt.show()
#
#     # # Create a 2x2 figure
#     # fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
#     #
#     # # Plot first subplot
#     # sns.boxplot(x='scan_cdr', y='age_shifts_wholebrain', data=df, ax=axs[0, 0])
#     # axs[0, 0].set_title('Age Shifts wholebrain')
#     #
#     # # Plot second subplot
#     # sns.boxplot(x='scan_cdr', y='age_shifts_ventri', data=df, ax=axs[0, 1])
#     # axs[0, 1].set_title('Age Shifts ventri')
#     #
#     # # Plot third subplot
#     # sns.boxplot(x='scan_cdr', y='age_shifts_hippo', data=df, ax=axs[1, 0])
#     # axs[1, 0].set_title('Age Shifts Hippo')
#     #
#     # # Show the figure
#     # plt.suptitle(csv_name)
#     # plt.show()
#
#     # t-test
#     for region in ['age_shifts_wholebrain', 'age_shifts_ventri', 'age_shifts_hippo']:
#         pairwise_ttest(
#             [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 0.5]],
#             [f"{region}_CDR0", f"{region}_CDR0.5"])
#         pairwise_ttest(
#             [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 1]],
#             [f"{region}_CDR0", f"{region}_CDR1"])
#         pairwise_ttest(
#             [df[region][df["scan_cdr"] == 0.5], df[region][df["scan_cdr"] == 1]],
#             [f"{region}_CDR0.5", f"{region}_CDR1"])
#
#     for region in ['wholebrain_lst_tad', 'ventri_lst_tad', 'hippo_lst_tad']:
#         pairwise_ttest(
#             [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 0.5]],
#             [f"{region}_CDR0", f"{region}_CDR0.5"])
#         pairwise_ttest(
#             [df[region][df["scan_cdr"] == 0], df[region][df["scan_cdr"] == 1]],
#             [f"{region}_CDR0", f"{region}_CDR1"])
#         pairwise_ttest(
#             [df[region][df["scan_cdr"] == 0.5], df[region][df["scan_cdr"] == 1]],
#             [f"{region}_CDR0.5", f"{region}_CDR1"])
#
#     print("================================================================================")
#
# # csv_names = ['./figs/Vel30_invert_AD_ventri_hippo_test_CDR.csv']
# csv_names = ['./figs/Vel30_invert_AD_wholebrain_ventri_hippo_test_CDR.csv']
# for csv_name, svf_u0 in zip(csv_names, svfs):
#     plot_AD_tk_according_CDR(saved_svf_path, svf_u0, csv_name, saved = False)