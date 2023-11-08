#!/usr/bin/env python
# -*- coding:utf-8 -*-

# First extract voxels in real test dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import src.visualize_tools as vt

def read_csv(set = "train"):
    path =  f'{main_path}FS_scans_metadata_with_split.csv'
    df = pd.read_csv(path)
    ids_set = df["ID"][df["Partition"]==set]
    ids_set = [fpath+i.split("_")[0]+"_"+i.split("_")[2]+'.npz' for i in list(ids_set)]
    cohorts = df["HC0/AD1"][df["Partition"]==set]
    cohorts = [i for i in list(cohorts)]
    return ids_set, cohorts

def read_train_AD_csv(set = "train"):
    path =  f'{main_path}FS_scans_metadata_with_split.csv'
    df = pd.read_csv(path)
    ids_set = df["ID"][(df["Partition"]==set)  & (df["HC0/AD1"]==1)]
    ids_set = [fpath+i.split("_")[0]+"_"+i.split("_")[2]+'.npz' for i in list(ids_set)]
    cohorts = df["HC0/AD1"][(df["Partition"]==set)  & (df["HC0/AD1"]==1)]
    cohorts = [i for i in list(cohorts)]
    return ids_set, cohorts

def load_img_all(img_path, load_synthseg=False):
    img = np.load(img_path)['vol']
    if load_synthseg:
        seg = np.load(img_path)['synth_seg']
    else:
        seg = np.load(img_path)['seg']
    age = np.load(img_path)['age']
    return img, seg, int(age)

def extract_voxels_in_seg(seg, labels, sumup = True):
    voxels = []
    for label in labels:
        voxels.append(np.count_nonzero(seg == label))

    if sumup:
        voxels = sum(voxels)
    return voxels

def calc_vx_given_labels(img_paths, labels, img_cohorts, cohorts_flag =False, partition = 'Test', load_synthseg=False):
    ages = []
    voxels = []
    cohorts = []
    print(f'load_synthseg={load_synthseg}')
    for img_path in img_paths:
        img, seg, age = load_img_all(img_path, load_synthseg=load_synthseg)
        # set age range as 60~80
        if age in range(60, 81):
            voxel = extract_voxels_in_seg(seg, labels)

            ages.append(age)
            voxels.append(voxel)
            if cohorts_flag: cohorts.append(f'{partition}/HC' if img_cohorts[img_paths.index(img_path)] == 0 else f'{partition}/AD')
    return ages, voxels, cohorts

def calc_vax_given_templates_list(list_templates_segs, labels, cohort_name):
    ages = []
    voxels = []
    cohorts = []
    for list_count, templates_seg in enumerate(list_templates_segs):
        seg = vt.load_nii(templates_seg)
        ages.append(range(60, 81)[list_count])
        voxels.append(extract_voxels_in_seg(seg, labels))
        cohorts.append(cohort_name)

    return ages, voxels, cohorts

def calc_vox_given_templates_list_ratio(list_templates_segs, labels1, labels2, cohort_name):
    ages = []
    voxels = []
    cohorts = []
    for list_count, templates_seg in enumerate(list_templates_segs):
        seg = vt.load_nii(templates_seg)
        ages.append(range(60, 81)[list_count])
        vox1 = extract_voxels_in_seg(seg, labels1)
        vox2 = extract_voxels_in_seg(seg, labels2)
        voxels.append(vox1/vox2)
        cohorts.append(cohort_name)

    return ages, voxels, cohorts

def plot_given_structure(ages, voxels, cohorts, columns, save_path_name = None):
    data = {'age': ages, 'Number of Voxels': voxels, 'Cohorts': cohorts}
    df = pd.DataFrame(data, columns = columns)
    plt.rcParams.update({'font.size': 14})
    sns.lineplot(data=df, x="age", y="Number of Voxels", hue='Cohorts',  style="Cohorts", markers=True)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    if save_path_name:
        plt.savefig(save_path_name, bbox_inches='tight')
    plt.show()

def draw_final_plot_given_labels(labels,  save_path_name=None):
    ages1, voxels1, cohorts1 = calc_vax_given_templates_list(list_HC_segs, labels, 'Templates/HC')
    ages2, voxels2, cohorts2 = calc_vax_given_templates_list(list_AD_segs, labels, 'Templates/AD')

    ages1.extend(ages2)
    voxels1.extend(voxels2)
    cohorts1.extend(cohorts2)

    plot_given_structure(ages1, voxels1, cohorts1, columns=['age', 'Number of Voxels', 'Cohorts'], save_path_name=save_path_name)

def draw_final_plot_given_labels1(labels, img_paths_test, img_cohorts_test, img_paths_train, img_cohorts_train, save_path_name=None):
    ages, voxels, cohorts = calc_vx_given_labels(img_paths_test, labels, img_cohorts_test, cohorts_flag=True, load_synthseg=True)
    ages0, voxels0, cohorts0 = calc_vx_given_labels(img_paths_train, labels, img_cohorts_train, cohorts_flag=True, partition='Train', load_synthseg=True)

    ages1, voxels1, cohorts1 = calc_vax_given_templates_list(list_HC_segs, labels, 'Templates/HC')
    ages2, voxels2, cohorts2 = calc_vax_given_templates_list(list_AD_segs, labels, 'Templates/AD')

    ages.extend(ages1)
    voxels.extend(voxels1)
    cohorts.extend(cohorts1)

    ages.extend(ages2)
    voxels.extend(voxels2)
    cohorts.extend(cohorts2)

    ages.extend(ages0)
    voxels.extend(voxels0)
    cohorts.extend(cohorts0)

    plot_given_structure(ages, voxels, cohorts, columns=['age', 'Number of Voxels', 'Cohorts'], save_path_name = save_path_name)

def draw_final_plot_given_labels_usingSynthSeg(labels, img_paths_test, img_cohorts_test, cohort_tag = 'AD', save_path_name=None):
    ages, voxels, cohorts = calc_vx_given_labels(img_paths_test, labels, img_cohorts_test, cohorts_flag=True, load_synthseg=True)

    ages1, voxels1, cohorts1 = calc_vax_given_templates_list(list_HC_segs, labels, f'Templates/{cohort_tag}')

    ages.extend(ages1)
    voxels.extend(voxels1)
    cohorts.extend(cohorts1)

    plot_given_structure(ages, voxels, cohorts, columns=['age', 'Number of Voxels', 'Cohorts'], save_path_name = save_path_name)

def draw_final_plot_given_labels_ratio(labels1, labels2, save_path_name=None):
    ages1, voxels1, cohorts1 = calc_vox_given_templates_list_ratio(list_HC_segs, labels1, labels2, 'Templates/HC')
    ages2, voxels2, cohorts2 = calc_vox_given_templates_list_ratio(list_AD_segs, labels1, labels2, 'Templates/AD')

    ages1.extend(ages2)
    voxels1.extend(voxels2)
    cohorts1.extend(cohorts2)

    plot_given_structure(ages1, voxels1, cohorts1, columns=['age', 'Number of Voxels', 'Cohorts'], save_path_name=save_path_name)

if __name__ == "__main__":
    # =========== use MAIA ======================
    # overall parameters - path of real data
    main_path = './data/OASIS3/'
    fpath = main_path + 'all_npz/'

    # path of templates
    # -- HConly model --
    templates_path = './trained_models/HC_only/my_plot_1e-4/'  # sharpest
    model_name = 'Ep300_HConly_synthSegAll'

    print(f'model name is {model_name}')

    list_HC_segs = [f'{templates_path}age_{i}disease_0_SynthSeg.nii.gz' for i in range(60, 81)]
    list_AD_segs = [f'{templates_path}age_{i}disease_1_SynthSeg.nii.gz' for i in range(60, 81)]

    labels_ventricles  = [ 4, 14, 15, 43]
    labels_hippocampus = [17, 53, 18, 54]
    labels_WM          = [ 2,  7, 41, 46]
    labels_brain_stem  = [ 16 ]

    # single cohort
    # read real data
    img_paths, img_cohorts = read_csv("dev")
    img_paths.extend(read_csv("test")[0])
    img_cohorts.extend(read_csv("test")[1])

    img_paths_test = img_paths
    img_cohorts_test = img_cohorts

    # ventricles
    save_path_name = f'./figs/ventricles_syntheSeg{model_name}.pdf'
    draw_final_plot_given_labels_usingSynthSeg(labels_ventricles, img_paths_test, img_cohorts_test,  cohort_tag = 'HC', save_path_name=save_path_name)

    # hippocampus
    save_path_name = f'./figs/hippocampus_syntheSeg{model_name}.pdf'
    draw_final_plot_given_labels_usingSynthSeg(labels_hippocampus, img_paths_test, img_cohorts_test, cohort_tag = 'HC', save_path_name=save_path_name)

    # WM
    save_path_name = f'./figs/WM_syntheSeg1{model_name}.pdf'
    draw_final_plot_given_labels_usingSynthSeg(labels_WM, img_paths_test, img_cohorts_test,  cohort_tag = 'HC', save_path_name=save_path_name)

    # brain_stem
    save_path_name = f'./figs/BrainStem_syntheSeg{model_name}.pdf'
    draw_final_plot_given_labels_usingSynthSeg(labels_brain_stem, img_paths_test, img_cohorts_test, cohort_tag = 'HC', save_path_name=save_path_name)

