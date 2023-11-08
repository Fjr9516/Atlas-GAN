#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import statistics
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import src.visualize_tools as vt

def EFC(nii_img, img_mask = None):
    '''Entropy focus criteria:
    Ref: https://rdrr.io/github/TKoscik/nifti.qc/man/nii.qc.efc.html'''
    img = vt.load_nii(nii_img, way='sitk')
    #vt.TEST_Ori_3slice(img, title='img')
    if img_mask:
        img_mask = np.load(img_mask)['vol']
        #vt.TEST_Ori_3slice(img_mask, title='img_mask')
        mask = img_mask < 1e-6
        mask_img = np.ma.masked_array(img, mask = mask)
        B_max = np.sqrt(np.sum(np.square(mask_img)))
        efc = - np.sum((img/B_max)*np.log((np.abs(img)+1e-16)/B_max)) # add 1e-16 to deal with log(0) being Inf
        n = mask.sum()
        efc = - efc / (n/np.sqrt(n)*np.log(1/np.sqrt(n)))
    else:
        B_max = np.sqrt(np.sum(np.square(img)))
        efc = - np.sum((img/B_max)*np.log((np.abs(img)+1e-16)/B_max)) # add 1e-16 to deal with log(0) being Inf
        n = img.size
        efc = - efc / (n/np.sqrt(n)*np.log(1/np.sqrt(n)))
    return efc

def calcu_efc(model_path, mask=None):
    print(f'Model path is {model_path}!')
    list = []
    cohort_lst = []
    for nii in [i for i in os.listdir(model_path) if i.endswith('.nii.gz')]:
        if len(nii) < 23: # test only for image
            if int(nii[4:6])<80:
                disease_cond = nii[-8]
                if mask:
                    list.append(EFC(model_path + nii, mask))
                else:
                    list.append(EFC(model_path + nii))
                cohort_lst.append(disease_cond)
    print(f'Model EFC mean = {statistics.mean(list)}; std = {statistics.stdev(list)}')
    return list, cohort_lst

def efc_plot(np_array, order, colms, save_name = 'EFCs'):
    # array = np.load('EFC1.npz')
    np_array = np_array[order]  # [1,3,0,2][2,0,3,1]

    lst1 = []
    lst2 = []
    for i in range(np_array.shape[0]):
        lst1.extend(np_array[i, :])
        lst2.extend([colms[i]] * len(np_array[i, :]))

    plt.figure(figsize=(8, 7), dpi=100)
    plt.rcParams.update({'font.size': 22})

    df = pd.DataFrame(list(zip(lst1, lst2)),
                      columns=['Entropy Focus Criteria', 'Models'])

    sns.boxplot( # violinplot
        data=df, x="Models", y="Entropy Focus Criteria",
        palette = "deep"#"deep"/"pastel", saturation=0.5
    )
    plt.savefig(f'./figs/{save_name}.pdf', bbox_inches='tight')

    plt.show()

def main( mask , saved = False, main_dir = '/proj/outputs/plots/', save_name='EFC_Wmask_60to80'):
    if saved == False:
        model_list =[]

        save_path = [f'{main_dir}my_plot_1e-3/',
                     f'{main_dir}my_plot_1e-4/',
                     f'{main_dir}my_plot_5e-3/',
                     f'{main_dir}my_plot_5e-4/']
        save_path_name=None
        for i, model_path in enumerate(save_path):
            if i == len(save_path)-1:
                save_path_name = f'./temp_files/{save_name}.npz'
            model_list.append(calcu_efc(model_path, mask)[0])

            if save_path_name != None:
                print(f'Saving to {save_path_name}')
                model_list = np.array(model_list)
                np.savez_compressed(
                    save_path_name,
                    vol=model_list,
                )
    else:
        model_list = np.load(f'./temp_files/{save_name}.npz', allow_pickle=True)['vol']
    efc_plot(np.array(model_list), [1, 3, 0, 2], ['1e-4','5e-4', '10e-4', '50e-4'], save_name)

if __name__ == "__main__":
    main(saved = True, mask = './linearaverageof100.npz',
         main_dir='./plots/',
         save_name='EFC_Wmask_60to80_HConly_300ep')