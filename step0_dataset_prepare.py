#!/usr/bin/env python
# -*- coding:utf-8 -*-

import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import src.visualize_tools as vt

# paths
PATH_TO_DATASET = '/PATH/TO/RAW/DATASET'
PATH_TO_METADATA = '/PATH/TO/METADATA'
SAVE_NPZ_PATH = '/SAVE/PATH/NPZ/'
AVG_IMG_PATH    = f"{PATH_TO_DATASET}/linearaverageof100.npz"
OASIS_FS_old_path = "/PATH/TO/OLD/DATA"
OASIS_FS_extra_path = "/PATH/TO/NEW/DATA"
FS_metadata_csv = f"{PATH_TO_METADATA}/oasis-scripts/download_freesurfer/OASIS3_Freesurfer_output_221012.csv"
CDR_metadata_csv = f"{PATH_TO_METADATA}/OASIS3_data_files/scans/UDSb4-Form_B4__Global_Staging__CDR__Standard_and_Supplemental/resources/csv/files/OASIS3_UDSb4_cdr.csv"
HC_csv = f"{PATH_TO_METADATA}/OASIS_cohort_files/scans/CogNorm-Cognitively_Normal_Cohorts/resources/csv/files/OASIS3_unchanged_CDR_cognitively_healthy.csv"
demograpgics_csv = f"{PATH_TO_METADATA}/OASIS3_data_files/scans/demo-demographics/resources/csv/files/OASIS3_demographics.csv"


# calculate statistical size from 200 randomly selected scans
# / calculate avg_img from 100 randomly selected scans
def crop_size_calc(OASIS_FS_old_path, nb_samples = 200):
    '''
    Original scan is 256X256X256 after FreeSurfer process, reduce size to speed up training
    Reduced: 208X176X160 (size should be a multiple of 16 to fit U-net)
    '''
    list_old = [OASIS_FS_old_path + x + '/mri/align_norm.nii.gz' for x in os.listdir(OASIS_FS_old_path) if os.path.isdir(OASIS_FS_old_path+x)]

    def avg_template(list_old, nb_samples):
        random_idx = np.random.choice(
            len(list_old), nb_samples, replace=False,
        )

        list_old = np.array(list_old)
        selected_imgs = list_old[random_idx]
        template = 0
        for s_img in selected_imgs:
            simg = sitk.ReadImage(s_img, sitk.sitkFloat32)
            simg = vt.rescale(simg)
            template += sitk.GetArrayFromImage(simg)
        template /= len(selected_imgs)
        return template
    def calcu_mid(template):
        x_sum = np.apply_over_axes(np.sum, template, [1, 2])
        y_sum = np.apply_over_axes(np.sum, template, [0, 2])
        z_sum = np.apply_over_axes(np.sum, template, [0, 1])

        def find_positive_range(x_sum):
            range = np.where(x_sum > 0)
            print("From {} to {}, range {}".format(np.min(range), np.max(range), np.max(range) - np.min(range)))
            return np.min(range), np.max(range), np.max(range) - np.min(range)

        def myplot(x_sum):
            plt.plot(x_sum)
            plt.ylabel('x_sum')
            plt.show()
            return find_positive_range(x_sum)

        x_start, x_end, out_x = myplot(x_sum.flatten())
        y_start, y_end, out_y = myplot(y_sum.flatten())
        z_start, z_end, out_z = myplot(z_sum.flatten())
        return x_start, y_start, z_start, out_x, out_y, out_z

    template200 = avg_template(list_old, nb_samples)
    calcu_mid(template200)

    uppoint = [7, 37, 48]
    out_size = [208, 176, 160]
    template_crop = vt.crop_ndarray(template200, uppoint=uppoint,
                                    out_size=out_size, show=True)

    # calculate average template from 100 randomly selected scans
    def crop_save_template(npy_img, save_path):
        # need to both crop the img and seg
        # uppoint = [3, 30, 42]
        # out_size = [216, 190, 172]
        uppoint = [7, 37, 48]
        out_size = [208, 176, 160]
        npy_img = vt.crop_ndarray(npy_img, uppoint=uppoint, out_size=out_size)

        # Assuming that you have 'age' and 'attribute' loaded:
        # save_path: i.e. './data/dataset_name/train_npz/fname.npz'
        np.savez_compressed(
            save_path,
            vol=npy_img
        )

    crop_save_template(avg_template(list_old, 100), AVG_IMG_PATH)

# construct train data
def construct_train_data():
    def harmonize_ID_string(old_name):
        new_name = old_name.split("_")[0] + "_" + old_name.split("_")[2]
        return new_name

    list_old = [x for x in os.listdir(OASIS_FS_old_path) if os.path.isdir(OASIS_FS_old_path + x)]
    list_extra = [x for x in os.listdir(OASIS_FS_extra_path) if os.path.isdir(OASIS_FS_extra_path + x)]
    print("--- OASIS-3 with {}+{}={} FS scans in total ---".format(len(list_old), len(list_extra),
                                                                   len(list_old + list_extra)))
    def FS_QC_status_failed_scans():
        df = pd.read_csv(FS_metadata_csv)
        delete_MR_sessions = df['MR_session'][df['FS QC Status'] == "Quarantined"]
        delete_MR_sessions = [harmonize_ID_string(old) for old in list(delete_MR_sessions)]
        print("-- MR sessions with bad QC are {} --".format(delete_MR_sessions))
        return delete_MR_sessions, [x.split("_")[0] for x in df["MR_session"].to_numpy()]

    FS_QC_failed_scans, FS_subj = FS_QC_status_failed_scans()
    Undifined_CDR = "OAS30753_d0035"

    def load_all_metadata_Dict(CDR_metadata_csv):
        df = pd.read_csv(CDR_metadata_csv)
        OASIS_session_label = list(df['OASIS_session_label'])
        MMSE = list(df["MMSE"])
        CDRTOT = list(df["CDRTOT"])
        dx1 = list(df["dx1"])
        ages = list(df["age at visit"])
        return OASIS_session_label, MMSE, CDRTOT, dx1, ages

    IDs, MMSEs, CDRs, Dignoses, ages = load_all_metadata_Dict(CDR_metadata_csv)

    Harmonized_IDs = [harmonize_ID_string(x) for x in IDs]
    # find AD subject: any AD found in any visit, assigned as AD for this subject
    Subject_ID = [x.split("_")[0] for x in IDs]

    AD_sub = []
    for count, dignose in enumerate(Dignoses):
        if dignose in ['AD Dementia', 'AD dem Language dysf after', 'AD dem Language dysf prior',
                       'AD dem Language dysf with',
                       'AD dem cannot be primary', 'AD dem distrubed social, after', 'AD dem distrubed social, prior',
                       'AD dem distrubed social, with',
                       'AD dem visuospatial, after', 'AD dem visuospatial, prior', 'AD dem visuospatial, with',
                       'AD dem w/CVD contribut',
                       'AD dem w/CVD not contrib', 'AD dem w/Frontal lobe/demt at onset',
                       'AD dem w/PDI after AD dem contribut',
                       'AD dem w/PDI after AD dem not contrib', 'AD dem w/depresss, contribut',
                       'AD dem w/depresss, not contribut', 'AD dem w/oth (list B) contribut',
                       'AD dem w/oth (list B) not contrib', 'AD dem w/oth unusual feat/subs demt',
                       'AD dem w/oth unusual features', 'AD dem w/oth unusual features/demt on',
                       'AD dem/FLD prior to AD dem''DAT', 'DAT Language dysf after', 'DAT distrubed social, after',
                       'DAT distrubed social, prior', 'DAT w/CVD contribut', 'DAT w/CVD not contrib',
                       'DAT w/PDI after DAT not contrib',
                       'DAT w/depresss, contribut', 'DAT w/depresss, not contribut']:
            # print(dignose)
            AD_sub.append(Subject_ID[count])
    AD_sub = np.unique(AD_sub)
    print(">> AD_sub with {} subjects".format(len(AD_sub)))

    # find healthy using provided CDR csv
    def HC_subjects(HC_csv):
        df = pd.read_csv(HC_csv)
        return df["OASIS3_id"]

    HC_sub = HC_subjects(HC_csv).to_numpy()
    print(">> HC_sub with {} subjects".format(len(HC_sub)))

    Other_dementia_sub = []
    hc_i = 0
    ad_i = 0
    for i, subj in enumerate(np.unique(Subject_ID)):  # all subjects: Subject_ID; FS subject: FS_subj
        # print("> {} <".format(i))
        if subj in HC_sub:
            hc_i += 1
            # print("HC", hc_i)
            continue
        elif subj in AD_sub:
            ad_i += 1
            # print("AD", ad_i)
            continue
        else:
            # print("Other")
            Other_dementia_sub.append(subj)
    print(">> Other_dementia_sub with {} subjects".format(len(Other_dementia_sub)))

    for hc in HC_sub:
        if hc in AD_sub:
            print("Wrong one assigned to both AD and HC".format(hc))  # checked ok

    # split according to the remaining scans
    def load_all_metadata_FS():
        df = pd.read_csv(FS_metadata_csv)
        MR_session = list(df['MR_session'])
        return MR_session

    FS_IDs = load_all_metadata_FS()  # should be 2681 scans

    # after deleting:
    FS_IDs_after = []
    for FS_id in FS_IDs:
        if harmonize_ID_string(FS_id) in FS_QC_failed_scans:
            continue
        elif harmonize_ID_string(FS_id) == Undifined_CDR:
            continue
        elif harmonize_ID_string(FS_id)[:8] in Other_dementia_sub:
            continue
        else:
            FS_IDs_after.append(FS_id)

    # okay, now lets see the statistic info from the selected 2366 images
    FS_subj_after = np.unique([harmonize_ID_string(x)[:8] for x in FS_IDs_after])
    print("After selection, we have {} subjects in total of {} scans in total".format(len(FS_subj_after),
                                                                                      len(FS_IDs_after)))
    hc = 0
    ad = 0
    hc_scan = 0
    ad_scan = 0
    for fs_sub in FS_subj_after:
        if fs_sub in HC_sub:
            hc += 1
        if fs_sub in AD_sub:
            ad += 1

    for scan in FS_IDs_after:
        if scan[:8] in HC_sub:
            hc_scan += 1
        if scan[:8] in AD_sub:
            ad_scan += 1
    print("After selection, we have {} HC subjects with {} scans, and {} AD subjects with {} scans".format(hc, hc_scan,
                                                                                                           ad, ad_scan))

    # construct metadata for each scan
    # Harmonized_IDs, MMSEs, CDRs, Dignoses, ages
    FS_metadata_after = {}
    FS_metadata_after["ID"] = []
    FS_metadata_after["Path_of_img"] = []
    FS_metadata_after["Path_of_seg"] = []
    FS_metadata_after["age"] = []
    # FS_metadata_after["MMSE"] = []
    # FS_metadata_after["CDR"] = []
    FS_metadata_after["HC0/AD1"] = []

    # load age at entry ages from demographics
    def load_entry_ages():
        df = pd.read_csv(demograpgics_csv)
        MR_session = list(df['OASISID'])
        AgeatEntry = list(df['AgeatEntry'])
        return MR_session, AgeatEntry

    Demo_IDs, Demo_angesentry = load_entry_ages()

    # extract data and form new npz
    def constrcut_npz(save_path, img_path, seg_path, age, disease_condition):
        if os.path.exists(save_path):
            return 0
        simg = sitk.ReadImage(img_path, sitk.sitkFloat32)
        simg = vt.rescale(simg)
        npy_img = sitk.GetArrayFromImage(simg)

        sseg = sitk.ReadImage(seg_path, sitk.sitkInt16)
        npy_seg = sitk.GetArrayFromImage(sseg)

        # need to both crop the img and seg
        uppoint = [7, 37, 48]
        out_size = [208, 176, 160]
        npy_img = vt.crop_ndarray(npy_img, uppoint=uppoint, out_size=out_size)
        npy_seg = vt.crop_ndarray(npy_seg, uppoint=uppoint, out_size=out_size)

        # Assuming that you have 'age' and 'attribute' loaded:
        # save_path: i.e. './data/dataset_name/train_npz/fname.npz'
        np.savez_compressed(
            save_path,
            vol=npy_img,
            seg=npy_seg,
            age=age,
            disease_condition=disease_condition,
        )

    for scan in FS_IDs_after:
        if scan in list_old:
            path_of_img = OASIS_FS_old_path + scan + "/mri/align_norm.nii.gz"
            path_of_seg = OASIS_FS_old_path + scan + "/mri/align_aseg.nii.gz"
        elif scan in list_extra:
            path_of_img = OASIS_FS_extra_path + scan + "/mri/align_norm.nii.gz"
            path_of_seg = OASIS_FS_extra_path + scan + "/mri/align_aseg.nii.gz"
        else:
            raise ValueError('Unexpected FS scan!')

        if scan[:8] in HC_sub:
            disease_condition = 0
        elif scan[:8] in AD_sub:
            disease_condition = 1
        else:
            raise ValueError('Unexpected disease conditions!')

        # calculate age from entry age
        sub_scan = scan.split("_")[0]
        entryage = Demo_angesentry[Demo_IDs.index(sub_scan)]
        day_scan = int(scan.split("_")[2][1:])
        age = float(entryage + day_scan / 365)

        FS_metadata_after["ID"].append(scan)
        FS_metadata_after["Path_of_img"].append(path_of_img)
        FS_metadata_after["Path_of_seg"].append(path_of_seg)
        FS_metadata_after["HC0/AD1"].append(disease_condition)
        FS_metadata_after["age"].append(age)

        if harmonize_ID_string(scan) in Harmonized_IDs:
            print("Wow, {} has MMSE and CDR values!!!".format(scan))
            # index = Harmonized_IDs.index(harmonize_ID_string(scan))
            # FS_metadata_after["MMSE"].append(MMSEs[index])
            # FS_metadata_after["CDR"].append(CDRs[index])

        # save to my driver
        save_path = f"{SAVE_NPZ_PATH}/all_npz/" + sub_scan + "_" + scan.split("_")[2] + '.npz'
        constrcut_npz(save_path, path_of_img, path_of_seg, age, disease_condition)

    # csv_file = "FS_scans_metadata.csv"
    # pd.DataFrame(FS_metadata_after).to_csv(csv_file, index=False)
    # csv_columns = ['ID', 'Path_of_img', 'Path_of_seg', 'HC0/AD1', 'age', 'MMSE', 'CDR']
    # try:
    #     with open(csv_file, 'w') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #         writer.writeheader()
    #         for data in FS_metadata_after:
    #             writer.writerow(data)
    # except IOError:
    #     print("I/O error")

# split dataset
def split_dataset():
    np.random.seed(82018)
    # --- split OASIS-3 into train/dev/test with 80%/5%/15% ---
    df   = pd.read_csv(f"{SAVE_NPZ_PATH}FS_scans_metadata.csv")
    IDs  = list(df["ID"])
    Ages = list(df["age"])
    Disease = list(df["HC0/AD1"])

    print("The max age for OASIS3 is {}".format(max(Ages)))
    def oversample(Ages):
        frequencies = Ages
        frequencies = np.array(frequencies).round()  # quantize (essentially bin)
        ages, counts = np.unique(frequencies, return_counts=True)  # get age hist.
        prob = counts/counts.sum()  # get age probabilities

        # Get inverted probabilities:
        iprob = 1 - prob
        iprob = iprob/iprob.sum()  # renormalize

        ifrequencies = frequencies.copy()
        for i in range(len(ages)):
            idx = np.where(frequencies == ages[i])
            ifrequencies[idx] = iprob[i]

        ifrequencies = ifrequencies/ifrequencies.sum()

        return ifrequencies
    ifrequencies = oversample(Ages)
    train_idx = np.random.choice(len(IDs), round(len(IDs)*0.8),  replace=False, p=ifrequencies)
    train_list = np.array(IDs)[train_idx]

    # # AD is not included in training set
    # IDs_AD = list(df["ID"][df['HC0/AD1']==1])
    # IDs_HC = list(df["ID"][df['HC0/AD1']==0])
    #
    # train_list1 = []
    # train_list2 = []
    # for c, id in enumerate(train_list):
    #     if id not in IDs_AD:
    #         print(c, id)
    #         train_list1.append(id)
    #     if id in IDs_HC:
    #         print(c, id)
    #         train_list2.append(id)

    left_IDs = [id for id in IDs if id not in train_list]
    left_ages = []
    for id in left_IDs:
        left_ages.append(Ages[IDs.index(id)])

    ifrequencies = oversample(left_ages)
    test_idx  = np.random.choice(len(left_IDs), round(len(left_IDs)*0.75), replace=False, p=ifrequencies)
    test_list = np.array(left_IDs)[test_idx]

    valid_list = [id for id in left_IDs if id not in test_list]
    # plot
    print("Split train:{}/valid:{}/test:{} according to ages distribution".format(len(train_list), len(valid_list), len(test_list)))

    def plot_stacked_bar(Ages, Diseases, title):
        x = np.array(Ages).round()
        uni_x = np.unique(x)
        y1 = []
        y2 = []
        for i in uni_x:
            idexes = np.where(x == i)
            dis_i  = np.array(Diseases)[idexes]
            idx_hc = np.where(dis_i == 0)
            idx_ad = np.where(dis_i == 1)
            y1.append(idx_hc[0].shape[0])
            y2.append(idx_ad[0].shape[0])

        plt.rcParams.update({'font.size': 20})

        plt.bar(uni_x, y1, color='orange')
        plt.bar(uni_x, y2, bottom=y1, color='b')
        plt.xlabel("Ages")
        plt.ylabel("Count of scans")
        plt.legend(["HC", "AD"])
        # plt.title(title)
        plt.savefig('./figs/'+title + '.pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    plot_stacked_bar(Ages, Disease, 'OASIS3 All scans (2366 images)')

    def find_age_disease_for_given_subset(subset_list):
        ages = []
        diseases = []
        for id in subset_list:
            ages.append(Ages[IDs.index(id)])
            diseases.append(Disease[IDs.index(id)])
        return ages, diseases

    plot_stacked_bar(*(find_age_disease_for_given_subset(train_list)), 'OASIS3 train scans (1893 images)')
    plot_stacked_bar(*(find_age_disease_for_given_subset(test_list)), 'OASIS3 test scans (355 images)')
    plot_stacked_bar(*(find_age_disease_for_given_subset(valid_list)), 'OASIS3 valid scans (118 images)')

    plot_stacked_bar(*(find_age_disease_for_given_subset(valid_list+list(test_list))), 'OASIS3 test scans (473 images)')

    csv_file = "FS_scans_metadata_with_split.csv"
    def calcu_idx(list):
        idx = []
        for id in list:
            idx.append(IDs.index(id))
        print(len(idx))
        return idx
    partition = np.empty(len(IDs), dtype="S5")
    partition[train_idx] = "train"
    partition[calcu_idx(test_list)] = "test"
    partition[calcu_idx(valid_list)] = "dev"
    partition = [p.decode("utf-8") for p in partition]
    df['Partition'] = partition

# show histogram
def show_histogram_given_partition(csv_path):
    # Load the CSV file into a Pandas DataFrame
    data = pd.read_csv(csv_path)

    # Get unique values of the Partition column
    partitions = data['Partition'].unique()

    # Create a subplot for each unique partition value
    fig, axs = plt.subplots(nrows=len(partitions), ncols=1, figsize=(8, 6), sharex=True, sharey=True)

    # Create a histogram for each group and plot it in the corresponding subplot
    for i, partition in enumerate(partitions):
        group = data.loc[data['Partition'] == partition, 'age']
        counts, bins, patches = axs[i].hist(group, bins=20)
        axs[i].set_title(f'Histogram of Age for {partition} (n={int(sum(counts))})')
        axs[i].set_xlabel('Age')
        axs[i].set_ylabel('Frequency')

    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.4)

    # Show the plot
    plt.show()
