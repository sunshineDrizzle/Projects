

if __name__ == '__main__':
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from scipy.stats import pearsonr
    from commontool.io.io import CiftiReader, CsvReader

    hemis = ('lh', 'rh')
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    acti_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/activation')
    acti_maps_file = pjoin(project_dir, 'data/HCP_1080/S1200_1080_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii')
    group_labels_file = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/group_labels')
    group_labels = np.array(open(group_labels_file).read().split(' '), dtype=np.uint16)
    group_labels_uniq = np.unique(group_labels)

    # ---prepare ROIs---
    FSR_nonFFA_file_l = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/PAM_z165_p025_ROI_l.nii.gz')
    FSR_nonFFA_file_r = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/PAM_z165_p025_ROI_r.nii.gz')
    FSR_nonFFA_config_l = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/PAM_z165_p025_ROI_labelconfig_l.csv')
    FSR_nonFFA_config_r = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/PAM_z165_p025_ROI_labelconfig_r.csv')
    FFA_file_l = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/lFFA_25.label')
    FFA_file_r = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/rFFA_25.label')
    subFFA_files = pjoin(acti_dir, '{}{}_FFA.nii.gz')
    acti_analysis_dir = pjoin(acti_dir, 'acti_of_FSR')
    if not os.path.exists(acti_analysis_dir):
        os.makedirs(acti_analysis_dir)

    trg_regions_lr = dict()
    label_names_lr = dict()
    FSR_nonFFA_l = nib.load(FSR_nonFFA_file_l).get_data().ravel()
    FSR_nonFFA_r = nib.load(FSR_nonFFA_file_r).get_data().ravel()
    trg_regions_lr['lh'] = [np.where(FSR_nonFFA_l == i)[0] for i in np.unique(FSR_nonFFA_l) if i != 0]
    label_names_lr['lh'] = CsvReader(FSR_nonFFA_config_l).to_dict()['label_name']
    trg_regions_lr['rh'] = [np.where(FSR_nonFFA_r == i)[0] for i in np.unique(FSR_nonFFA_r) if i != 0]
    label_names_lr['rh'] = CsvReader(FSR_nonFFA_config_r).to_dict()['label_name']
    trg_regions_lr['lh'].append(nib.freesurfer.read_label(FFA_file_l))
    label_names_lr['lh'].append('lFFA mask')
    trg_regions_lr['rh'].append(nib.freesurfer.read_label(FFA_file_r))
    label_names_lr['rh'].append('rFFA mask')
    for group_label in group_labels_uniq:
        for hemi in hemis:
            subFFA_file = subFFA_files.format(hemi[0], group_label)
            subFFA_file_name = os.path.basename(subFFA_file)
            subFFA_name = subFFA_file_name.split('.')[0]
            subFFA_data = nib.load(subFFA_file).get_data().ravel()
            trg_regions_lr[hemi].append(np.where(subFFA_data != 0)[0])
            label_names_lr[hemi].append(subFFA_name)
            subFFA_data_uniq = np.unique(subFFA_data).astype(np.uint8)
            for subFFA_label in subFFA_data_uniq:
                if subFFA_label != 0:
                    trg_regions_lr[hemi].append(np.where(subFFA_data == subFFA_label)[0])
                    label_names_lr[hemi].append(subFFA_name + str(subFFA_label))

    reader = CiftiReader(acti_maps_file)

    # label_names = label_names_lr[hemis[0]] + label_names_lr[hemis[1]]
    # open(pjoin(acti_analysis_dir, 'npy_column_name'), 'w+').write(','.join(label_names))
    # ---activation intensity start---
    # for group_label in group_labels_uniq:
    #     indices = np.where(group_labels == group_label)[0]
    #     roi_means_arr = np.zeros((len(indices), 0))
    #     for hemi in hemis:
    #         acti_maps = reader.get_data(brain_structure[hemi], True)
    #         sub_acti_maps = acti_maps[indices]
    #         for region in trg_regions_lr[hemi]:
    #             roi_mean = np.atleast_2d(np.nanmean(sub_acti_maps[:, region], 1)).T
    #             roi_means_arr = np.c_[roi_means_arr, roi_mean]
    #     np.save(pjoin(acti_analysis_dir, 'g{}_intensity.npy'.format(group_label)), roi_means_arr)
    # ---activation intensity end---

    # ---intra subgroup activation pattern similarity start---
    for group_label in group_labels_uniq:
        pattern_similarities_list = []
        for hemi in hemis:
            acti_maps = reader.get_data(brain_structure[hemi], True)
            sub_acti_maps = acti_maps[group_labels == group_label]
            subj_num = sub_acti_maps.shape[0]
            for region in trg_regions_lr[hemi]:
                sub_acti_maps_masked = sub_acti_maps[:, region]
                pattern_similarities = [pearsonr(sub_acti_maps_masked[i], sub_acti_maps_masked[j])[0] for i in range(subj_num-1) for j in range(i+1, subj_num)]
                pattern_similarities_list.append(pattern_similarities)
        out_file = pjoin(acti_analysis_dir, 'g{}_intra_pattern_similarity.npy'.format(group_label))
        np.save(out_file, np.array(pattern_similarities_list).T)
    # ---intra subgroup activation pattern similarity end---

    # ---inter subgroup activation pattern similarity start---
    group_label_pair = (1, 2)
    pattern_similarities_list = []
    for hemi in hemis:
        acti_maps = reader.get_data(brain_structure[hemi], True)
        sub_acti_maps1 = acti_maps[group_labels == group_label_pair[0]]
        sub_acti_maps2 = acti_maps[group_labels == group_label_pair[1]]
        for region in trg_regions_lr[hemi]:
            sub_acti_maps1_masked = sub_acti_maps1[:, region]
            sub_acti_maps2_masked = sub_acti_maps2[:, region]
            pattern_similarities = [pearsonr(x, y)[0] for x in sub_acti_maps1_masked for y in sub_acti_maps2_masked]
            pattern_similarities_list.append(pattern_similarities)
    out_file = pjoin(acti_analysis_dir, 'g{}_and_g{}_inter_pattern_similarity.npy'.format(group_label_pair[0], group_label_pair[1]))
    np.save(out_file, np.array(pattern_similarities_list).T)
    # ---inter subgroup activation pattern similarity end---
