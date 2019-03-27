if __name__ == '__main__':
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from scipy.stats.stats import pearsonr, zscore
    from commontool.io.io import CiftiReader

    hemi = 'rh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    patterns1_file = pjoin(project_dir, 'data/HCP_1080/S1200_1080_MyelinMap_BC_MSMAll_32k_fs_LR.dscalar.nii')
    patterns2_file = pjoin(project_dir, 'data/HCP_1080/S1200_1080_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii')
    mask_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/{}FFA_25.label'.format(hemi[0]))
    corr_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/pattern_corr')
    if not os.path.exists(corr_dir):
        os.makedirs(corr_dir)

    # mask_vertices = nib.freesurfer.read_label(mask_file)[-1::-1]
    mask_vertices = nib.freesurfer.read_label(mask_file)
    patterns1_map = CiftiReader(patterns1_file).get_data(brain_structure[hemi], True)
    patterns2_map = CiftiReader(patterns2_file).get_data(brain_structure[hemi], True)
    # patterns1_map = nib.load(patterns1_file).get_data()
    # patterns2_map = nib.load(patterns2_file).get_data()
    patterns1 = patterns1_map[:, mask_vertices]
    patterns2 = patterns2_map[:, mask_vertices]

    # ---1080---
    # corr_file = pjoin(corr_dir, 'acti_corr_MBS_{}FFA_PA_1080.npy'.format(hemi[0]))
    # corr_arr = np.array([pearsonr(x, y) for x, y in zip(patterns1, patterns2)])
    # np.save(corr_file, corr_arr)
    # ---1080---

    # ---intersubgroup---
    group_labels_file = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/group_labels')

    label_pair = ['1', '2']
    group_labels = np.array(open(group_labels_file).read().split(' '))
    patterns1_g1 = patterns1[group_labels == label_pair[0]]
    patterns1_g2 = patterns1[group_labels == label_pair[1]]
    patterns2_g1 = patterns2[group_labels == label_pair[0]]
    patterns2_g2 = patterns2[group_labels == label_pair[1]]
    patterns1_g1_mean = np.mean(patterns1_g1, 0)
    patterns1_g2_mean = np.mean(patterns1_g2, 0)
    patterns2_g1_mean = np.mean(patterns2_g1, 0)
    patterns2_g2_mean = np.mean(patterns2_g2, 0)
    print(pearsonr(patterns1_g1_mean, patterns1_g2_mean))
    print(pearsonr(patterns2_g1_mean, patterns2_g2_mean))
    print(pearsonr(patterns2_g1_mean, patterns1_g1_mean))
    print(pearsonr(patterns2_g1_mean, patterns1_g2_mean))
    print(pearsonr(patterns2_g2_mean, patterns1_g1_mean))
    print(pearsonr(patterns2_g2_mean, patterns1_g2_mean))

    # corr_file = pjoin(corr_dir, 'g{}_acti_corr_g{}_curv_{}FFA.npy'.format(label_pair[1], label_pair[1], hemi[0]))
    # corr_arr = np.array([pearsonr(x, y) for x in patterns2_g2 for y in patterns1_g2])
    # np.save(corr_file, corr_arr)
    # ---intersubgroup---
