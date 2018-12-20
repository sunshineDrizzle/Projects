if __name__ == '__main__':
    import os
    import numpy as np

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    # predefine some variates
    # -----------------------
    # predefine parameters
    hemi = 'lh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    maps_file = pjoin(project_dir, 'data/HCP_1080/S1200_1080_WM_Mean_BOLD_Signal_MSMAll_32k_fs_LR.dscalar.nii')
    cluster_num_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters')
    subject_labels_file = pjoin(cluster_num_dir, 'subject_labels')
    mean_bold_signal_dir = pjoin(cluster_num_dir, 'mean_bold_signal')
    if not os.path.exists(mean_bold_signal_dir):
        os.makedirs(mean_bold_signal_dir)
    mean_bold_signal_maps_file = pjoin(mean_bold_signal_dir, '{}_mean_bold_signal_maps.nii.gz')
    # -----------------------

    maps = CiftiReader(maps_file).get_data(brain_structure[hemi], True)
    subject_labels = np.array(open(subject_labels_file).read().split(' '), dtype=np.uint16)

    mean_bold_signal_maps = []
    for label in np.unique(subject_labels):
        sub_maps = np.atleast_2d(maps[subject_labels == label])
        mean_bold_signal_maps.append(np.mean(sub_maps, 0))

    save2nifti(mean_bold_signal_maps_file.format(hemi), np.array(mean_bold_signal_maps))
