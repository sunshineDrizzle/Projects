if __name__ == '__main__':
    import os
    import numpy as np

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    # predefine some variates
    # -----------------------
    # predefine parameters
    cluster_nums = range(1, 21)
    hemi = 'rh'

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    cluster_num_dirs = pjoin(project_dir, '2mm_KM_zscore/{}clusters')
    if hemi == 'lh':
        brain_structure = 'CIFTI_STRUCTURE_CORTEX_LEFT'
    elif hemi == 'rh':
        brain_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    else:
        raise RuntimeError("hemi must be one of ('lh', 'rh')!")
    # -----------------------

    # prepare data
    curv_reader = CiftiReader(pjoin(project_dir, 'data/HCP_face-avg/S1200.1080.curvature_MSMAll.32k_fs_LR.dscalar.nii'))
    curv_data = curv_reader.get_data(brain_structure, True)
    sulc_label = curv_data < 0
    curv_names = curv_reader.map_names()

    # calculate subgroup curvature
    for cluster_num in cluster_nums:
        # get clustering labels of subjects
        cluster_num_dir = cluster_num_dirs.format(cluster_num)
        subject_labels_path = pjoin(cluster_num_dir, 'subject_labels')
        with open(subject_labels_path) as rf:
            subject_labels = np.array(rf.read().split(' '), dtype=np.uint16)

        curv_mean_maps = np.zeros((0, curv_data.shape[1]))
        sulc_prob_maps = np.zeros((0, curv_data.shape[1]))
        for label in sorted(set(subject_labels)):
            subgroup_curv_data = curv_data[subject_labels == label]
            subgroup_curv_data_mean = np.atleast_2d(np.mean(subgroup_curv_data, 0))
            subgroup_sulc_prob = np.atleast_2d(np.mean(sulc_label[subject_labels == label], 0))

            curv_mean_maps = np.r_[curv_mean_maps, subgroup_curv_data_mean]
            sulc_prob_maps = np.r_[sulc_prob_maps, subgroup_sulc_prob]

        # output
        curv_dir = pjoin(cluster_num_dir, 'curvature')
        if not os.path.exists(curv_dir):
            os.makedirs(curv_dir)
        save2nifti(pjoin(curv_dir, '{}_curv_mean_maps.nii.gz'.format(hemi)), curv_mean_maps)
        # save2nifti(pjoin(curv_dir, '{}_sulc_prob_maps.nii.gz'.format(hemi)), sulc_prob_maps)

        print('{}clusters: done'.format(cluster_num))
