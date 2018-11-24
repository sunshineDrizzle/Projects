if __name__ == '__main__':
    import os
    import numpy as np

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2cifti

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    curv_file = pjoin(project_dir, 'data/HCP_face-avg/S1200.1080.curvature_MSMAll.32k_fs_LR.dscalar.nii')
    aparc_file = pjoin(project_dir, 'data/HCP_face-avg/S1200_1080_aparc_a2009s_32k_fs_LR.dlabel.nii')
    cluster_num_dir = pjoin(project_dir, '2mm_KM_zscore/3clusters')
    subject_labels_file = pjoin(cluster_num_dir, 'subject_labels')
    mfs_dir = pjoin(cluster_num_dir, 'mfs')
    if not os.path.exists(mfs_dir):
        os.makedirs(mfs_dir)

    curv_reader = CiftiReader(curv_file)
    aparc_reader = CiftiReader(aparc_file)
    sulc_mask = curv_reader.get_data() < 0
    fusiform_mask = np.logical_or(aparc_reader.get_data() == 21, aparc_reader.get_data() == 96)

    with open(subject_labels_file) as rf:
        subject_labels = np.array(rf.read().split(' '), dtype=np.uint16)

    mfs_prob_maps = []
    fusiform_prob_maps = []
    map_names = []
    for label in sorted(set(subject_labels)):
        indices = subject_labels == label
        subgroup_mfs_mask = np.logical_and(sulc_mask[indices], fusiform_mask[indices])
        subgroup_mfs_prob = np.mean(subgroup_mfs_mask, 0)
        subgroup_fusiform_prob = np.mean(fusiform_mask[indices], 0)

        mfs_prob_maps.append(subgroup_mfs_prob)
        fusiform_prob_maps.append(subgroup_fusiform_prob)
        map_names.append('label{}'.format(label))

    save2cifti(pjoin(mfs_dir, 'MFS_prob_maps.dscalar.nii'), np.array(mfs_prob_maps), curv_reader.brain_models(),
               map_names)
    save2cifti(pjoin(mfs_dir, 'fusiform_prob_maps.dscalar.nii'), np.array(fusiform_prob_maps),
               curv_reader.brain_models(), map_names)
