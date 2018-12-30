import os
import numpy as np

from os.path import join as pjoin
from commontool.io.io import CiftiReader, save2nifti


def save_mean_maps(src_file, hemi, cluster_nums, structure_name):

    if hemi == 'lh':
        brain_structure = 'CIFTI_STRUCTURE_CORTEX_LEFT'
    elif hemi == 'rh':
        brain_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    else:
        raise RuntimeError("hemi must be one of ('lh', 'rh')!")

    # prepare data
    reader = CiftiReader(src_file)
    maps = reader.get_data(brain_structure, True)

    for cluster_num in cluster_nums:
        # get clustering labels of subjects
        cluster_num_dir = cluster_num_dirs.format(cluster_num)
        group_labels_path = pjoin(cluster_num_dir, 'group_labels')
        with open(group_labels_path) as rf:
            group_labels = np.array(rf.read().split(' '), dtype=np.uint16)

        mean_maps = np.zeros((0, maps.shape[1]))
        for label in sorted(set(group_labels)):
            subgroup_maps = maps[group_labels == label]
            subgroup_maps_mean = np.atleast_2d(np.mean(subgroup_maps, 0))
            mean_maps = np.r_[mean_maps, subgroup_maps_mean]

        # output
        out_dir = pjoin(cluster_num_dir, 'structure')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        save2nifti(pjoin(out_dir, '{}_{}_mean_maps.nii.gz'.format(hemi, structure_name)), mean_maps)

        print('{}_{}clusters: done'.format(structure_name, cluster_num))


if __name__ == '__main__':
    # predefine some variates
    # -----------------------
    # predefine parameters
    cluster_nums = [2]
    hemi = 'lh'

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    cluster_num_dirs = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/{}clusters')
    # -----------------------

    save_mean_maps(pjoin(project_dir, 'data/HCP_1080/S1200_1080_curvature_MSMAll_32k_fs_LR.dscalar.nii'),
                   hemi, cluster_nums, 'curv')
    # save_mean_maps(pjoin(project_dir, 'data/HCP_1080/S1200_1080_MyelinMap_BC_MSMAll_32k_fs_LR.dscalar.nii'),
    #                hemi, cluster_nums, 'myelin')
    # save_mean_maps(pjoin(project_dir, 'data/HCP_1080/S1200_1080_thickness_MSMAll_32k_fs_LR.dscalar.nii'),
    #                hemi, cluster_nums, 'thickness')
