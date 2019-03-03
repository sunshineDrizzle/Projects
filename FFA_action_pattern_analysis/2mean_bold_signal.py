import numpy as np

from commontool.io.io import save2nifti


def save2mean_map(src_maps, group_labels, output_name):
    mean_maps = []
    for label in np.unique(group_labels):
        sub_maps = np.atleast_2d(src_maps[group_labels == label])
        mean_maps.append(np.mean(sub_maps, 0))
    save2nifti(output_name, np.array(mean_maps))


if __name__ == '__main__':
    import os
    import nibabel as nib

    from os.path import join as pjoin
    from statsmodels.stats.multitest import multipletests
    from commontool.io.io import CiftiReader, CsvReader
    from commontool.algorithm.statistics import ttest_ind_pairwise

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
    group_labels_file = pjoin(cluster_num_dir, 'group_labels')
    mean_bold_signal_dir = pjoin(cluster_num_dir, 'mean_bold_signal')
    if not os.path.exists(mean_bold_signal_dir):
        os.makedirs(mean_bold_signal_dir)
    # -----------------------

    maps = CiftiReader(maps_file).get_data(brain_structure[hemi], True)
    group_labels = np.array(open(group_labels_file).read().split(' '), dtype=np.uint16)

    # ---mean_map start---
    # save2mean_map(maps, group_labels,
    #               pjoin(mean_bold_signal_dir, '{}_mean_maps.nii.gz'.format(hemi)))
    # ---mean_map end---

    compare_dir = pjoin(mean_bold_signal_dir, 'compare')
    if not os.path.exists(compare_dir):
        os.makedirs(compare_dir)
    # ---compare start---
    pair_labels = [1, 2]
    samples1 = maps[group_labels == pair_labels[0]].T
    samples2 = maps[group_labels == pair_labels[1]].T
    sample_names = list(map(str, range(maps.shape[1])))
    compare_file = pjoin(compare_dir, '{}_g{}_vs_g{}'.format(hemi, pair_labels[0], pair_labels[1]))
    ttest_ind_pairwise(samples1, samples2, compare_file, sample_names)
    # ---compare end---

    # ---compare2nifti start---
    compare_file = pjoin(compare_dir, '{}_g1_vs_g2'.format(hemi))
    mask_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/{}_posterior_brain_mask.label'.format(hemi))

    compare_dict = CsvReader(compare_file).to_dict(1)
    valid_idx_mat = np.array(compare_dict['p']) != 'nan'
    if mask_file is not None:
        mask_vertices = nib.freesurfer.read_label(mask_file)
        mask_idx_mat = np.zeros_like(valid_idx_mat, dtype=np.bool)
        mask_idx_mat[mask_vertices] = True
        valid_idx_mat = np.logical_and(valid_idx_mat, mask_idx_mat)

    compare_data = np.zeros((3, maps.shape[1]))
    ps_uncorrected = np.array([float(p) for idx, p in enumerate(compare_dict['p']) if valid_idx_mat[idx]])
    reject, ps_corrected, alpha_sidak, alpha_bonf = multipletests(ps_uncorrected, 0.05, 'fdr_bh')
    ts = [float(t) for idx, t in enumerate(compare_dict['t']) if valid_idx_mat[idx]]
    compare_data[0, valid_idx_mat] = ts
    compare_data[1, valid_idx_mat] = -ps_uncorrected
    compare_data[2, valid_idx_mat] = -ps_corrected
    compare_data[0, np.logical_not(valid_idx_mat)] = np.min(ts)
    compare_data[1, np.logical_not(valid_idx_mat)] = np.min(-ps_uncorrected)
    compare_data[2, np.logical_not(valid_idx_mat)] = np.min(-ps_corrected)
    save2nifti(pjoin(compare_dir, '{}_g1_vs_g2_posterior_masked.nii.gz'.format(hemi)), compare_data)
    # ---compare2nifti end---
