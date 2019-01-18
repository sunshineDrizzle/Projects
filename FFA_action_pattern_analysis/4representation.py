import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt


def subgroup_mean_representation(pattern_mean_maps, FFA_vertices, FFA_patterns, group_labels, metric):

    labels_uniq = np.unique(group_labels)
    labels_num = len(labels_uniq)
    X = np.zeros((labels_num, labels_num), np.object)
    for row, pattern_mean_map in enumerate(np.atleast_2d(pattern_mean_maps)):
        sub_FFA_patterns_mean = np.atleast_2d(pattern_mean_map[FFA_vertices])
        for col, label in enumerate(labels_uniq):
            sub_FFA_patterns = np.atleast_2d(FFA_patterns[group_labels == label])
            X[row, col] = cdist(sub_FFA_patterns_mean, sub_FFA_patterns, metric)[0]

    fig, axes = plt.subplots(labels_num)
    axes[0].set_title('{}FFA_patterns'.format(hemi[0]))
    xlabels = 'mean{} and individual{}'
    for row in range(labels_num):
        print('row{0}col1 vs. row{0}col2'.format(row + 1), ttest_ind(X[row][0], X[row][1]))
        axes[row].violinplot(X[row], showmeans=True)
        axes[row].set_ylabel(metric)
        axes[row].set_xticks(np.arange(1, labels_num + 1))
        axes[row].set_xticklabels([xlabels.format(labels_uniq[row], labels_uniq[col]) for col in range(labels_num)])
    plt.tight_layout()
    plt.show()


def leave_one_out_representation(FFA_patterns, group_labels, metric):

    labels_uniq = np.unique(group_labels)
    labels_num = len(labels_uniq)
    sub_FFA_patterns_list = [np.atleast_2d(FFA_patterns[group_labels == label]) for label in labels_uniq]

    X = np.zeros((labels_num, labels_num), np.object)
    for row in range(labels_num):
        sub_FFA_patterns_mean = np.atleast_2d(np.mean(sub_FFA_patterns_list[row], 0))
        for col in range(labels_num):
            if row == col:
                sub_subjects = list(range(sub_FFA_patterns_list[row].shape[0]))
                dists = []
                for subject in sub_subjects:
                    sub_FFA_patterns_leave_out = np.atleast_2d(sub_FFA_patterns_list[row][subject])
                    sub_subjects_reserve = sub_subjects.copy()
                    sub_subjects_reserve.remove(subject)
                    sub_FFA_patterns_reserve = np.atleast_2d(sub_FFA_patterns_list[row][sub_subjects_reserve])
                    sub_FFA_patterns_reserve_mean = np.atleast_2d(np.mean(sub_FFA_patterns_reserve, 0))
                    dists.append(cdist(sub_FFA_patterns_reserve_mean, sub_FFA_patterns_leave_out, metric)[0][0])
                X[row, col] = np.array(dists)
            else:
                X[row, col] = cdist(sub_FFA_patterns_mean, sub_FFA_patterns_list[col], metric)[0]
    fig, axes = plt.subplots(labels_num)
    axes[0].set_title('{}FFA_patterns'.format(hemi[0]))
    xlabels = 'mean{} and individual{}'
    for row in range(labels_num):
        print('row{0}col1 vs. row{0}col2'.format(row + 1), ttest_ind(X[row][0], X[row][1]))
        axes[row].violinplot(X[row], showmeans=True)
        axes[row].set_ylabel(metric)
        axes[row].set_xticks(np.arange(1, labels_num + 1))
        axes[row].set_xticklabels([xlabels.format(labels_uniq[row], labels_uniq[col]) for col in range(labels_num)])
    plt.tight_layout()
    plt.show()

    return X


if __name__ == '__main__':
    import nibabel as nib

    from os.path import join as pjoin

    hemi = 'rh'
    metric = 'euclidean'
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering/'
    analysis_dir = pjoin(project_dir, 's2_25_zscore')
    cluster_num_dir = pjoin(analysis_dir, 'HAC_ward_euclidean/2clusters')
    acti_dir = pjoin(cluster_num_dir, 'activation')

    # pattern_mean_maps_file = pjoin(acti_dir, '{}_pattern_mean_maps.nii.gz'.format(hemi))
    # FFA_label_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/{}FFA_25.label'.format(hemi[0]))
    # FFA_patterns_file = pjoin(analysis_dir, '{}FFA_patterns.nii.gz'.format(hemi[0]))
    group_labels_file = pjoin(cluster_num_dir, 'group_labels')

    # pattern_mean_maps = nib.load(pattern_mean_maps_file).get_data()
    # FFA_vertices = nib.freesurfer.read_label(FFA_label_file)
    # FFA_patterns = nib.load(FFA_patterns_file).get_data()
    group_labels = np.array(open(group_labels_file).read().split(' '), dtype=np.uint16)

    # subgroup_mean_representation(pattern_mean_maps, FFA_vertices, FFA_patterns, group_labels, metric)
    # leave_one_out_representation(FFA_patterns, group_labels, metric)

    # For other ROIs
    from commontool.io.io import CiftiReader, save2nifti
    from FFA_action_pattern_analysis.tmp_tool.get_roi_pattern import get_roi_pattern

    acti_maps_file = pjoin(project_dir, 'data/HCP_1080/S1200_1080_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii')
    mask_files = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/PAM_z165_p025_ROI_{}.nii.gz')

    if hemi == 'lh':
        acti_maps = CiftiReader(acti_maps_file).get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)
        mask = nib.load(mask_files.format(hemi[0])).get_data().ravel()
    elif hemi == 'rh':
        acti_maps = CiftiReader(acti_maps_file).get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)
        mask = nib.load(mask_files.format(hemi[0])).get_data().ravel()
    else:
        raise RuntimeError("hemi error!")

    ROIs = [np.where(mask == i)[0] for i in np.unique(mask) if i != 0]
    intra_subgroup_dissimilarity = None
    inter_subgroup_dissimilarity = None
    for roi in ROIs:
        patterns = get_roi_pattern(acti_maps, roi, True)
        X = leave_one_out_representation(patterns, group_labels, metric)
        row_num, col_num = X.shape
        intra_tmp = []
        inter_tmp = []
        for row in range(row_num):
            for col in range(col_num):
                if row == col:
                    intra_tmp.extend(X[row, col])
                else:
                    inter_tmp.extend(X[row, col])
        if intra_subgroup_dissimilarity is None:
            intra_subgroup_dissimilarity = np.zeros((len(intra_tmp), acti_maps.shape[1]))
        if inter_subgroup_dissimilarity is None:
            inter_subgroup_dissimilarity = np.zeros((len(intra_tmp), acti_maps.shape[1]))
        intra_subgroup_dissimilarity[:, roi] = np.atleast_2d(np.array(intra_tmp)).T
        inter_subgroup_dissimilarity[:, roi] = np.atleast_2d(np.array(inter_tmp)).T

    repre_dir = pjoin(cluster_num_dir, 'representation')
    save2nifti(pjoin(repre_dir, '{}_intra_subgroup_dissimilarity.nii.gz'.format(hemi)), intra_subgroup_dissimilarity)
    save2nifti(pjoin(repre_dir, '{}_inter_subgroup_dissimilarity.nii.gz'.format(hemi)), inter_subgroup_dissimilarity)
