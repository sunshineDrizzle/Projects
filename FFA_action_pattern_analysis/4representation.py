import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import sem
from matplotlib import pyplot as plt


def subgroup_mean_representation(mean_maps, maps, group_labels):

    labels_uniq = np.unique(group_labels)
    representations = np.zeros_like(labels_uniq, np.object)
    for i, label in enumerate(labels_uniq):
        mean_map = mean_maps[[i]]
        sub_maps = np.atleast_2d(maps[group_labels == label])
        representations[i] = 1 - cdist(mean_map, sub_maps, 'correlation')[0]

    representation_means = []
    representation_sems = []
    for repre in representations:
        representation_means.append(np.mean(repre))
        representation_sems.append(sem(repre))
    x = np.arange(len(representations))
    plt.bar(x, representation_means, yerr=representation_sems,
            color='white', edgecolor='black')
    plt.title('{}FFA_patterns'.format(hemi[0]))
    plt.ylabel('correlation')
    plt.xticks(x, labels_uniq)
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
    from commontool.io.io import CiftiReader

    hemi = 'lh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering/'
    analysis_dir = pjoin(project_dir, 's2_25_zscore')
    cluster_num_dir = pjoin(analysis_dir, 'HAC_ward_euclidean/50clusters')
    acti_dir = pjoin(cluster_num_dir, 'activation')

    mean_map_file = pjoin(acti_dir, '{}_mean_maps.nii.gz'.format(hemi))
    FFA_label_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/{}FFA_25.label'.format(hemi[0]))
    map_file = pjoin(project_dir,
                     'data/HCP_1080/face-avg_s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    group_labels_file = pjoin(cluster_num_dir, 'group_labels')

    mean_maps = nib.load(mean_map_file).get_data()
    FFA_vertices = nib.freesurfer.read_label(FFA_label_file)
    reader = CiftiReader(map_file)
    maps = reader.get_data(brain_structure[hemi], True)
    group_labels = np.array(open(group_labels_file).read().split(' '), dtype=np.uint16)

    subgroup_mean_representation(mean_maps[:, FFA_vertices], maps[:, FFA_vertices], group_labels)
