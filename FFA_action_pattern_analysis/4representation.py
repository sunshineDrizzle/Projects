if __name__ == '__main__':
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from scipy.spatial.distance import cdist
    from scipy.stats import ttest_ind
    from matplotlib import pyplot as plt

    hemi = 'rh'
    metric = 'euclidean'
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering/'
    analysis_dir = pjoin(project_dir, 's2_25_zscore')
    cluster_num_dir = pjoin(analysis_dir, 'HAC_ward_euclidean/2clusters')
    acti_dir = pjoin(cluster_num_dir, 'activation')

    pattern_mean_maps_file = pjoin(acti_dir, '{}_pattern_mean_maps.nii.gz'.format(hemi))
    FFA_label_file = pjoin(project_dir, 'data/HCP_1080/face-avg_s2/label/{}FFA_25.label'.format(hemi[0]))
    FFA_patterns_file = pjoin(analysis_dir, '{}FFA_patterns.nii.gz'.format(hemi[0]))
    subject_labels_file = pjoin(cluster_num_dir, 'subject_labels')

    pattern_mean_maps = nib.load(pattern_mean_maps_file).get_data()
    FFA_vertices = nib.freesurfer.read_label(FFA_label_file)
    FFA_patterns = nib.load(FFA_patterns_file).get_data()
    subject_labels = np.array(open(subject_labels_file).read().split(' '), dtype=np.uint16)

    labels_uniq = np.unique(subject_labels)
    labels_num = len(labels_uniq)
    X = np.zeros((labels_num, labels_num), np.object)
    for row, pattern_mean_map in enumerate(np.atleast_2d(pattern_mean_maps)):
        sub_FFA_patterns_mean = np.atleast_2d(pattern_mean_map[FFA_vertices])
        for col, label in enumerate(labels_uniq):
            sub_FFA_patterns = np.atleast_2d(FFA_patterns[subject_labels == label])
            X[row, col] = cdist(sub_FFA_patterns_mean, sub_FFA_patterns, metric)[0]

    fig, axes = plt.subplots(labels_num)
    axes[0].set_title('{}FFA_patterns'.format(hemi[0]))
    xlabels = 'mean{} and individual{}'
    for row in range(labels_num):
        print('row{0}col1 vs. row{0}col2'.format(row+1), ttest_ind(X[row][0], X[row][1]))
        axes[row].violinplot(X[row], showmeans=True)
        axes[row].set_ylabel(metric)
        axes[row].set_xticks(np.arange(1, labels_num+1))
        axes[row].set_xticklabels([xlabels.format(labels_uniq[row], labels_uniq[col]) for col in range(labels_num)])

    plt.tight_layout()
    plt.show()
