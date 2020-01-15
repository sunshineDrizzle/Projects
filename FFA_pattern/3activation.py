

def calc_subgroup_mean_representation():
    import os
    import numpy as np
    import nibabel as nib
    import pickle as pkl

    from os.path import join as pjoin
    from scipy.spatial.distance import cdist

    hemi = 'rh'
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    clustering_dir = pjoin(proj_dir, 'analysis/s4_clustering_rh_thr0.5')
    n_cluster_dir = pjoin(clustering_dir, 'raw/HAC_average_correlation/100clusters')
    repre_dir = pjoin(n_cluster_dir, 'activation/representation')
    if not os.path.exists(repre_dir):
        os.makedirs(repre_dir)

    roi_file = pjoin(proj_dir, 'data/HCP/label/MMPprob_OFA_FFA_thr1_{}.label'.format(hemi))
    activ_file = pjoin(clustering_dir, 'activation_rh.nii.gz')
    group_labels_file = pjoin(n_cluster_dir, 'group_labels')

    roi = nib.freesurfer.read_label(roi_file)
    activ = nib.load(activ_file).get_data().squeeze().T
    group_labels = np.array(open(group_labels_file).read().split(' '), dtype=np.uint16)
    roi_activ = activ[:, roi]

    labels_uniq = np.unique(group_labels)
    representations = dict()
    for label in labels_uniq:
        sub_roi_activ = np.atleast_2d(roi_activ[group_labels == label])
        sub_roi_mean = np.mean(sub_roi_activ, 0)[None, :]
        representations[str(label)] = 1 - cdist(sub_roi_mean, sub_roi_activ, 'correlation')[0]

    pkl.dump(representations, open(pjoin(repre_dir, f'representation_{hemi}.pkl'), 'wb'))


def plot_representation():
    import numpy as np
    import pickle as pkl

    from scipy.stats import sem
    from matplotlib import pyplot as plt

    fname = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern/analysis/s4_clustering_rh_thr0.5/' \
            'raw/HAC_average_correlation/100clusters/activation/representation/representation_rh.pkl'
    representations = pkl.load(open(fname, 'rb'))
    representation_means = []
    representation_sems = []
    for repre in representations.values():
        representation_means.append(np.mean(repre))
        representation_sems.append(sem(repre))
    labels = list(representations.keys())
    x = np.arange(len(labels))
    plt.bar(x, representation_means, yerr=representation_sems,
            color='white', edgecolor='black')
    plt.title('representation')
    plt.ylabel('correlation')
    plt.xticks(x, labels)
    plt.setp(plt.gca().get_xticklabels(), rotation=-90, ha='left', rotation_mode='anchor')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    calc_subgroup_mean_representation()
    plot_representation()
