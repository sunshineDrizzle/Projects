

def calc_reliability():
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from scipy.stats import pearsonr
    from commontool.io.io import CiftiReader

    hemi = 'lh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    data_dir = pjoin(proj_dir, 'data/HCP_1080')
    trg_dir = pjoin(proj_dir, 'analysis/1outliers')
    if not os.path.isdir(trg_dir):
        os.makedirs(trg_dir)

    test_activ_file = pjoin(data_dir, 'face-avg_s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    retest_activ_file = pjoin(data_dir, 'face-avg_s2/retest/S1200_retest_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii')
    test_subject_ids_file = pjoin(data_dir, 'subject_id')
    retest_subject_ids_file = pjoin(data_dir, 'face-avg_s2/retest/subject_id')
    FFA_vertices_file = pjoin(data_dir, f'face-avg_s2/label/{hemi[0]}FFA_25_lr_merge.label')

    test_subject_ids = open(test_subject_ids_file).read().splitlines()
    retest_subject_ids = open(retest_subject_ids_file).read().splitlines()
    test_indices = []
    for sid in retest_subject_ids:
        test_indices.append(test_subject_ids.index(sid))
    FFA_vertices = nib.freesurfer.read_label(FFA_vertices_file)

    test_reader = CiftiReader(test_activ_file)
    retest_reader = CiftiReader(retest_activ_file)
    test_activ_FFA = test_reader.get_data(brain_structure[hemi], True)[test_indices][:, FFA_vertices]
    retest_activ_FFA = retest_reader.get_data(brain_structure[hemi], True)[:, FFA_vertices]

    corrs = [pearsonr(i, j)[0] for i, j in zip(test_activ_FFA, retest_activ_FFA)]
    np.save(pjoin(trg_dir, f'{hemi[0]}FFA_reliability.npy'), np.array(corrs))


def plot_reliability():
    import numpy as np

    from os.path import join as pjoin
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import show_bar_value

    trg_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis/1outliers'

    rFFA_reliability = np.load(pjoin(trg_dir, 'rFFA_reliability.npy'))
    lFFA_reliability = np.load(pjoin(trg_dir, 'lFFA_reliability.npy'))
    FFA_reliability = np.r_[rFFA_reliability, lFFA_reliability]
    bins = np.linspace(FFA_reliability.min(), FFA_reliability.max(), 10)
    plt.figure(figsize=(6, 4))
    _, _, patches = plt.hist(FFA_reliability, bins, color='white', edgecolor='black')
    plt.xlabel('correlation')
    plt.title('FFA pattern reliability')
    show_bar_value(patches, '.0f')

    plt.tight_layout()
    plt.show()


def calc_corr_lr(out_name=None):
    """
    Calculate correlations between lFFA and rFFA activation patterns for 1080 subjects

    Parameter:
    ---------
    out_name[str]: output file name

    Return:
    ------
    corr_lr[list]: 1080 correlations
    """
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from scipy.stats.stats import pearsonr
    from commontool.io.io import CiftiReader

    # prepare path
    proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
    data_dir = pjoin(proj_dir, 'data/HCP_1080/face-avg_s2')
    trg_dir = pjoin(proj_dir, 'analysis/1outliers')
    if not os.path.isdir(trg_dir):
        os.makedirs(trg_dir)

    # load activation
    activ_file = pjoin(data_dir, 'S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    reader = CiftiReader(activ_file)
    activ_l = reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)
    activ_r = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)

    # get FFA activation
    lFFA_vertices = nib.freesurfer.read_label(pjoin(data_dir, 'label/lFFA_25_lr_merge.label'))
    rFFA_vertices = nib.freesurfer.read_label(pjoin(data_dir, 'label/rFFA_25_lr_merge.label'))
    activ_lFFA = activ_l[:, lFFA_vertices]
    activ_rFFA = activ_r[:, rFFA_vertices]

    # calculate L-R correlation
    corr_lr = [pearsonr(l, r)[0] for l, r in zip(activ_lFFA, activ_rFFA)]

    if out_name is not None:
        np.save(pjoin(trg_dir, out_name), np.array(corr_lr))
    return corr_lr


def plot_corr_lr(out_name=None):
    """
    Plot correlations between lFFA and rFFA activation patterns for 1080 subjects

    Parameter:
    ---------
    out_name[str]: output figure name
    """
    import numpy as np

    from os.path import join as pjoin
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import show_bar_value

    trg_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis/1outliers'

    corr_lr = np.load(pjoin(trg_dir, '1080_lr_corr.npy'))
    bins = np.linspace(corr_lr.min(), corr_lr.max(), 30)
    plt.figure(figsize=(6, 4))
    _, _, patches = plt.hist(corr_lr, bins, color='white', edgecolor='black')
    plt.title('histogram of FFA pattern corr between lh and rh')
    plt.xlabel('correlation')
    show_bar_value(patches, '.0f')

    plt.tight_layout()
    if out_name is not None:
        plt.savefig(pjoin(trg_dir, out_name))
    plt.show()


def calc_pattern_mds(n_component, out_name=None):
    """
    Calculate distribution of 1080 subjects' FFA activation patterns

    Parameter:
    ---------
    n_component[int]: MDS components
    out_name[str]: output result file

    Return:
    ------
    X[ndarray]: 1080 x n_components
    """
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from sklearn.manifold import MDS
    from commontool.io.io import CiftiReader

    # prepare
    hemi = 'rh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    proj_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern'
    data_dir = pjoin(proj_dir, 'data/HCP_1080/face-avg_s2')
    trg_dir = pjoin(proj_dir, 'analysis/1outliers')
    if not os.path.isdir(trg_dir):
        os.makedirs(trg_dir)

    # load activation
    activ_file = pjoin(data_dir, 'S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    reader = CiftiReader(activ_file)
    activ = reader.get_data(brain_structure[hemi], True)

    # get FFA activation
    FFA_vertices = nib.freesurfer.read_label(pjoin(data_dir, f'label/{hemi[0]}FFA_25_lr_merge.label'))
    activ_FFA = activ[:, FFA_vertices]

    # calculate MDS
    embedding = MDS(n_component)
    X = embedding.fit_transform(activ_FFA)

    if out_name is not None:
        np.save(pjoin(trg_dir, out_name), X)
    return X


def plot_pattern_mds():
    import numpy as np

    from os.path import join as pjoin
    from matplotlib import pyplot as plt

    trg_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis/1outliers'
    X = np.load(pjoin(trg_dir, 'rFFA_pattern_mds2.npy'))
    plt.scatter(X[:, 0], X[:, 1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # calc_reliability()
    # plot_reliability()
    # calc_corr_lr(out_name='1080_lr_corr.npy')
    # plot_corr_lr(out_name='1080_lr_corr_hist.jpg')
    calc_pattern_mds(2, 'rFFA_pattern_mds2.npy')
    plot_pattern_mds()
