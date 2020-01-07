

def calc_reliability():
    """
    Calculate FFA activation reliability
    """
    import os
    import nibabel as nib
    import pickle as pkl

    from os.path import join as pjoin
    from scipy.stats import pearsonr
    from commontool.io.io import CiftiReader

    hemi = 'rh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    data_dir = pjoin(proj_dir, 'data/HCP')
    trg_dir = pjoin(proj_dir, 'analysis/reliability')
    if not os.path.isdir(trg_dir):
        os.makedirs(trg_dir)

    test_activ_file = pjoin(data_dir, 'wm/analysis_s2/S1200_1080_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii')
    retest_activ_file = pjoin(data_dir, f'emotion/analysis_s2/cope3_face-shape_zstat_{hemi}.nii.gz')
    test_subject_ids_file = pjoin(data_dir, 'wm/analysis_s2/subject_id')
    retest_subject_ids_file = pjoin(data_dir, 'emotion/analysis_s2/subject_id')
    mask_vertices_file = pjoin(data_dir, f'label/MMPprob_OFA_FFA_thr1_{hemi}.label')

    # get subjects
    test_subject_ids = open(test_subject_ids_file).read().splitlines()
    retest_subject_ids = open(retest_subject_ids_file).read().splitlines()
    subj_ids = sorted(set(test_subject_ids).intersection(retest_subject_ids))
    test_indices = []
    retest_indices = []
    for sid in subj_ids:
        test_indices.append(test_subject_ids.index(sid))
        retest_indices.append(retest_subject_ids.index(sid))

    # get activation
    mask_vertices = nib.freesurfer.read_label(mask_vertices_file)
    test_reader = CiftiReader(test_activ_file)
    test_activ_mask = test_reader.get_data(brain_structure[hemi], True)[test_indices][:, mask_vertices]
    retest_activ_mask = nib.load(retest_activ_file).get_data().squeeze().T
    retest_activ_mask = retest_activ_mask[retest_indices][:, mask_vertices]

    corrs = [pearsonr(i, j)[0] for i, j in zip(test_activ_mask, retest_activ_mask)]
    reliability_dict = {'sid': subj_ids, 'corr': corrs}
    out_path = pjoin(trg_dir, f'reliability_{hemi}.pkl')
    pkl.dump(reliability_dict, open(out_path, 'wb'))


def plot_reliability():
    import numpy as np
    import pickle as pkl

    from os.path import join as pjoin
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import show_bar_value

    hemi = 'rh'  # lh, rh, or both
    thr = 0.5
    trg_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis/reliability'

    if hemi == 'both':
        reliability_lh = pkl.load(open(pjoin(trg_dir, 'reliability_lh.pkl'), 'rb'))
        reliability_rh = pkl.load(open(pjoin(trg_dir, 'reliability_rh.pkl'), 'rb'))
        corrs = reliability_lh['corr'] + reliability_rh['corr']
    else:
        reliability = pkl.load(open(pjoin(trg_dir, f'reliability_{hemi}.pkl'), 'rb'))
        corrs = reliability['corr']

    if thr is not None:
        corrs = [i for i in corrs if i > 0.5]

    print('#correlation:', len(corrs))

    bins = np.linspace(min(corrs), max(corrs), 30)
    # plt.figure(figsize=(9, 4))
    _, _, patches = plt.hist(corrs, bins, color='white', edgecolor='black')
    plt.xlabel('correlation')
    plt.title(f'{hemi} activation pattern reliability')
    show_bar_value(patches, '.0f')

    plt.tight_layout()
    plt.show()


def select_subject():
    import os
    import numpy as np
    import pickle as pkl

    from os.path import join as pjoin

    hemi = 'rh'
    thr = 0.5
    anal_dir = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/analysis'
    src_dir = pjoin(anal_dir, 'reliability')
    trg_dir = pjoin(anal_dir, f'clustering_{hemi}')
    if not os.path.isdir(trg_dir):
        os.makedirs(trg_dir)

    reliability = pkl.load(open(pjoin(src_dir, f'reliability_{hemi}.pkl'), 'rb'))
    corrs = np.array(reliability['corr'])
    subj_ids = np.array(reliability['sid'])

    subj_ids_selected = subj_ids[corrs > thr]
    with open(pjoin(trg_dir, 'subject_id'), 'w') as wf:
        wf.write('\n'.join(subj_ids_selected))


def merge_activation():
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    hemi = 'rh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    proj_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern'
    data_dir = pjoin(proj_dir, 'data/HCP')
    trg_dir = pjoin(proj_dir, f'analysis/clustering_{hemi}')
    subj_id_file = pjoin(trg_dir, 'subject_id')
    subj_id_wm_file = pjoin(data_dir, 'wm/analysis_s2/subject_id')
    subj_id_emotion_file = pjoin(data_dir, 'emotion/analysis_s2/subject_id')
    activ_wm_file = pjoin(data_dir, 'wm/analysis_s2/S1200_1080_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii')
    activ_emotion_file = pjoin(data_dir, f'emotion/analysis_s2/cope3_face-shape_zstat_{hemi}.nii.gz')

    subj_ids = open(subj_id_file).read().splitlines()
    subj_ids_wm = open(subj_id_wm_file).read().splitlines()
    subj_ids_emotion = open(subj_id_emotion_file).read().splitlines()
    activ_wm = CiftiReader(activ_wm_file).get_data(brain_structure[hemi], True)
    activ_emotion = nib.load(activ_emotion_file).get_data().squeeze().T

    wm_indices = []
    emotion_indices = []
    for i in subj_ids:
        wm_indices.append(subj_ids_wm.index(i))
        emotion_indices.append(subj_ids_emotion.index(i))

    activ = (activ_wm[wm_indices] + activ_emotion[emotion_indices]) / 2
    activ = activ.T[:, None, None, :]
    save2nifti(pjoin(trg_dir, f'activation_{hemi}.nii.gz'), activ)


if __name__ == '__main__':
    # calc_reliability()
    # plot_reliability()
    # select_subject()
    merge_activation()
