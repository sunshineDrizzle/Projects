

if __name__ == '__main__':
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from scipy.spatial.distance import cdist
    from commontool.io.io import save2nifti

    n_top = 10
    activ_file = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern/' \
                 'analysis/s4_clustering_rh_thr0.5/activation_rh.nii.gz'
    roi_file = '/nfs/t3/workingshop/chenxiayu/study/FFA_pattern/data/HCP/label/MMPprob_OFA_FFA_thr1_rh.label'
    trg_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_pattern/analysis/s4_clustering_rh_thr0.5/top_corr'
    if not os.path.isdir(trg_dir):
        os.makedirs(trg_dir)

    activ = nib.load(activ_file).get_data().squeeze().T
    roi = nib.freesurfer.read_label(roi_file)
    activ_roi = activ[:, roi]

    corrs = 1 - cdist(activ_roi, activ_roi, 'correlation')
    top_corrs_stats = {'mean': [], 'min': [], 'max': []}
    top_activ_means = []
    for row in corrs:
        top_indices = np.argsort(-row)[:n_top]
        top_corrs = row[top_indices]
        top_corrs_stats['mean'].append(str(np.mean(top_corrs)))
        top_corrs_stats['min'].append(str(np.min(top_corrs)))
        top_corrs_stats['max'].append(str(np.max(top_corrs)))
        top_activ_means.append(np.mean(activ[top_indices], 0))

    top_corrs_stats_file = pjoin(trg_dir, f'top{n_top}_corrs_stats.csv')
    with open(top_corrs_stats_file, 'w') as wf:
        text = [','.join(i) for i in zip(*list(top_corrs_stats.values()))]
        text.insert(0, ','.join(top_corrs_stats.keys()))
        wf.write('\n'.join(text))

    top_activ_means = np.array(top_activ_means).T[:, None, None, :]
    top_activ_means_file = pjoin(trg_dir, f'top{n_top}_activ_means.nii.gz')
    save2nifti(top_activ_means_file, top_activ_means)

    top_activ_means_roi = np.ones_like(top_activ_means) * np.min(top_activ_means[roi], 0)
    top_activ_means_roi[roi] = top_activ_means[roi]
    top_activ_means_roi_file = pjoin(trg_dir, f'top{n_top}_activ_means_roi.nii.gz')
    save2nifti(top_activ_means_roi_file, top_activ_means_roi)
