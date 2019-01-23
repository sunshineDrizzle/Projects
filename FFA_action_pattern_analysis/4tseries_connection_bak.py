import os
import math
import numpy as np
import nibabel as nib

from scipy import stats
from commontool.io.io import CiftiReader


def calc_connect(series_files, subject, hemis, brain_structures, group_labels_uniq, roi_files,
                 label1, count, sub_subject_num):

    # check if the file exists
    series_LR_file = series_files.format(subject=subject, phase='LR')
    series_RL_file = series_files.format(subject=subject, phase='RL')
    pass_subject = False
    log_list = []
    if not os.path.exists(series_LR_file):
        pass_subject = True
        log_list.append('Path-{} does not exist!'.format(series_LR_file))
    if not os.path.exists(series_RL_file):
        pass_subject = True
        log_list.append('Path-{} dose not exist!'.format(series_RL_file))
    if pass_subject:
        return log_list

    result = dict()
    reader_LR = CiftiReader(series_LR_file)
    reader_RL = CiftiReader(series_RL_file)
    for hemi1 in hemis:
        series_LR, map_shape_LR, idx2vtx_LR = reader_LR.get_data(brain_structures[hemi1])
        series_RL, map_shape_RL, idx2vtx_RL = reader_RL.get_data(brain_structures[hemi1])
        assert map_shape_LR == map_shape_RL and idx2vtx_LR == idx2vtx_RL
        series_LR = stats.zscore(series_LR, 0)
        series_RL = stats.zscore(series_RL, 0)
        series = np.r_[series_LR, series_RL]
        vtx2idx = np.ones(map_shape_LR, dtype=np.uint16) * -1
        vtx2idx[idx2vtx_LR] = np.arange(len(idx2vtx_LR))
        for label2 in group_labels_uniq:
            for hemi2 in hemis:
                roi_file = roi_files.format(hemi2=hemi2[0], label2=label2)
                roi_labels_arr = nib.load(roi_file).get_data().ravel()
                roi_labels_uniq = np.unique(roi_labels_arr).astype(np.uint8)
                for roi_label in roi_labels_uniq:
                    if roi_label == 0:
                        continue
                    seed_vertices = np.where(roi_labels_arr == roi_label)[0]
                    seed_series = np.mean(series[:, vtx2idx[seed_vertices]], 1)
                    connections = np.ones_like(vtx2idx, np.float64) * math.nan
                    for idx in range(series.shape[1]):
                        trg_series = series[:, idx]
                        connections[idx2vtx_LR[idx]] = stats.pearsonr(seed_series, trg_series)[0]

                    result[(hemi2[0], label2, roi_label, hemi1[0], label1)] = connections
                    result[('subject', label1)] = subject
                    print('group{}_subject{}/{}_{}{}_FFA{}_connect_{}'.format(label1, count, sub_subject_num,
                                                                              hemi2[0], label2, roi_label, hemi1[0]))
    return result


if __name__ == '__main__':
    from os.path import join as pjoin
    from multiprocessing import Pool
    from commontool.io.io import save2nifti

    hemis = ('lh', 'rh')
    brain_structures = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering/data/HCP/tseries_test_dir'
    series_files = pjoin(project_dir, '{subject}/tfMRI_WM_{phase}_Atlas_MSMAll.dtseries.nii')
    group_labels_file = pjoin(project_dir, 'group_labels')
    subject_ids_file = pjoin(project_dir, 'subject_id')
    roi_files = pjoin(project_dir, '{hemi2}{label2}_FFA.nii.gz')
    out_files = pjoin(project_dir, '{hemi2}{label2}_FFA{roi_label}_connect_{hemi1}{label1}.nii.gz')
    trg_regions_lr = dict()

    with open(group_labels_file) as rf:
        group_labels = np.array(rf.read().split(' '), dtype=np.uint16)
    with open(subject_ids_file) as rf:
        subject_ids = np.array(rf.read().splitlines())
    group_labels_uniq = np.unique(group_labels)

    # ---calculate connectivity start---
    results = []
    process_pool = Pool(processes=4)
    for label1 in group_labels_uniq:
        sub_subject_ids = subject_ids[group_labels == label1]
        sub_subject_num = len(sub_subject_ids)
        for count, subject in enumerate(sub_subject_ids, 1):
            args = (series_files, subject, hemis, brain_structures, group_labels_uniq, roi_files,
                    label1, count, sub_subject_num)
            results.append(process_pool.apply_async(calc_connect, args=args))
    process_pool.close()
    process_pool.join()
    # ---calculate connectivity end---

    log_list = []
    connect_dict = dict()
    for result in results:
        value = result.get()
        if isinstance(value, list):
            log_list.extend(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                if connect_dict.get(k) is None:
                    connect_dict[k] = [v]
                else:
                    connect_dict[k].append(v)
        else:
            raise RuntimeError("invalid return")
    for k, v in connect_dict.items():
        if 'subject' in k:
            subject_out = '\n'.join(v)
            open(pjoin(project_dir, 'subject{}_id'.format(k[1])), 'w+').writelines(subject_out)
        else:
            out_file = out_files.format(hemi2=k[0], label2=k[1], roi_label=k[2], hemi1=k[3], label1=k[4])
            save2nifti(out_file, np.array(v))
    log_out = '\n'.join(log_list)
    open(pjoin(project_dir, 'tseries_connectivity_log'), 'w+').writelines(log_out)
