import os
import math
import numpy as np
import nibabel as nib

from os.path import join as pjoin
from scipy import stats
from commontool.io.io import CiftiReader, save2nifti

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
out_dirs = pjoin(project_dir, '{hemi2}{label2}_FFA{roi_label}_connect_{hemi1}{label1}')
trg_regions_lr = dict()

with open(group_labels_file) as rf:
    group_labels = np.array(rf.read().split(' '), dtype=np.uint16)
with open(subject_ids_file) as rf:
    subject_ids = np.array(rf.read().splitlines())
group_labels_uniq = np.unique(group_labels)


def get_trg_regions(file, brain_structure=None):

    if file.endswith('.label.nii'):
        reader = CiftiReader(file)
        data = reader.get_data(brain_structure, True).ravel()
    elif file.endswith('.nii.gz'):
        data = nib.load(file).get_data().ravel()
    else:
        raise RuntimeError('get_trg_regions does not support the file format!')

    return [np.where(data == i)[0] for i in np.unique(data) if i != 0]


def calc_connect(label1, subject, count, sub_subject_num, log_list):
    series_LR_file = series_files.format(subject=subject, phase='LR')
    series_RL_file = series_files.format(subject=subject, phase='RL')
    pass_subject = False
    if not os.path.exists(series_LR_file):
        pass_subject = True
        log_list.append('Path-{} does not exist!'.format(series_LR_file))
    if not os.path.exists(series_RL_file):
        pass_subject = True
        log_list.append('Path-{} dose not exist!'.format(series_RL_file))
    if pass_subject:
        return
    reader_LR = CiftiReader(series_LR_file)
    reader_RL = CiftiReader(series_RL_file)
    for hemi1 in hemis:
        series_LR = reader_LR.get_data(brain_structures[hemi1], True)
        series_RL = reader_RL.get_data(brain_structures[hemi1], True)
        series_LR = stats.zscore(series_LR, 0)
        series_RL = stats.zscore(series_RL, 0)
        series = np.r_[series_LR, series_RL]
        for label2 in group_labels_uniq:
            for hemi2 in hemis:
                roi_file = roi_files.format(hemi2=hemi2[0], label2=label2)
                roi_labels_arr = nib.load(roi_file).get_data().ravel()
                roi_labels_uniq = np.unique(roi_labels_arr).astype(np.uint8)
                for roi_label in roi_labels_uniq:
                    if roi_label == 0:
                        continue
                    seed_region = np.where(roi_labels_arr == roi_label)[0]
                    seed_series = np.mean(series[:, seed_region], 1)
                    if trg_regions_lr:
                        trg_regions = trg_regions_lr[hemi1]
                    else:
                        trg_regions = [[vtx] for vtx in range(series.shape[1])]
                    connections = np.ones(series.shape[1]) * math.nan
                    for trg_region in trg_regions:
                        trg_series = np.mean(series[:, trg_region], 1)
                        connections[trg_region] = stats.pearsonr(seed_series, trg_series)[0]
                        # connect = stats.pearsonr(seed_series, trg_series)[0]
                        # connections[trg_region] = 0 if math.isnan(connect) else connect

                    out_dir = out_dirs.format(hemi2=hemi2[0], label2=label2, roi_label=roi_label,
                                              hemi1=hemi1[0], label1=label1)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    save2nifti(pjoin(out_dir, '{}.nii.gz'.format(subject)), connections)
                    print('group{}_subject{}/{}_{}{}_FFA{}_connect_{}'.format(label1, count, sub_subject_num,
                                                                              hemi2[0], label2, roi_label, hemi1[0]))


def merge_subjects(item_dir):
    item_par, item = os.path.split(item_dir)
    files = sorted(os.listdir(item_dir))
    subjects = [file.split('.')[0] for file in files]
    merged_map = np.atleast_2d(nib.load(pjoin(item_dir, files[0])).get_data())
    for file in files[1:]:
        merged_map = np.r_[merged_map, np.atleast_2d(nib.load(pjoin(item_dir, file)).get_data())]
    save2nifti(pjoin(item_par, '{}_new.nii.gz'.format(item)), merged_map)
    subjects = '\n'.join(subjects)
    open(pjoin(item_par, '{}_subjects'.format(item)), 'w+').writelines(subjects)


if __name__ == '__main__':
    from multiprocessing import Pool

    # ---calculate connectivity start---
    log_list = []
    process_pool = Pool(processes=4)
    for label1 in group_labels_uniq:
        sub_subject_ids = subject_ids[group_labels == label1]
        sub_subject_num = len(sub_subject_ids)
        for count, subject in enumerate(sub_subject_ids, 1):
            process_pool.apply_async(calc_connect, args=(label1, subject, count, sub_subject_num, log_list))
    process_pool.close()
    process_pool.join()
    log_out = '\n'.join(log_list)
    open(pjoin(project_dir, 'tseries_connectivity_log'), 'w+').writelines(log_out)
    # ---calculate connectivity end---

    # ---merge subjects start---
    process_pool = Pool(processes=4)
    for label1 in group_labels_uniq:
        for hemi1 in hemis:
            for label2 in group_labels_uniq:
                for hemi2 in hemis:
                    roi_file = roi_files.format(hemi2=hemi2[0], label2=label2)
                    roi_labels_arr = nib.load(roi_file).get_data().ravel()
                    roi_labels_uniq = np.unique(roi_labels_arr).astype(np.uint8)
                    for roi_label in roi_labels_uniq:
                        if roi_label == 0:
                            continue
                        item_dir = out_dirs.format(hemi2=hemi2[0], label2=label2, roi_label=roi_label,
                                                   hemi1=hemi1[0], label1=label1)
                        process_pool.apply_async(merge_subjects, args=(item_dir,))
    process_pool.close()
    process_pool.join()
    # ---merge subjects end---
