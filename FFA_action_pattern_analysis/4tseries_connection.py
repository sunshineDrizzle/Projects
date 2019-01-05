if __name__ == '__main__':
    import math
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from scipy import stats
    from commontool.io.io import CiftiReader, save2niftil

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering/data/HCP/tseries_test_dir'
    series_files = pjoin(project_dir, '{subject}/tfMRI_WM_{phase}_Atlas_MSMAll.dtseries.nii')
    group_labels_file = pjoin(project_dir, 'group_labels')
    subject_ids_file = pjoin(project_dir, 'subject_id')
    roi_files = pjoin(project_dir, '{hemi2}{label}_FFA.nii.gz')
    out_dir = pjoin(project_dir, 'group{label}_{hemi2}FFA{roilabel}_connect_{hemi1}_new.nii.gz')

    hemis = ('l', 'r')
    brain_structures = ('CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT')

    with open(group_labels_file) as rf:
        group_labels = np.array(rf.read().split(' '), dtype=np.uint16)
    with open(subject_ids_file) as rf:
        subject_ids = np.array(rf.read().splitlines())

    for label in sorted(set(group_labels)):
        sub_subject_ids = subject_ids[group_labels == label]
        sub_subject_num = len(sub_subject_ids)
        sub_connections_dict = dict()
        for count, subject in enumerate(sub_subject_ids, 1):
            series_LR_file = series_files.format(subject=subject, phase='LR')
            series_RL_file = series_files.format(subject=subject, phase='RL')
            reader_LR = CiftiReader(series_LR_file)
            reader_RL = CiftiReader(series_RL_file)
            for hemi_idx1 in range(2):
                series_LR = reader_LR.get_data(brain_structures[hemi_idx1], True)
                series_RL = reader_RL.get_data(brain_structures[hemi_idx1], True)
                series_LR = stats.zscore(series_LR, 0)
                series_RL = stats.zscore(series_RL, 0)
                series = np.r_[series_LR, series_RL]
                for hemi_idx2 in range(2):
                    roi_file = roi_files.format(hemi2=hemis[hemi_idx2], label=label)
                    roi_labels_arr = nib.load(roi_file).get_data().ravel()
                    roi_labels_uniq = np.unique(roi_labels_arr).astype(np.uint8)
                    for roi_label in roi_labels_uniq:
                        if roi_label == 0:
                            continue
                        seed_vertices = np.where(roi_labels_arr == roi_label)[0]
                        seed_series = np.mean(series[:, seed_vertices], 1)
                        trg_vertices_list = [[vtx] for vtx in range(series.shape[1])]
                        connections = np.zeros(series.shape[1])
                        for trg_vertices in trg_vertices_list:
                            trg_series = np.mean(series[:, trg_vertices], 1)
                            connect = stats.pearsonr(seed_series, trg_series)[0]
                            connections[trg_vertices] = 0 if math.isnan(connect) else connect

                        k = (hemi_idx1, hemi_idx2, roi_label)
                        if sub_connections_dict.get(k) is None:
                            sub_connections_dict[k] = [connections]
                        else:
                            sub_connections_dict[k].append(connections)
                        print('group{}_subject{}/{}_{}FFA{}_connect_{}'.format(label, count, sub_subject_num,
                                                                               hemis[k[1]], k[2], hemis[k[0]]))
        for k, v in sub_connections_dict.items():
            save2nifti(out_dir.format(label=label, hemi2=hemis[k[1]], roilabel=k[2], hemi1=hemis[k[0]]), np.mean(v, 0))
