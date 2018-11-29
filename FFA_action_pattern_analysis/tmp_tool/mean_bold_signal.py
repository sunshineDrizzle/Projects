if __name__ == '__main__':
    import numpy as np

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2cifti

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    FSR_maps_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    curv_maps_file = pjoin(project_dir, 'data/HCP_face-avg/S1200.1080.curvature_MSMAll.32k_fs_LR.dscalar.nii')
    test_par = pjoin(project_dir, 'data/HCP/tseries_test_dir')
    series_LR_files = pjoin(test_par, '{subject}/tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii')
    series_RL_files = pjoin(test_par, '{subject}/tfMRI_WM_RL_Atlas_MSMAll.dtseries.nii')
    mean_signal_maps_out = pjoin(test_par, 'tfMRI_WM_Mean_BOLD_Signal_MSMAll.dscalar.nii')
    FSR_maps_test_out = pjoin(test_par, 'FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    curv_maps_test_out = pjoin(test_par, 'curvature_MSMAll_32k_fs_LR.dscalar.nii')

    subject_ids = open(pjoin(project_dir, 'data/HCP_face-avg/s2/subject_id')).read().splitlines()
    subject_ids_test = open(pjoin(test_par, 'subject_id')).read().splitlines()
    reader_FSR = CiftiReader(FSR_maps_file)
    FSR_maps = reader_FSR.get_data()
    reader_curv = CiftiReader(curv_maps_file)
    curv_maps = reader_curv.get_data()

    mean_signal_maps = []
    FSR_maps_test = []
    curv_maps_test = []
    for subject in subject_ids_test:
        series_LR_file = series_LR_files.format(subject=subject)
        series_RL_file = series_RL_files.format(subject=subject)
        reader_LR = CiftiReader(series_LR_file)
        reader_RL = CiftiReader(series_RL_file)
        series_LR = reader_LR.get_data()
        series_RL = reader_RL.get_data()
        series = np.r_[series_LR, series_RL]
        mean_signal = np.mean(series, 0)
        mean_signal_maps.append(mean_signal)

        subject_idx = subject_ids.index(subject)
        FSR_map = FSR_maps[subject_idx]
        FSR_maps_test.append(FSR_map)
        curv_map = curv_maps[subject_idx]
        curv_maps_test.append(curv_map)

        print('Finish:', subject)

    save2cifti(mean_signal_maps_out, np.array(mean_signal_maps), reader_RL.brain_models(), subject_ids_test)
    save2cifti(FSR_maps_test_out, np.array(FSR_maps_test), reader_FSR.brain_models(), subject_ids_test)
    save2cifti(curv_maps_test_out, np.array(curv_maps_test), reader_curv.brain_models(), subject_ids_test)
