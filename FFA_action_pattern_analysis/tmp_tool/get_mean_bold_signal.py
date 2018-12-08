if __name__ == '__main__':
    import numpy as np

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2cifti

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    test_par = pjoin(project_dir, 'data/HCP/tseries_test_dir')
    series_LR_files = pjoin(test_par, '{subject}/tfMRI_WM_LR_Atlas_MSMAll.dtseries.nii')
    series_RL_files = pjoin(test_par, '{subject}/tfMRI_WM_RL_Atlas_MSMAll.dtseries.nii')
    mean_signal_maps_out = pjoin(test_par, 'tfMRI_WM_Mean_BOLD_Signal_MSMAll_new.dscalar.nii')

    subject_ids = open(pjoin(test_par, 'subject_id')).read().splitlines()

    mean_signal_maps = []
    for subject in subject_ids:
        series_LR_file = series_LR_files.format(subject=subject)
        series_RL_file = series_RL_files.format(subject=subject)
        reader_LR = CiftiReader(series_LR_file)
        reader_RL = CiftiReader(series_RL_file)
        series_LR = reader_LR.get_data()
        series_RL = reader_RL.get_data()
        series = np.r_[series_LR, series_RL]
        mean_signal = np.mean(series, 0)
        mean_signal_maps.append(mean_signal)

        print('Finish:', subject)

    save2cifti(mean_signal_maps_out, np.array(mean_signal_maps), reader_RL.brain_models(), subject_ids)
