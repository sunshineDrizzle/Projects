

if __name__ == '__main__':
    import numpy as np

    from os.path import join as pjoin
    from scipy import stats
    from commontool.io.io import CsvReader

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    connect_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/rfMRI_connectivity/PAM_z165_p025_ROI')
    subject_ids_file = pjoin(connect_dir, 'subject_id_4run_1200')
    tseries_LR_file1 = pjoin(connect_dir, '{subject}/rfMRI_REST2_LR.npy')
    tseries_RL_file1 = pjoin(connect_dir, '{subject}/rfMRI_REST2_RL.npy')
    npy_info_file1 = pjoin(connect_dir, 'npy_info')
    tseries_LR_file2 = pjoin(connect_dir, 'addition/{subject}/rfMRI_REST2_LR.npy')
    tseries_RL_file2 = pjoin(connect_dir, 'addition/{subject}/rfMRI_REST2_RL.npy')
    npy_info_file2 = pjoin(connect_dir, 'addition/npy_info')
    out_conn_file = pjoin(connect_dir, 'connectivity_sess2.npy')
    out_info_file = pjoin(connect_dir, 'connectivity_info')

    subject_ids = np.array(open(subject_ids_file).read().splitlines())

    r_names1 = CsvReader(npy_info_file1).to_dict(1)['region_name']
    invalid_rois_of_1 = ['l2_FFA1', 'r2_FFA1']
    invalid_rows_of_1 = [r_names1.index(roi) for roi in invalid_rois_of_1]
    for roi in invalid_rois_of_1:
        r_names1.remove(roi)

    r_names2 = CsvReader(npy_info_file2).to_dict(1)['region_name']
    new_rois_of_2 = ['l2_FFA1', 'l2_FFA2', 'r2_FFA1', 'r2_FFA2']
    new_rows_of_2 = [r_names2.index(roi) for roi in new_rois_of_2]

    r_names = r_names1 + new_rois_of_2

    connectivity = []
    for subject in subject_ids:
        tseries_LR1 = np.load(tseries_LR_file1.format(subject=subject))
        tseries_RL1 = np.load(tseries_RL_file1.format(subject=subject))
        tseries_LR2 = np.load(tseries_LR_file2.format(subject=subject))
        tseries_RL2 = np.load(tseries_RL_file2.format(subject=subject))
        tseries_LR = np.r_[np.delete(tseries_LR1, invalid_rows_of_1, axis=0), tseries_LR2[new_rows_of_2]]
        tseries_RL = np.r_[np.delete(tseries_RL1, invalid_rows_of_1, axis=0), tseries_RL2[new_rows_of_2]]
        tseries_LR = stats.zscore(tseries_LR, 1)
        tseries_RL = stats.zscore(tseries_RL, 1)
        tseries = np.c_[tseries_LR, tseries_RL]
        site_num = tseries.shape[0]
        subject_connectivity = np.zeros((site_num, site_num))
        for i in range(site_num):
            for j in range(i, site_num):
                r = stats.pearsonr(tseries[i], tseries[j])[0]
                subject_connectivity[i, j] = r
                subject_connectivity[j, i] = r
        connectivity.append(subject_connectivity)
        print(subject)

    arr_shape_info = 'array_shape,(subject_num region_num region_num)'
    region_names = ['region_name'] + r_names
    region_names_info = ','.join(region_names)
    open(out_info_file, 'w+').writelines('\n'.join([arr_shape_info, region_names_info]))
    np.save(out_conn_file, np.array(connectivity))
