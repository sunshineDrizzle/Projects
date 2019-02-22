

if __name__ == '__main__':
    import numpy as np

    from os.path import join as pjoin
    from scipy import stats

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    connect_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/rfMRI_connectivity/PAM_z165_p025_ROI')
    subject_ids_file = pjoin(connect_dir, 'subject_id_4run_1200')
    output_file = pjoin(connect_dir, 'connectivity.npy')
    tseries_LR_file = pjoin(connect_dir, '{subject}/rfMRI_REST1_LR.npy')
    tseries_RL_file = pjoin(connect_dir, '{subject}/rfMRI_REST1_RL.npy')

    subject_ids = np.array(open(subject_ids_file).read().splitlines())

    connectivity = []
    for subject in subject_ids:
        tseries_LR = np.load(tseries_LR_file.format(subject=subject))
        tseries_RL = np.load(tseries_RL_file.format(subject=subject))
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
    np.save(output_file, np.array(connectivity))
