if __name__ == '__main__':
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    # predefine some variates
    # -----------------------
    # predefine parameters
    n_clusters = 2
    subproject_name = '2mm_KM_zscore_left'
    brain_structure = 'CIFTI_STRUCTURE_CORTEX_LEFT'
    acti_thr = 2.3  # a threshold about significantly activated
    prob_thr = 0.8
    top_acti_percent = 0.1

    # predefine paths
    working_dir = '/nfs/s2/userhome/chenxiayu/workingdir'
    project_dir = pjoin(working_dir, 'study/FFA_clustering')
    subproject_dir = pjoin(project_dir, subproject_name)
    n_clusters_dir = pjoin(subproject_dir, '{}clusters'.format(n_clusters))
    FFA_label = pjoin(project_dir, 'data/HCP_face-avg/label/lFFA_2mm.label')
    FSR_maps = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    subject_labels_path = pjoin(n_clusters_dir, 'subject_labels')
    # -----------------------

    # get data
    FFA_vertices = nib.freesurfer.read_label(FFA_label)
    reader = CiftiReader(FSR_maps)
    data = reader.get_data(brain_structure, True)
    FFA_data = data[:, FFA_vertices]
    with open(subject_labels_path) as f:
        subject_labels = np.array(f.read().split(' '))

    # analyze labels
    # --------------
    stats_table_titles = ['label', '#subjects',
                          'map_min', 'map_max', 'map_mean',
                          'FFA_min', 'FFA_max', 'FFA_mean']
    stats_table_content = dict()
    for title in stats_table_titles:
        # initialize statistics table content
        stats_table_content[title] = []

    data_means = np.zeros((0, data.shape[1]))
    data_probs = np.zeros((0, data.shape[1]))
    data_nums = np.zeros((0, data.shape[1]))
    for label in sorted(set(subject_labels)):
        # get subgroup data
        subgroup_data = data[subject_labels == str(label)]
        subgroup_data_mean = np.mean(subgroup_data, 0)
        subgroup_data_prob = np.mean(subgroup_data > acti_thr, 0)
        subgroup_data_numb = np.sum(subgroup_data > acti_thr, 0)
        subgroup_FFA_data_mean = subgroup_data_mean[FFA_vertices]

        data_means = np.r_[data_means, np.atleast_2d(subgroup_data_mean)]
        data_probs = np.r_[data_probs, np.atleast_2d(subgroup_data_prob)]
        data_nums = np.r_[data_nums, np.atleast_2d(subgroup_data_numb)]

        stats_table_content['label'].append(str(label))
        stats_table_content['#subjects'].append(str(subgroup_data.shape[0]))
        stats_table_content['map_min'].append(str(np.min(subgroup_data_mean)))
        stats_table_content['map_max'].append(str(np.max(subgroup_data_mean)))
        stats_table_content['map_mean'].append(str(np.mean(subgroup_data_mean)))
        stats_table_content['FFA_min'].append(str(np.min(subgroup_FFA_data_mean)))
        stats_table_content['FFA_max'].append(str(np.max(subgroup_FFA_data_mean)))
        stats_table_content['FFA_mean'].append(str(np.mean(subgroup_FFA_data_mean)))

    max_num_map = np.argmax(data_nums, 0) + 1
    max_prob_map = np.argmax(data_probs, 0) + 1
    top_prob_ROIs = (data_probs > prob_thr).astype(np.int8)
    top_acti_ROIs = np.zeros_like(data_means)
    for row, mean_map_FFA in enumerate(data_means[:, FFA_vertices]):
        col_val = list(zip(FFA_vertices, mean_map_FFA))
        col_val_sorted = sorted(col_val, key=lambda x: x[1], reverse=True)
        col_val_top = col_val_sorted[:int(len(FFA_vertices)*top_acti_percent)]
        for col, val in col_val_top:
            top_acti_ROIs[row, col] = 1

    # output data
    save2nifti(pjoin(n_clusters_dir, 'mean_maps.nii.gz'), data_means)
    save2nifti(pjoin(n_clusters_dir, 'prob_maps_z{}.nii.gz'.format(acti_thr)), data_probs)
    # save2nifti(pjoin(n_clusters_dir, 'max_num_map_z{}.nii.gz'.format(acti_thr)), max_num_map)
    # save2nifti(pjoin(n_clusters_dir, 'max_prob_map_z{}.nii.gz'.format(acti_thr)), max_prob_map)
    # save2nifti(pjoin(n_clusters_dir, 'top_prob_ROIs_z{}_p{}.nii.gz'.format(acti_thr, prob_thr)), top_prob_ROIs)
    save2nifti(pjoin(n_clusters_dir, 'top_acti_ROIs_percent{}.nii.gz'.format(top_acti_percent * 100)), top_acti_ROIs)

    # output statistics
    with open(pjoin(n_clusters_dir, 'statistics.csv'), 'w+') as f:
        f.write(','.join(stats_table_titles) + '\n')
        lines = []
        for title in stats_table_titles:
            lines.append(stats_table_content[title])
        for line in zip(*lines):
            f.write(','.join(line) + '\n')
