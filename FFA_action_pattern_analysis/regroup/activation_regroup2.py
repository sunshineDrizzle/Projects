if __name__ == '__main__':
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti

    # predefine some variates
    # -----------------------
    # predefine parameters
    n_clusters = 20
    subproject_name = '2mm_ward_regress'
    result_name = 'regroup/1_2_2'
    regroups = [[2, 3, 4, 5, 6, 7, 12, 13, 14],
                [8, 9, 10, 17, 18],
                [1, 11, 15, 16, 19, 20]]
    acti_thr = 2.3  # a threshold about significantly activated
    prob_thr = 0.8
    top_acti_percent = 0.1

    # predefine paths
    working_dir = '/nfs/s2/userhome/chenxiayu/workingdir'
    project_dir = pjoin(working_dir, 'study/rFFA_clustering')
    subproject_dir = pjoin(project_dir, subproject_name)
    n_clusters_dir = pjoin(subproject_dir, '{}clusters'.format(n_clusters))
    result_dir = pjoin(n_clusters_dir, result_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    rFFA_label = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm.label')
    FSR_maps = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    subject_labels_path = pjoin(n_clusters_dir, 'subject_labels')
    # -----------------------

    # get data
    rFFA_vertices = nib.freesurfer.read_label(rFFA_label)
    reader = CiftiReader(FSR_maps)
    data = reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)
    rFFA_data = data[:, rFFA_vertices]
    with open(subject_labels_path) as f:
        subject_labels = np.array(f.read().split(' '))
    subjects_id = [name.split('_')[0] for name in reader.map_names()]
    subjects_id = np.array(subjects_id)

    # analyze labels
    # --------------
    stats_table_titles = ['regroup_id', '#subjects',
                          'map_min', 'map_max', 'map_mean',
                          'rFFA_min', 'rFFA_max', 'rFFA_mean',
                          'subgroups']
    stats_table_content = dict()
    for title in stats_table_titles:
        # initialize statistics table content
        stats_table_content[title] = []

    mean_maps = np.zeros((0, data.shape[1]))
    prob_maps = np.zeros((0, data.shape[1]))
    nums_maps = np.zeros((0, data.shape[1]))
    for idx, regroup in enumerate(regroups, 1):
        # get regroup index array
        idx_arr = np.zeros_like(subject_labels, dtype=np.bool)
        # get regroup IDs
        regroup_ids_verification = set()
        for label in regroup:
            idx_arr = np.logical_or(idx_arr, subject_labels == str(label))
            with open(pjoin(n_clusters_dir, 'subjects{}_id'.format(label))) as rf:
                id_tmp = rf.read().splitlines()
            regroup_ids_verification.update(id_tmp)
        regroup_ids = subjects_id[idx_arr]
        print(set(regroup_ids) == regroup_ids_verification)
        regroup_ids = '\n'.join(regroup_ids)
        # get regroup data
        subgroup_data = data[idx_arr]
        subgroup_data_mean = np.mean(subgroup_data, 0)
        subgroup_data_prob = np.mean(subgroup_data > acti_thr, 0)
        subgroup_data_numb = np.sum(subgroup_data > acti_thr, 0)
        subgroup_rFFA_data_mean = subgroup_data_mean[rFFA_vertices]

        mean_maps = np.r_[mean_maps, np.atleast_2d(subgroup_data_mean)]
        prob_maps = np.r_[prob_maps, np.atleast_2d(subgroup_data_prob)]
        nums_maps = np.r_[nums_maps, np.atleast_2d(subgroup_data_numb)]

        stats_table_content['regroup_id'].append(str(idx))
        stats_table_content['subgroups'].append('|'.join(map(str, regroup)))
        stats_table_content['#subjects'].append(str(subgroup_data.shape[0]))
        stats_table_content['map_min'].append(str(np.min(subgroup_data_mean)))
        stats_table_content['map_max'].append(str(np.max(subgroup_data_mean)))
        stats_table_content['map_mean'].append(str(np.mean(subgroup_data_mean)))
        stats_table_content['rFFA_min'].append(str(np.min(subgroup_rFFA_data_mean)))
        stats_table_content['rFFA_max'].append(str(np.max(subgroup_rFFA_data_mean)))
        stats_table_content['rFFA_mean'].append(str(np.mean(subgroup_rFFA_data_mean)))

        # output regroup IDs
        with open(pjoin(result_dir, 'regroup{}_id'.format(idx)), 'w+') as f:
            f.writelines(regroup_ids)

    max_num_map = np.argmax(nums_maps, 0) + 1
    max_prob_map = np.argmax(prob_maps, 0) + 1
    top_prob_ROIs = (prob_maps > prob_thr).astype(np.int8)
    top_acti_ROIs = np.zeros_like(mean_maps)
    for row, mean_map_rFFA in enumerate(mean_maps[:, rFFA_vertices]):
        col_val = list(zip(rFFA_vertices, mean_map_rFFA))
        col_val_sorted = sorted(col_val, key=lambda x: x[1], reverse=True)
        col_val_top = col_val_sorted[:int(len(rFFA_vertices)*top_acti_percent)]
        for col, val in col_val_top:
            top_acti_ROIs[row, col] = 1

    # output data
    save2nifti(pjoin(result_dir, 'mean_maps.nii.gz'), mean_maps)
    save2nifti(pjoin(result_dir, 'prob_maps_z{}.nii.gz'.format(acti_thr)), prob_maps)
    # save2nifti(pjoin(result_dir, 'max_num_map_z{}.nii.gz'.format(acti_thr)), max_num_map)
    # save2nifti(pjoin(result_dir, 'max_prob_map_z{}.nii.gz'.format(acti_thr)), max_prob_map)
    # save2nifti(pjoin(result_dir, 'top_prob_ROIs_z{}_p{}.nii.gz'.format(acti_thr, prob_thr)), top_prob_ROIs)
    save2nifti(pjoin(result_dir, 'top_acti_ROIs_percent{}.nii.gz'.format(top_acti_percent * 100)), top_acti_ROIs)

    # output statistics
    with open(pjoin(result_dir, 'statistics.csv'), 'w+') as f:
        f.write(','.join(stats_table_titles) + '\n')
        lines = []
        for title in stats_table_titles:
            lines.append(stats_table_content[title])
        for line in zip(*lines):
            f.write(','.join(line) + '\n')
