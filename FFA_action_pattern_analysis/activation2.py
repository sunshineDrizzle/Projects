if __name__ == '__main__':
    import os
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from commontool.io.io import CiftiReader, save2nifti, GiftiReader
    from FFA_action_pattern_analysis.clustering1 import get_roi_patterns

    # predefine some variates
    # -----------------------
    # predefine parameters
    cluster_nums = [2]
    hemi = 'lh'
    brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
    acti_thr = 2.3  # a threshold about significantly activated
    prob_thr = 0.8
    top_acti_percent = 0.1
    stats_table_titles = ['label', '#subjects',
                          'map_min', 'map_max', 'map_mean',
                          'FFA_min', 'FFA_max', 'FFA_mean']

    zscore = False  # If true, do z-score on each subject's FFA pattern
    thr = 2.3  # a threshold used to cut FFA_data before clustering (default: None)
    bin = True  # If true, binarize FFA_data according to clustering_thr
    size_min = 5

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    cluster_num_dirs = pjoin(project_dir, '2mm_15_HAC_ward_euclidean_thr2.3_lt5_bin/{}clusters')
    FFA_label_file = pjoin(project_dir, 'data/HCP_face-avg/label/{}FFA_2mm_15.label')
    maps_path = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    lh_geo_file = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/' \
                  'HCP_S1200_GroupAvg_v1/S1200.L.white_MSMAll.32k_fs_LR.surf.gii'
    rh_geo_file = '/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/' \
                  'HCP_S1200_GroupAvg_v1/S1200.R.white_MSMAll.32k_fs_LR.surf.gii'
    mask_file = pjoin(project_dir, 'data/HCP_face-avg/s2/patches_15/crg2.3/{}FFA_patch_maps_lt5.nii.gz')
    # -----------------------

    # get maps
    FFA_vertices = nib.freesurfer.read_label(FFA_label_file.format(hemi[0]))
    reader = CiftiReader(maps_path)
    maps = reader.get_data(brain_structure[hemi], True)

    if hemi == 'lh':
        geo_reader = GiftiReader(lh_geo_file)
    elif hemi == 'rh':
        geo_reader = GiftiReader(rh_geo_file)
    else:
        raise RuntimeError("invalid hemi: {}".format(hemi))

    if mask_file is not None:
        mask = nib.load(mask_file.format(hemi[0])).get_data() != 0
    else:
        mask = None

    # analyze labels
    # --------------
    for cluster_num in cluster_nums:
        # get clustering labels of subjects
        cluster_num_dir = cluster_num_dirs.format(cluster_num)
        subject_labels_path = pjoin(cluster_num_dir, 'subject_labels')
        with open(subject_labels_path) as rf:
            subject_labels = np.array(rf.read().split(' '), dtype=np.uint16)

        stats_table_content = dict()
        for title in stats_table_titles:
            # initialize statistics table content
            stats_table_content[title] = []

        mean_maps = np.zeros((0, maps.shape[1]))
        prob_maps = np.zeros((0, maps.shape[1]))
        num_maps = np.zeros((0, maps.shape[1]))
        processed_mean_maps = np.zeros((0, maps.shape[1]))
        for label in sorted(set(subject_labels)):
            # get subgroup data
            subgroup_maps = maps[subject_labels == label]
            subgroup_maps_mean = np.mean(subgroup_maps, 0)
            subgroup_maps_prob = np.mean(subgroup_maps > acti_thr, 0)
            subgroup_maps_numb = np.sum(subgroup_maps > acti_thr, 0)
            subgroup_FFA_maps_mean = subgroup_maps_mean[FFA_vertices]

            mean_maps = np.r_[mean_maps, np.atleast_2d(subgroup_maps_mean)]
            prob_maps = np.r_[prob_maps, np.atleast_2d(subgroup_maps_prob)]
            num_maps = np.r_[num_maps, np.atleast_2d(subgroup_maps_numb)]

            stats_table_content['label'].append(str(label))
            stats_table_content['#subjects'].append(str(subgroup_maps.shape[0]))
            stats_table_content['map_min'].append(str(np.min(subgroup_maps_mean)))
            stats_table_content['map_max'].append(str(np.max(subgroup_maps_mean)))
            stats_table_content['map_mean'].append(str(np.mean(subgroup_maps_mean)))
            stats_table_content['FFA_min'].append(str(np.min(subgroup_FFA_maps_mean)))
            stats_table_content['FFA_max'].append(str(np.max(subgroup_FFA_maps_mean)))
            stats_table_content['FFA_mean'].append(str(np.mean(subgroup_FFA_maps_mean)))

            # get processed mean maps
            if mask is None:
                subgroup_FFA_patterns = get_roi_patterns(subgroup_maps, FFA_vertices,
                                                         zscore, thr, bin, size_min, geo_reader.faces, mask)
            else:
                subgroup_FFA_patterns = get_roi_patterns(subgroup_maps, FFA_vertices,
                                                         zscore, thr, bin, size_min, geo_reader.faces,
                                                         mask[subject_labels == label])

            subgroup_FFA_patterns_mean = np.atleast_2d(np.mean(subgroup_FFA_patterns, 0))
            processed_mean_map = np.ones((1, maps.shape[1])) * np.min(subgroup_FFA_patterns_mean)
            processed_mean_map[:, FFA_vertices] = subgroup_FFA_patterns_mean
            processed_mean_maps = np.r_[processed_mean_maps, processed_mean_map]

        max_num_map = np.argmax(num_maps, 0) + 1
        max_prob_map = np.argmax(prob_maps, 0) + 1
        top_prob_ROIs = (prob_maps > prob_thr).astype(np.int8)
        top_acti_ROIs = np.zeros_like(mean_maps)
        for row, mean_map_FFA in enumerate(mean_maps[:, FFA_vertices]):
            col_val = list(zip(FFA_vertices, mean_map_FFA))
            col_val_sorted = sorted(col_val, key=lambda x: x[1], reverse=True)
            col_val_top = col_val_sorted[:int(len(FFA_vertices)*top_acti_percent)]
            for col, val in col_val_top:
                top_acti_ROIs[row, col] = 1

        # output maps
        activation_dir = pjoin(cluster_num_dir, 'activation')
        if not os.path.exists(activation_dir):
            os.makedirs(activation_dir)
        save2nifti(pjoin(activation_dir, '{}_mean_maps.nii.gz'.format(hemi)), mean_maps)
        save2nifti(pjoin(activation_dir, '{}_prob_maps_z{}.nii.gz'.format(hemi, acti_thr)), prob_maps)
        # save2nifti(pjoin(activation_dir, 'max_num_map_z{}.nii.gz'.format(acti_thr)), max_num_map)
        # save2nifti(pjoin(activation_dir, 'max_prob_map_z{}.nii.gz'.format(acti_thr)), max_prob_map)
        # save2nifti(pjoin(activation_dir, 'top_prob_ROIs_z{}_p{}.nii.gz'.format(acti_thr, prob_thr)), top_prob_ROIs)
        save2nifti(pjoin(activation_dir, '{}_top_acti_ROIs_percent{}.nii.gz'.format(hemi, top_acti_percent * 100)), top_acti_ROIs)
        save2nifti(pjoin(activation_dir, '{}_processed_mean_maps.nii.gz'.format(hemi)), processed_mean_maps)

        # output statistics
        with open(pjoin(activation_dir, '{}_statistics.csv'.format(hemi)), 'w+') as f:
            f.write(','.join(stats_table_titles) + '\n')
            lines = []
            for title in stats_table_titles:
                lines.append(stats_table_content[title])
            for line in zip(*lines):
                f.write(','.join(line) + '\n')

        print('{}clusters: done'.format(cluster_num))
