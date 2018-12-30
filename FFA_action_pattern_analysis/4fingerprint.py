if __name__ == '__main__':
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from collections import OrderedDict
    from commontool.io.io import CiftiReader

    # predefine some variates
    # -----------------------
    # predefine parameters
    brain_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    interested_copes = OrderedDict([(18, 'BODY-AVG'),
                                    (19, 'FACE-AVG'),
                                    (20, 'PLACE-AVG'),
                                    (21, 'TOOL-AVG')])
    rois = ['r1_FFA_m', 'r2_FFA_p', 'r2_FFA_a', 'r3_FFA_p', 'r3_FFA_a']
    roi2label = {'r1_FFA_m': '3', 'l1_FFA_m': '2',
                 'r2_FFA_p': '1', 'l2_FFA_p': '1',
                 'r2_FFA_a': '1', 'l2_FFA_a': '1',
                 'r3_FFA_p': '2', 'l3_FFA_p': '3',
                 'r3_FFA_a': '2', 'l3_FFA_a': '3'}

    # predefine paths
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    n_clusters_dir = pjoin(project_dir, '2mm_KM_init10_regress_right/3clusters')
    roi_file = pjoin(n_clusters_dir, '{}.label')
    cope_files = pjoin(project_dir, 'data/S1200_1080_WM_cope{0}_{1}_s2_MSMAll_32k_fs_LR.dscalar.nii')
    group_labels_path = pjoin(n_clusters_dir, 'group_labels')
    # -----------------------

    # get data
    with open(group_labels_path) as f:
        group_labels = np.array(f.read().split(' '))

    # analyze labels
    # --------------
    cope_dict = dict()
    for roi in rois:
        cope_dict[roi] = dict()
    for k, v in interested_copes.items():
        cope_file = cope_files.format(k, v)
        cope_data = CiftiReader(cope_file).get_data(brain_structure, True)
        for roi in rois:
            roi_vertices = nib.freesurfer.read_label(roi_file.format(roi))
            cope_data_roi = cope_data[:, roi_vertices]
            if roi2label:
                # get subgroup data
                sub_cope_data_roi = np.atleast_2d(cope_data_roi[group_labels == roi2label[roi]])
                sub_cope_data_roi_mean = np.mean(sub_cope_data_roi, axis=1)
                cope_dict[roi][v] = sub_cope_data_roi_mean
            else:
                cope_data_roi_mean = np.mean(cope_data_roi, axis=1)
                cope_dict[roi][v] = cope_data_roi_mean
        print('Finished: {0}_{1}'.format(k, v))

    # output copes
    cope_titles = interested_copes.values()
    for roi in cope_dict.keys():
        with open(pjoin(n_clusters_dir, '{}_func_fingerprint.csv'.format(roi)), 'w+') as wf:
            wf.write(','.join(cope_titles) + '\n')
            columns = []
            for title in cope_titles:
                columns.append(cope_dict[roi][title])
            for row in zip(*columns):
                row = map(str, row)
                wf.write(','.join(row) + '\n')
