if __name__ == '__main__':
    import numpy as np
    import nibabel as nib

    from os.path import join as pjoin
    from collections import OrderedDict
    from commontool.io.io import CiftiReader

    # predefine some variates
    # -----------------------
    # predefine parameters
    n_clusters = 3
    subproject_name = '2mm_KM_init10_regress_right'
    brain_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    interested_copes = OrderedDict([(18, 'BODY-AVG'),
                                    (19, 'FACE-AVG'),
                                    (20, 'PLACE-AVG'),
                                    (21, 'TOOL-AVG')])
    roi2label = {'r1_FFA_m': '3',
                 'r2_FFA_p': '1',
                 'r2_FFA_a': '1',
                 'r3_FFA_p': '2',
                 'r3_FFA_a': '2'}

    # predefine paths
    working_dir = '/nfs/s2/userhome/chenxiayu/workingdir'
    project_dir = pjoin(working_dir, 'study/FFA_clustering')
    subproject_dir = pjoin(project_dir, subproject_name)
    n_clusters_dir = pjoin(subproject_dir, '{}clusters'.format(n_clusters))
    # FFA_label = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm.label')
    FFA_label = pjoin(n_clusters_dir, '{}.label')
    cope_files = pjoin(project_dir, 'data/S1200_1080_WM_cope{0}_{1}_s2_MSMAll_32k_fs_LR.dscalar.nii')
    subject_labels_path = pjoin(n_clusters_dir, 'subject_labels')
    # -----------------------

    # get data
    with open(subject_labels_path) as f:
        subject_labels = np.array(f.read().split(' '))

    # analyze labels
    # --------------
    cope_titles = ['{}_FFA_mean'.format(v) for v in interested_copes.values()]
    cope_dict = dict()
    for roi in roi2label.keys():
        cope_dict[roi] = dict()
    for k, v in interested_copes.items():
        cope_file = cope_files.format(k, v)
        cope_data = CiftiReader(cope_file).get_data(brain_structure, True)
        for roi, label in roi2label.items():
            FFA_vertices = nib.freesurfer.read_label(FFA_label.format(roi))
            cope_data_FFA = cope_data[:, FFA_vertices]
            # get subgroup data
            sub_cope_data_FFA = np.atleast_2d(cope_data_FFA[subject_labels == label])
            sub_cope_data_FFA_mean = np.mean(sub_cope_data_FFA, axis=1)
            cope_dict[roi]['{}_FFA_mean'.format(v)] = sub_cope_data_FFA_mean
        print('Finished: {0}_{1}'.format(k, v))

    # output copes
    for roi in cope_dict.keys():
        with open(pjoin(n_clusters_dir, '{}_func_fingerprint.csv'.format(roi)), 'w+') as wf:
            wf.write(','.join(cope_titles) + '\n')
            columns = []
            for title in cope_titles:
                columns.append(cope_dict[roi][title])
            for row in zip(*columns):
                row = [str(_) for _ in row]
                wf.write(','.join(row) + '\n')
