import nibabel as nib

from commontool.io.io import CiftiReader

hemis = ['lh', 'rh']
brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
roi_names = '{hemi}{label}_FFA{roi_label}'


def calc_roi_mean(src_file, roi_files, subject_labels, trg_file):
    reader = CiftiReader(src_file)
    labels = np.unique(subject_labels)
    roi_mean_rows = []
    for hemi in hemis:
        maps = reader.get_data(brain_structure[hemi], True)
        for label in labels:
            sub_maps = np.atleast_2d(maps[subject_labels == label])
            roi_file = roi_files.format(hemi=hemi[0], label=label)
            roi_mask = nib.load(roi_file).get_data().ravel()
            roi_labels = np.unique(roi_mask)
            for roi_label in roi_labels:
                if roi_label == 0:
                    continue
                roi_vertices = np.where(roi_mask == roi_label)[0]
                roi_name = roi_names.format(hemi=hemi[0], label=label, roi_label=int(roi_label))
                roi_means = np.mean(sub_maps[:, roi_vertices], 1)

                roi_mean_row = [roi_name]
                roi_mean_row.extend([str(_) for _ in roi_means])
                roi_mean_rows.append(','.join(roi_mean_row))
    open(trg_file, 'w+').writelines('\n'.join(roi_mean_rows))


if __name__ == '__main__':
    import os
    import numpy as np

    from os.path import join as pjoin

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    cluster_num_dir = pjoin(project_dir, '2mm_25_HAC_ward_euclidean_zscore/2clusters')
    roi_files = pjoin(cluster_num_dir, 'activation/{hemi}{label}_FFA.nii.gz')
    subject_labels_file = pjoin(cluster_num_dir, 'subject_labels')
    roi_dir = pjoin(cluster_num_dir, 'roi_analysis')
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)

    subject_labels = np.array(open(subject_labels_file).read().split(' '), dtype=np.uint16)

    calc_roi_mean(pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii'),
                  roi_files, subject_labels, pjoin(roi_dir, 'roi_mean_face-avg'))
