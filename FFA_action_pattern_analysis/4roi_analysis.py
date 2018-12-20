import nibabel as nib

from commontool.io.io import CiftiReader

hemis = ['lh', 'rh']
brain_structure = {
        'lh': 'CIFTI_STRUCTURE_CORTEX_LEFT',
        'rh': 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    }
roi_names = '{hemi}{label}_FFA{roi_label}'


def calc_roi_mean_intrasubgroup(src_file, roi_files, subject_labels, trg_file):
    """
    Calculate ROI means for each subject in corresponding subgroup.
    """
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


def calc_roi_mean_allsubgroup(src_file, roi_files, subject_labels, trg_file):
    """
    Calculate ROI means for each subject of each subgroup in corresponding hemisphere.
    For example, subgourp1's right ROI1 will be used to calculate ROI means not only for subgroup1's subjects,
    but also for other subgroups' subjects.
    """
    reader = CiftiReader(src_file)
    labels = np.unique(subject_labels)
    roi_mean_rows = []
    for hemi in hemis:
        maps = reader.get_data(brain_structure[hemi], True)
        for label1 in labels:
            sub_maps = np.atleast_2d(maps[subject_labels == label1])
            for label2 in labels:
                roi_file = roi_files.format(hemi=hemi[0], label=label2)
                roi_mask = nib.load(roi_file).get_data().ravel()
                roi_labels = np.unique(roi_mask)
                for roi_label in roi_labels:
                    if roi_label == 0:
                        continue
                    roi_vertices = np.where(roi_mask == roi_label)[0]
                    roi_name = roi_names.format(hemi=hemi[0], label=label2, roi_label=int(roi_label))
                    roi_name += '_in_subgroup{}'.format(label1)
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
    cluster_num_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters')
    roi_files = pjoin(cluster_num_dir, 'activation/{hemi}{label}_FFA.nii.gz')
    subject_labels_file = pjoin(cluster_num_dir, 'subject_labels')
    roi_dir = pjoin(cluster_num_dir, 'roi_analysis')
    if not os.path.exists(roi_dir):
        os.makedirs(roi_dir)

    subject_labels = np.array(open(subject_labels_file).read().split(' '), dtype=np.uint16)

    calc_roi_mean_intrasubgroup(
        pjoin(project_dir, 'data/HCP_1080/S1200_1080_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii'),
        roi_files, subject_labels, pjoin(roi_dir, 'roi_mean_face-avg_intrasubgroup')
    )
    calc_roi_mean_allsubgroup(
        pjoin(project_dir, 'data/HCP_1080/S1200_1080_WM_cope19_FACE-AVG_s2_MSMAll_32k_fs_LR.dscalar.nii'),
        roi_files, subject_labels, pjoin(roi_dir, 'roi_mean_face-avg_allsubgroup')
    )
    # calc_roi_mean_allsubgroup(
    #     pjoin(project_dir, 'data/HCP_1080/S1200_1080_WM_Mean_BOLD_Signal_MSMAll_32k_fs_LR.dscalar.nii'),
    #     roi_files, subject_labels, pjoin(roi_dir, 'roi_mean_mean_bold_signal_allsubgroup')
    # )
