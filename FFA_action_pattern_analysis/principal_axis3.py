import numpy as np

from os.path import join as pjoin
from scipy import stats
from commontool.io.io import CiftiReader


def PA_plot(src_file, brain_structures, PAs, subject_labels, axes, color, item, zscore):

    reader = CiftiReader(src_file)
    for col in range(2):
        PA = PAs[col]
        x = np.arange(len(PA))
        PA_maps = reader.get_data(brain_structures[col], True)[:, PA]
        if zscore:
            PA_maps = stats.zscore(PA_maps, 1)
        for row, label in enumerate(sorted(set(subject_labels))):
            subgroup_PA_maps = np.atleast_2d(PA_maps[subject_labels == label])
            y = np.mean(subgroup_PA_maps, 0)
            sem = stats.sem(subgroup_PA_maps, 0)
            axes[row, col].errorbar(x, y, yerr=sem, color=color, label=item)
            axes[row, col].set_title(label)


if __name__ == '__main__':
    import nibabel as nib
    import matplotlib.pyplot as plt

    cluster_num = 3
    items = [
        'activation',
        # 'curvature',
        'thickness',
        # 'myelin'
    ]

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    cluster_num_dir = pjoin(project_dir, '2mm_KM_zscore/{}clusters'.format(cluster_num))
    lh_PA_file = pjoin(project_dir, 'data/HCP_face-avg/label/lFFA_2mm_PA.label')
    rh_PA_file = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm_PA.label')
    subject_labels_path = pjoin(cluster_num_dir, 'subject_labels')

    lh_PA = nib.freesurfer.read_label(lh_PA_file)[-1::-1]  # from posterior to anterior
    rh_PA = nib.freesurfer.read_label(rh_PA_file)[-1::-1]  # from posterior to anterior
    with open(subject_labels_path) as rf:
        subject_labels = np.array(rf.read().split(' '), dtype=np.uint16)

    fig, axes = plt.subplots(cluster_num, 2, sharex=True)
    axes[-1, 0].set_xlabel('from posterior to anterior of lFFA')
    axes[-1, 1].set_xlabel('from posterior to anterior of rFFA')

    PAs = [lh_PA, rh_PA]
    brain_structures = ['CIFTI_STRUCTURE_CORTEX_LEFT', 'CIFTI_STRUCTURE_CORTEX_RIGHT']
    for item in items:
        if item == 'pattern':
            lh_src_file = pjoin(cluster_num_dir, 'activation/lh_zscore_mean_maps.nii.gz')
            rh_src_file = pjoin(cluster_num_dir, 'activation/rh_zscore_mean_maps.nii.gz')
            lmaps = nib.load(lh_src_file).get_data()
            rmaps = nib.load(rh_src_file).get_data()
            pass
        elif item == 'activation':
            src_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
            PA_plot(src_file, brain_structures, PAs, subject_labels, axes, 'y', item, True)
        elif item == 'curvature':
            src_file = pjoin(project_dir, 'data/HCP_face-avg/S1200.1080.curvature_MSMAll.32k_fs_LR.dscalar.nii')
            PA_plot(src_file, brain_structures, PAs, subject_labels, axes, 'r', item, True)
        elif item == 'thickness':
            src_file = pjoin(project_dir, 'data/HCP_face-avg/S1200_1080_thickness_MSMAll_32k_fs_LR.dscalar.nii')
            PA_plot(src_file, brain_structures, PAs, subject_labels, axes, 'g', item, True)
        elif item == 'myelin':
            src_file = pjoin(project_dir, 'data/HCP_face-avg/S1200_1080_MyelinMap_BC_MSMAll_32k_fs_LR.dscalar.nii')
            PA_plot(src_file, brain_structures, PAs, subject_labels, axes, 'b', item, True)
        else:
            raise RuntimeError("no such item: {}".format(item))

    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            axes[row, col].legend()

    plt.tight_layout()
    plt.show()
