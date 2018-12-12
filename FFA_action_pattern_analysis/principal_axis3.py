import numpy as np

from os.path import join as pjoin
from scipy import stats
from commontool.io.io import CiftiReader


def PA_plot(src_file, brain_structures, PAs, subject_labels, axes, color, item, zscore, item_ys_dict):

    item_ys_dict[item] = np.zeros_like(axes, np.object)
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
            item_ys_dict[item][row, col] = y


def inter_subgroup_corr(item_ys_dict, item):
    ys = item_ys_dict[item]
    print('\ninter_subgroup_corr of {}:'.format(item))
    for col in range(ys.shape[1]):
        print("(0, {0}) vs. (1, {0}):".format(col), stats.pearsonr(ys[0, col], ys[1, col]))


def inter_item_corr(item_ys_dict, item1, item2):
    ys1 = item_ys_dict[item1]
    ys2 = item_ys_dict[item2]
    print('\ninter_item_corr between {} and {}:'.format(item1, item2))
    for row in range(ys1.shape[0]):
        for col in range(ys1.shape[1]):
            print('({}, {}):'.format(row, col), stats.pearsonr(ys1[row, col], ys2[row, col]))


if __name__ == '__main__':
    import nibabel as nib
    import matplotlib.pyplot as plt

    cluster_num = 2
    items = [
        'pattern',
        # 'activation',
        # 'curvature',
        # 'thickness',
        # 'myelin',
        'mean_bold_signal'
    ]
    item2color = {
        'pattern': 'k',
        'activation': 'y',
        'curvature': 'r',
        'thickness': 'g',
        'myelin': 'b',
        'mean_bold_signal': 'c'
    }

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    cluster_num_dir = pjoin(project_dir, '2mm_25_HAC_ward_euclidean_zscore/{}clusters'.format(cluster_num))
    lh_PA_file = pjoin(project_dir, 'data/HCP_face-avg/label/lFFA_2mm_25_PA.label')
    rh_PA_file = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm_25_PA.label')
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
    item_ys_dict = dict()
    for item in items:
        if item == 'pattern':
            item_ys_dict[item] = np.zeros_like(axes, dtype=object)
            lh_src_file = pjoin(cluster_num_dir, 'activation/lh_pattern_mean_maps.nii.gz')
            rh_src_file = pjoin(cluster_num_dir, 'activation/rh_pattern_mean_maps.nii.gz')
            src_files = [lh_src_file, rh_src_file]
            for col in range(2):
                PA = PAs[col]
                x = np.arange(len(PA))
                PA_maps = nib.load(src_files[col]).get_data()[:, PA]
                for row in range(cluster_num):
                    y = PA_maps[row]
                    axes[row, col].plot(x, y, color=item2color[item], label=item)
                    axes[row, col].set_title(row+1)
                    item_ys_dict[item][row, col] = y
            continue
        if item == 'activation':
            src_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
        elif item == 'curvature':
            src_file = pjoin(project_dir, 'data/HCP_face-avg/S1200.1080.curvature_MSMAll.32k_fs_LR.dscalar.nii')
        elif item == 'thickness':
            src_file = pjoin(project_dir, 'data/HCP_face-avg/S1200_1080_thickness_MSMAll_32k_fs_LR.dscalar.nii')
        elif item == 'myelin':
            src_file = pjoin(project_dir, 'data/HCP_face-avg/S1200_1080_MyelinMap_BC_MSMAll_32k_fs_LR.dscalar.nii')
        elif item == 'mean_bold_signal':
            src_file = pjoin(project_dir, 'data/HCP_face-avg/S1200_1080_WM_Mean_BOLD_Signal_MSMAll.dscalar.nii')
        else:
            raise RuntimeError("no such item: {}".format(item))
        PA_plot(src_file, brain_structures, PAs, subject_labels, axes, item2color[item], item, True, item_ys_dict)

    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            axes[row, col].legend()

    inter_subgroup_corr(item_ys_dict, 'mean_bold_signal')
    inter_item_corr(item_ys_dict, 'pattern', 'mean_bold_signal')

    plt.tight_layout()
    plt.show()
