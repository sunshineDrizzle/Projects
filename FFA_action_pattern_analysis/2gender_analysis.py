import numpy as np
import nibabel as nib

from os.path import join as pjoin
from scipy.stats import ttest_ind, sem
from matplotlib import pyplot as plt
from commontool.io.io import CiftiReader
from commontool.algorithm.plot import show_bar_value, auto_bar_width

project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'


def different_activation_gender_1080(b_dict):
    # file paths
    maps_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    lFFA_file = pjoin(project_dir, 'data/HCP_face-avg/label/lFFA_2mm.label')
    rFFA_file = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm.label')

    # get maps
    cifti_reader = CiftiReader(maps_file)
    lFFA = nib.freesurfer.read_label(lFFA_file)
    rFFA = nib.freesurfer.read_label(rFFA_file)
    lFFA_maps = cifti_reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)[:, lFFA]
    rFFA_maps = cifti_reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)[:, rFFA]

    # get subjects' ids
    subjects = np.array(b_dict['Subject'])
    genders = np.array(b_dict['Gender'])
    subjects_m = subjects[genders == 'M']
    subjects_f = subjects[genders == 'F']
    map2subject = [name.split('_')[0] for name in cifti_reader.map_names()]

    gender_labels = np.zeros((lFFA_maps.shape[0],), dtype=np.str)
    for idx, subj_id in enumerate(map2subject):
        if subj_id in subjects_m:
            gender_labels[idx] = 'M'
        elif subj_id in subjects_f:
            gender_labels[idx] = 'F'

    lFFA_maps_mean = np.mean(lFFA_maps, 1)
    rFFA_maps_mean = np.mean(rFFA_maps, 1)
    lFFA_maps_mean_m = np.mean(lFFA_maps[gender_labels == 'M'], 1)
    lFFA_maps_mean_f = np.mean(lFFA_maps[gender_labels == 'F'], 1)
    rFFA_maps_mean_m = np.mean(rFFA_maps[gender_labels == 'M'], 1)
    rFFA_maps_mean_f = np.mean(rFFA_maps[gender_labels == 'F'], 1)
    print('lFFA vs. rFFA: p={}'.format(ttest_ind(lFFA_maps_mean, rFFA_maps_mean)[1]))
    print('lFFA_male vs. lFFA_female: p={}'.format(ttest_ind(lFFA_maps_mean_m, lFFA_maps_mean_f)[1]))
    print('rFFA_male vs. rFFA_female: p={}'.format(ttest_ind(rFFA_maps_mean_m, rFFA_maps_mean_f)[1]))
    print('lFFA_male vs. rFFA_male: p={}'.format(ttest_ind(lFFA_maps_mean_m, rFFA_maps_mean_m)[1]))
    print('lFFA_female vs. rFFA_female: p={}'.format(ttest_ind(lFFA_maps_mean_f, rFFA_maps_mean_f)[1]))

    l_mean = np.mean(lFFA_maps_mean)
    r_mean = np.mean(rFFA_maps_mean)
    l_std = np.std(lFFA_maps_mean)
    r_std = np.std(rFFA_maps_mean)
    l_sem = sem(lFFA_maps_mean)
    r_sem = sem(rFFA_maps_mean)
    l_m_mean = np.mean(lFFA_maps_mean_m)
    l_f_mean = np.mean(lFFA_maps_mean_f)
    r_m_mean = np.mean(rFFA_maps_mean_m)
    r_f_mean = np.mean(rFFA_maps_mean_f)
    l_m_std = np.std(lFFA_maps_mean_m)
    l_f_std = np.std(lFFA_maps_mean_f)
    r_m_std = np.std(rFFA_maps_mean_m)
    r_f_std = np.std(rFFA_maps_mean_f)
    l_m_sem = sem(lFFA_maps_mean_m)
    l_f_sem = sem(lFFA_maps_mean_f)
    r_m_sem = sem(rFFA_maps_mean_m)
    r_f_sem = sem(rFFA_maps_mean_f)

    x = np.arange(2)
    fig, ax = plt.subplots()
    width = auto_bar_width(x, 3)
    rects1 = ax.bar(x, [l_mean, r_mean], width, color='g', alpha=0.5,
                    yerr=[l_sem, r_sem], ecolor='green')
    rects2 = ax.bar(x - width, [l_m_mean, r_m_mean], width, color='b', alpha=0.5,
                    yerr=[l_m_sem, r_m_sem], ecolor='blue')
    rects3 = ax.bar(x + width, [l_f_mean, r_f_mean], width, color='r', alpha=0.5,
                    yerr=[l_f_sem, r_f_sem], ecolor='red')
    # show_bar_value(rects1, '.3f')
    # show_bar_value(rects2, '.3f')
    # show_bar_value(rects3, '.3f')
    ax.legend((rects1, rects2, rects3), ('both', 'male', 'female'))
    ax.set_xticks(x)
    ax.set_xticklabels(['lFFA_2mm', 'rFFA_2mm'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('activation (z-stat)')

    plt.tight_layout()
    plt.show()


def different_activation_gender_subgroup(b_dict):
    # predefine some variables
    brain_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    rois = ['r1_FFA_m', 'r2_FFA_p', 'r2_FFA_a', 'r3_FFA_p', 'r3_FFA_a']
    roi2label = {'r1_FFA_m': '3', 'l1_FFA_m': '2',
                 'r2_FFA_p': '1', 'l2_FFA_p': '1',
                 'r2_FFA_a': '1', 'l2_FFA_a': '1',
                 'r3_FFA_p': '2', 'l3_FFA_p': '3',
                 'r3_FFA_a': '2', 'l3_FFA_a': '3'}

    # file paths
    maps_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    n_clusters_dir = pjoin(project_dir, '2mm_KM_init10_regress_right/3clusters')
    roi_files = pjoin(n_clusters_dir, '{}.label')
    subject_labels_file = pjoin(n_clusters_dir, 'subject_labels')

    with open(subject_labels_file) as f:
        subject_labels = np.array(f.read().split(' '))

    cifti_reader = CiftiReader(maps_file)
    maps = cifti_reader.get_data(brain_structure, True)
    # get gender labels
    subjects = np.array(b_dict['Subject'])
    genders = np.array(b_dict['Gender'])
    subjects_m = subjects[genders == 'M']
    subjects_f = subjects[genders == 'F']
    map2subject = [name.split('_')[0] for name in cifti_reader.map_names()]
    gender_labels = np.zeros((maps.shape[0],), dtype=np.str)
    for idx, subj_id in enumerate(map2subject):
        if subj_id in subjects_m:
            gender_labels[idx] = 'M'
        elif subj_id in subjects_f:
            gender_labels[idx] = 'F'

    means_m = []
    means_f = []
    stds_m = []
    stds_f = []
    sems_m = []
    sems_f = []
    for roi in rois:
        roi_file = roi_files.format(roi)
        roi_vertices = nib.freesurfer.read_label(roi_file)
        roi_maps = maps[:, roi_vertices]

        male_indices = np.logical_and(gender_labels == 'M', subject_labels == roi2label[roi])
        female_indices = np.logical_and(gender_labels == 'F', subject_labels == roi2label[roi])
        roi_map_means_m = np.mean(roi_maps[male_indices], 1)
        roi_map_means_f = np.mean(roi_maps[female_indices], 1)
        print('the number of males about {0}: {1}'.format(roi, roi_map_means_m.shape[0]))
        print('the number of females about {0}: {1}'.format(roi, roi_map_means_f.shape[0]))
        print('{0}_male vs. {0}_female: p={1}\n'.format(roi, ttest_ind(roi_map_means_m, roi_map_means_f)[1]))

        means_m.append(np.mean(roi_map_means_m))
        means_f.append(np.mean(roi_map_means_f))
        stds_m.append(np.std(roi_map_means_m))
        stds_f.append(np.std(roi_map_means_f))
        sems_m.append(sem(roi_map_means_m))
        sems_f.append(sem(roi_map_means_f))

    x = np.arange(len(rois))
    fig, ax = plt.subplots()
    width = auto_bar_width(x, 2)
    rects1 = ax.bar(x, means_m, width, color='b', alpha=0.5,
                    yerr=sems_m, ecolor='blue')
    rects2 = ax.bar(x + width, means_f, width, color='r', alpha=0.5,
                    yerr=sems_f, ecolor='red')
    # show_bar_value(rects1, '.3f')
    # show_bar_value(rects2, '.3f')
    ax.legend((rects1, rects2), ('male', 'female'))
    ax.set_xticks(x + width/2.0)
    ax.set_xticklabels(rois)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('activation (z-stat)')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from commontool.io.io import CsvReader

    behavior_file = pjoin(project_dir, 'data/S1200_behavior.csv')
    csv_reader = CsvReader(behavior_file)
    behavior_dict = csv_reader.to_dict()

    # different_activation_gender_1080(behavior_dict)
    different_activation_gender_subgroup(behavior_dict)
