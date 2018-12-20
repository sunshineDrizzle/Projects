import numpy as np


def get_behavior_dict(d, names, subject_ids=None):
    """
    get behavior_dict from subject_dict

    :param d: dict
        subject_dict
    :param names: list
        behavior names
    :param subject_ids: list (default: None)
        subject ids in a subgroup
    :return behavior_dict: dict
        behavior_dict
    """
    subject_list = []
    if subject_ids is None:
        subject_list = list(d.values())
    else:
        for sub_id in subject_ids:
            subject_list.append(d[sub_id])

    behavior_list = list(zip(*subject_list))
    behavior_dict = {}
    for idx, name in enumerate(names):
        behavior_dict[name] = behavior_list[idx]

    return behavior_dict


def get_float_data(d, name):
    """
    get float data from behavior_dict, according to the behavior name

    Parameters:
    -----------
    d : dict
        behavior_dict
    name : str
        behavior name

    Return:
    -------
    float data
    """
    data = list(d[name])
    while '' in data:
        data.remove('')

    return np.array(list(map(float, data)))


def explore_float_data(d, name, label):
    """
    explore float data specified by behavior name in behavior_dict

    Parameters:
    ----------
    d : dict
        behavior_dict
    name : str
        behavior name
    label : integer
        the label of the current cluster
    """
    data = get_float_data(d, name)

    print('The number of valid subjects of subgroup{} for {}:'.format(label, name), len(data))
    return data


def explore_Gender(behavior_dict, label):
    male_num = behavior_dict['Gender'].count('M')
    female_num = behavior_dict['Gender'].count('F')

    print('The number of males in Cluster{}:'.format(label), male_num)
    print('The number of females in Cluster{}:'.format(label), female_num)
    return male_num, female_num


if __name__ == '__main__':
    from os.path import join as pjoin
    from scipy.stats import sem, ttest_ind
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import auto_bar_width, show_bar_value

    # predefine some variates
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    cluster_num_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters')
    subject_labels_file = pjoin(cluster_num_dir, 'subject_labels')
    subjects_id_file = pjoin(project_dir, 'data/HCP_1080/subject_id')

    with open(subject_labels_file) as rf:
        subject_labels = np.array(rf.read().split(' '), dtype=np.uint16)

    with open(subjects_id_file) as rf:
        subjects_1080 = np.array(rf.read().splitlines())

    with open(pjoin(project_dir, 'data/HCP/S1200_behavior.csv')) as f:
        lines = f.read().splitlines()
    behavior_names = lines[0].split(',')[1:]
    subject_dict = {}
    for line in lines[1:]:
        subject_id = line.split(',')[0]
        subject_dict[subject_id] = line.split(',')[1:]

    male_nums = []
    female_nums = []
    float_items = [
        # 'WM_Task_Acc',
        # 'WM_Task_0bk_Acc',
        # 'WM_Task_2bk_Acc',
        # 'WM_Task_0bk_Face_Acc',
        'WM_Task_2bk_Face_Acc',
        # 'WM_Task_Median_RT',
        # 'WM_Task_0bk_Median_RT',
        # 'WM_Task_2bk_Median_RT',
        # 'WM_Task_0bk_Face_Median_RT',
        'WM_Task_2bk_Face_Median_RT'
    ]
    float_data_dict = {}
    for item in float_items:
        float_data_dict[item] = []

    labels = np.unique(subject_labels)
    label_num = len(labels)
    for label in labels:
        subgroup_ids = subjects_1080[subject_labels == label]

        behavior_dict = get_behavior_dict(subject_dict, behavior_names, subgroup_ids)

        male_num, female_num = explore_Gender(behavior_dict, label)
        male_nums.append(male_num)
        female_nums.append(female_num)

        for item in float_items:
            float_data_dict[item].append(explore_float_data(behavior_dict, item, label))

    # plot
    x = np.arange(label_num)
    width = auto_bar_width(x, 2)
    plt.figure()
    ax = plt.gca()
    rects1 = ax.bar(x, male_nums, width, color='b')
    rects2 = ax.bar(x + width, female_nums, width, color='r')
    show_bar_value(rects1)
    show_bar_value(rects2)
    ax.legend((rects1[0], rects2[1]), ('male', 'female'))
    ax.set_xticks(x + width / 2.0)
    ax.set_xticklabels(labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Gender')
    ax.set_xlabel('subgroup label')
    ax.set_ylabel('count')
    # plt.savefig(pjoin(cluster_num_dir, 'Gender.png'))

    width = auto_bar_width(x)
    for item in float_items:
        plt.figure()
        ax = plt.gca()
        if 'Acc' in item:
            float_data_list = [float_data/100.0 for float_data in float_data_dict[item]]
        elif 'RT' in item:
            float_data_list = [float_data/1000.0 for float_data in float_data_dict[item]]
        else:
            raise RuntimeError('{} is not supported!'.format(item))
        y = [np.mean(float_data) for float_data in float_data_list]
        sems = [sem(float_data) for float_data in float_data_list]
        rects = ax.bar(x, y, width, color='b', yerr=sems, ecolor='blue', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(item)
        ax.set_xlabel('subgroup label')
        if 'Acc' in item:
            show_bar_value(rects, '.2%')
            ax.set_ylabel('accuracy')
        elif 'RT' in item:
            show_bar_value(rects, '.2f')
            ax.set_ylabel('reaction time (sec)')
        else:
            raise RuntimeError('{} is not supported!'.format(item))

        for i in range(label_num):
            for j in range(i+1, label_num):
                print('subgroup{} vs. subgroup{} with {}:'.format(labels[i], labels[j], item),
                      ttest_ind(float_data_list[i], float_data_list[j]))

        # plt.savefig(pjoin(cluster_num_dir, '{}.png'.format(item)))

    # addition
    male_num_total = np.sum(male_nums)
    female_num_total = np.sum(female_nums)
    x = np.arange(1)
    width = auto_bar_width(x)
    plt.figure()
    ax = plt.gca()
    rects1 = ax.bar(x, [male_num_total], width, color='b')
    rects2 = ax.bar(x + width, [female_num_total], width, color='r')
    show_bar_value(rects1)
    show_bar_value(rects2)
    ax.legend((rects1, rects2), ('male', 'female'))
    ax.set_xticks(x + width / 2.0)
    ax.set_xticklabels(['1080 group'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('1080 Gender')
    ax.set_ylabel('count')
    # plt.savefig(pjoin(cluster_num_dir, '1080_Gender.png'))

    x = np.arange(label_num)
    width = auto_bar_width(x, 2)
    plt.figure()
    ax = plt.gca()
    male_nums = np.array(male_nums) / male_num_total
    female_nums = np.array(female_nums) / female_num_total
    rects1 = ax.bar(x, male_nums, width, color='b')
    rects2 = ax.bar(x + width, female_nums, width, color='r')
    show_bar_value(rects1, '.2%')
    show_bar_value(rects2, '.2%')
    ax.legend((rects1[0], rects2[1]), ('male', 'female'))
    ax.set_xticks(x + width / 2.0)
    ax.set_xticklabels(labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Gender')
    ax.set_xlabel('subgroup label')
    ax.set_ylabel('percent')
    # plt.savefig(pjoin(cluster_num_dir, 'Gender_percent.png'))

    acc_item = 'WM_Task_2bk_Face_Acc'
    rt_item = 'WM_Task_2bk_Face_Median_RT'
    title = '{}/{}'.format(acc_item, rt_item)
    x = np.arange(label_num)
    width = auto_bar_width(x)
    plt.figure()
    ax = plt.gca()
    acc_data_list = [float_data / 100.0 for float_data in float_data_dict[acc_item]]
    rt_data_list = [float_data / 1000.0 for float_data in float_data_dict[rt_item]]
    float_data_list = [acc / rt for acc, rt in zip(acc_data_list, rt_data_list)]
    y = [np.mean(float_data) for float_data in float_data_list]
    sems = [sem(float_data) for float_data in float_data_list]
    rects = ax.bar(x, y, width, color='b', yerr=sems, ecolor='blue', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title)
    ax.set_xlabel('subgroup label')
    for i in range(label_num):
        for j in range(i + 1, label_num):
            print('subgroup{} vs. subgroup{} with {}:'.format(labels[i], labels[j], title),
                  ttest_ind(float_data_list[i], float_data_list[j]))

    plt.show()
