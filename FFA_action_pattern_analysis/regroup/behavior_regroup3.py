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

    return list(map(float, data))


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
    mean = np.mean(data)

    print('The mean {} of Cluster{}:'.format(name, label), mean,
          '(valid subject number: {})'.format(len(data)))
    return mean


def explore_Gender(behavior_dict, label):
    male_num = behavior_dict['Gender'].count('M')
    female_num = behavior_dict['Gender'].count('F')

    print('The number of males in Cluster{}:'.format(label), male_num)
    print('The number of females in Cluster{}:'.format(label), female_num)
    return male_num, female_num


def explore_WM_Task_Acc(behavior_dict, label):
    wm_task_acc = list(behavior_dict['WM_Task_Acc'])
    while '' in wm_task_acc:
        wm_task_acc.remove('')
    wm_task_acc = list(map(float, wm_task_acc))
    print('The mean WM_Task_Acc of Cluster{}:'.format(label), np.mean(wm_task_acc),
          '(valid subject number: {})'.format(len(wm_task_acc)))


def explore_WM_Task_2bk_Acc(behavior_dict, label):
    wm_task_2bk_acc = list(behavior_dict['WM_Task_2bk_Acc'])
    while '' in wm_task_2bk_acc:
        wm_task_2bk_acc.remove('')
    wm_task_2bk_acc = list(map(float, wm_task_2bk_acc))

    print('The mean WM_Task_2bk_Acc of Cluster{}:'.format(label), np.mean(wm_task_2bk_acc),
          '(valid subject number: {})'.format(len(wm_task_2bk_acc)))


def explore_WM_Task_0bk_Acc(behavior_dict, label):
    wm_task_0bk_acc = list(behavior_dict['WM_Task_0bk_Acc'])
    while '' in wm_task_0bk_acc:
        wm_task_0bk_acc.remove('')
    wm_task_0bk_acc = list(map(float, wm_task_0bk_acc))

    print('The mean WM_Task_0bk_Acc of Cluster{}:'.format(label), np.mean(wm_task_0bk_acc),
          '(valid subject number: {})'.format(len(wm_task_0bk_acc)))


def explore_WM_Task_Median_RT(behavior_dict, label):
    wm_task_median_rt = list(behavior_dict['WM_Task_Median_RT'])
    while '' in wm_task_median_rt:
        wm_task_median_rt.remove('')
    wm_task_median_rt = list(map(float, wm_task_median_rt))

    print('The mean WM_Task_Median_RT of Cluster{}:'.format(label), np.mean(wm_task_median_rt),
          '(valid subject number: {})'.format(len(wm_task_median_rt)))


def explore_WM_Task_2bk_Median_RT(behavior_dict, label):
    wm_task_2bk_median_rt = list(behavior_dict['WM_Task_2bk_Median_RT'])
    while '' in wm_task_2bk_median_rt:
        wm_task_2bk_median_rt.remove('')
    wm_task_2bk_median_rt = list(map(float, wm_task_2bk_median_rt))

    print('The mean WM_Task_2bk_Median_RT of Cluster{}:'.format(label), np.mean(wm_task_2bk_median_rt),
          '(valid subject number: {})'.format(len(wm_task_2bk_median_rt)))


def explore_WM_Task_0bk_Median_RT(behavior_dict, label):
    wm_task_0bk_median_rt = list(behavior_dict['WM_Task_0bk_Median_RT'])
    while '' in wm_task_0bk_median_rt:
        wm_task_0bk_median_rt.remove('')
    wm_task_0bk_median_rt = list(map(float, wm_task_0bk_median_rt))

    print('The mean WM_Task_0bk_Median_RT of Cluster{}:'.format(label), np.mean(wm_task_0bk_median_rt),
          '(valid subject number: {})'.format(len(wm_task_0bk_median_rt)))


def explore_WM_Task_0bk_Face_Acc(behavior_dict, label):
    wm_task_0bk_face_acc = list(behavior_dict['WM_Task_0bk_Face_Acc'])
    while '' in wm_task_0bk_face_acc:
        wm_task_0bk_face_acc.remove('')
    wm_task_0bk_face_acc = list(map(float, wm_task_0bk_face_acc))

    print('The mean WM_Task_0bk_Face_Acc of Cluster{}:'.format(label), np.mean(wm_task_0bk_face_acc),
          '(valid subject number: {})'.format(len(wm_task_0bk_face_acc)))


def explore_WM_Task_2bk_Face_Acc(behavior_dict, label):
    pass


if __name__ == '__main__':
    from os.path import join as pjoin
    from matplotlib import pyplot as plt
    from commontool.algorithm.plot import auto_bar_width, show_bar_value

    # predefine some variates
    n_clusters = 20
    working_dir = '/nfs/s2/userhome/chenxiayu/workingdir'
    project_dir = pjoin(working_dir, 'study/rFFA_clustering')
    subproject_dir = pjoin(project_dir, '2mm_ward_regress')
    n_clusters_dir = pjoin(subproject_dir, '{}clusters'.format(n_clusters))
    result_dir = pjoin(n_clusters_dir, 'regroup/1_or_2')
    regroup_num = 2

    with open(pjoin(project_dir, 'data/S1200_behavior.csv')) as f:
        lines = f.read().splitlines()
    behavior_names = lines[0].split(',')[1:]
    subject_dict = {}
    for line in lines[1:]:
        subject_id = line.split(',')[0]
        subject_dict[subject_id] = line.split(',')[1:]

    male_nums = []
    female_nums = []
    float_items = ['WM_Task_Acc', 'WM_Task_0bk_Acc', 'WM_Task_2bk_Acc',
                   'WM_Task_0bk_Face_Acc', 'WM_Task_2bk_Face_Acc',
                   'WM_Task_Median_RT', 'WM_Task_0bk_Median_RT', 'WM_Task_2bk_Median_RT',
                   'WM_Task_0bk_Face_Median_RT', 'WM_Task_2bk_Face_Median_RT']
    float_mean_dict = {}
    for item in float_items:
        float_mean_dict[item] = []

    for label in range(1, regroup_num+1):
        with open(pjoin(result_dir, 'regroup{}_id'.format(label))) as f:
            subgroup_ids = f.read().splitlines()

        behavior_dict = get_behavior_dict(subject_dict, behavior_names, subgroup_ids)

        male_num, female_num = explore_Gender(behavior_dict, label)
        male_nums.append(male_num)
        female_nums.append(female_num)

        for item in float_items:
            float_mean_dict[item].append(explore_float_data(behavior_dict, item, label))

    # plot
    x = np.arange(regroup_num)
    width = auto_bar_width(x, 2)
    plt.figure()
    ax = plt.gca()
    rects1 = ax.bar(x, male_nums, width, color='b')
    rects2 = ax.bar(x + width, female_nums, width, color='r')
    show_bar_value(rects1)
    show_bar_value(rects2)
    ax.legend((rects1[0], rects2[1]), ('male', 'female'))
    ax.set_xticks(x + width / 2.0)
    ax.set_xticklabels(x + 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Gender')
    ax.set_xlabel('subgroup label')
    ax.set_ylabel('count')
    plt.savefig(pjoin(result_dir, 'Gender.png'))

    width = auto_bar_width(x)
    for item in float_items:
        plt.figure()
        ax = plt.gca()
        if 'Acc' in item:
            y = [i/100.0 for i in float_mean_dict[item]]
        elif 'RT' in item:
            y = float_mean_dict[item]
        else:
            raise RuntimeError('{} is not supported!'.format(item))
        rects = ax.bar(x, y, width, color='b')
        ax.set_xticks(x)
        ax.set_xticklabels(x + 1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_title(item)
        ax.set_xlabel('subgroup label')
        if 'Acc' in item:
            show_bar_value(rects, '.2%')
            ax.set_ylabel('accuracy')
        elif 'RT' in item:
            show_bar_value(rects, '.2f')
            ax.set_ylabel('reaction time (mm)')
        else:
            raise RuntimeError('{} is not supported!'.format(item))

        plt.savefig(pjoin(result_dir, '{}.png'.format(item)))

    # addition
    # behavior_dict_full = get_behavior_dict(subject_dict, behavior_names)
    # male_num_total = behavior_dict_full['Gender'].count('M')
    # female_num_total = behavior_dict_full['Gender'].count('F')
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
    plt.savefig(pjoin(result_dir, '1080_Gender.png'))

    x = np.arange(regroup_num)
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
    ax.set_xticklabels(x + 1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Gender')
    ax.set_xlabel('subgroup label')
    ax.set_ylabel('percent')
    plt.savefig(pjoin(result_dir, 'Gender_percent.png'))

    plt.show()
