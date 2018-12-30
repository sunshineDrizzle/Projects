import numpy as np

from scipy.stats import ttest_ind, sem
from matplotlib import pyplot as plt
from commontool.io.io import CsvReader
from commontool.algorithm.plot import show_bar_value, auto_bar_width


def roi_mean_plot(roi_mean_file, ROIitems, colors, xticklabels, ylabel=None, title=None, plot_style='violin'):
    roi_mean_dict = CsvReader(roi_mean_file).to_dict(axis=1)
    roi_means_list = [list(map(float, roi_mean_dict[ROIitem])) for ROIitem in ROIitems]

    ROIitem_num = len(ROIitems)
    for i in range(ROIitem_num):
        for j in range(i+1, ROIitem_num):
            print('{} vs. {}'.format(ROIitems[i], ROIitems[j]),
                  ttest_ind(roi_means_list[i], roi_means_list[j]))

    plt.figure()
    if plot_style == 'violin':
        violin_parts = plt.violinplot(roi_means_list, showmeans=True)
        for idx, pc in enumerate(violin_parts['bodies']):
            # https://stackoverflow.com/questions/26291479/changing-the-color-of-matplotlibs-violin-plots
            pc.set_color(colors[idx])
        plt.xticks(range(1, ROIitem_num + 1), xticklabels)
    elif plot_style == 'bar':
        x = np.arange(ROIitem_num)
        y = [np.mean(roi_means) for roi_means in roi_means_list]
        sems = [sem(roi_means) for roi_means in roi_means_list]
        width = auto_bar_width(x)
        rects = plt.bar(x, y, width, edgecolor=colors[0], yerr=sems, facecolor='white')
        show_bar_value(rects, '.2f')
        plt.xticks(x, xticklabels)
    else:
        raise RuntimeError("Invalid plot style: {}".format(plot_style))
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()


def gender_diff_roi_mean_plot(roi_mean_file, items_m, items_f, xticklabels, ylabel=None, title=None):

    assert len(items_m) == len(items_f)

    roi_mean_dict = CsvReader(roi_mean_file).to_dict(axis=1)
    roi_means_list_m = [list(map(float, roi_mean_dict[item])) for item in items_m]
    roi_means_list_f = [list(map(float, roi_mean_dict[item])) for item in items_f]

    item_num = len(items_m)
    for i in range(item_num):
        print('{} vs. {}'.format(items_m[i], items_f[i]),
              ttest_ind(roi_means_list_m[i], roi_means_list_f[i]))

    fig, ax = plt.subplots()
    x = np.arange(item_num)
    width = auto_bar_width(x, 2)
    y_m = [np.mean(roi_means) for roi_means in roi_means_list_m]
    y_f = [np.mean(roi_means) for roi_means in roi_means_list_f]
    sems_m = [sem(roi_means) for roi_means in roi_means_list_m]
    sems_f = [sem(roi_means) for roi_means in roi_means_list_f]
    rects1 = ax.bar(x, y_m, width, color='b', alpha=0.5, yerr=sems_m, ecolor='blue')
    rects2 = ax.bar(x + width, y_f, width, color='r', alpha=0.5, yerr=sems_f, ecolor='red')
    # show_bar_value(rects1, '.3f')
    # show_bar_value(rects2, '.3f')
    ax.legend((rects1, rects2), ('male', 'female'))
    ax.set_xticks(x + width / 2.0)
    ax.set_xticklabels(xticklabels)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.tight_layout()


if __name__ == '__main__':
    from os.path import join as pjoin

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    roi_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/roi_analysis')

    # ROIitems = ['l1_FFA1', 'l2_FFA1', 'l1_FFA2', 'r1_FFA1', 'r2_FFA1', 'r1_FFA2']
    # xticklabels = ROIitems
    # colors = ['blue', 'red', 'blue'] * 2
    # roi_mean_plot(pjoin(roi_dir, 'roi_mean_face-avg_intrasubgroup'), ROIitems, colors, xticklabels, 'face-avg')

    # ROIitems = ['r1_FFA1_in_subgroup1', 'r2_FFA1_in_subgroup1', 'r1_FFA2_in_subgroup1']
    # xticklabels = [item[:7] for item in ROIitems]
    # colors = ['black'] * 3
    # roi_mean_plot(pjoin(roi_dir, 'roi_mean_face-avg_allsubgroup'), ROIitems, colors, xticklabels,
    #               'face-avg', 'values in subgroup1', 'bar')
    # roi_mean_plot(pjoin(roi_dir, 'roi_mean_mean_bold_signal_allsubgroup'), ROIitems, colors, xticklabels,
    #               'mean_bold_signal', 'values in subgroup1', 'bar')

    items_m = ['l1_FFA1_in_group1_male', 'l2_FFA1_in_group2_male', 'l1_FFA2_in_group1_male',
               'r1_FFA1_in_group1_male', 'r2_FFA1_in_group2_male', 'r1_FFA2_in_group1_male']
    items_f = ['l1_FFA1_in_group1_female', 'l2_FFA1_in_group2_female', 'l1_FFA2_in_group1_female',
               'r1_FFA1_in_group1_female', 'r2_FFA1_in_group2_female', 'r1_FFA2_in_group1_female']
    xticklabels = [item[:7] for item in items_m]
    gender_diff_roi_mean_plot(pjoin(roi_dir, 'gender_diff_top_acti_FFA_percent10_mean_face-avg_allsubgroup'),
                              items_m, items_f, xticklabels, 'face-avg', 'intrasubgroup')

    items_m = ['l1_FFA1_in_group2_male', 'l2_FFA1_in_group1_male', 'l1_FFA2_in_group2_male',
               'r1_FFA1_in_group2_male', 'r2_FFA1_in_group1_male', 'r1_FFA2_in_group2_male']
    items_f = ['l1_FFA1_in_group2_female', 'l2_FFA1_in_group1_female', 'l1_FFA2_in_group2_female',
               'r1_FFA1_in_group2_female', 'r2_FFA1_in_group1_female', 'r1_FFA2_in_group2_female']
    xticklabels = [item[:7] for item in items_m]
    gender_diff_roi_mean_plot(pjoin(roi_dir, 'gender_diff_top_acti_FFA_percent10_mean_face-avg_allsubgroup'),
                              items_m, items_f, xticklabels, 'face-avg', 'intersubgroup')

    plt.show()
