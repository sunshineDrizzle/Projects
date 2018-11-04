import numpy as np

from os.path import join as pjoin
from matplotlib import pyplot as plt
from commontool.io.io import CsvReader


# predefine some variates
# -----------------------
# predefine parameters
# rois = ['r1_FFA_m', 'r2_FFA_p', 'r2_FFA_a', 'r3_FFA_p', 'r3_FFA_a']
rois = ['l1_FFA_m', 'l2_FFA_p', 'l2_FFA_a', 'l3_FFA_p', 'l3_FFA_a']
roi2color = {'r1_FFA_m': 'k', 'l1_FFA_m': 'k',
             'r2_FFA_p': 'r', 'l2_FFA_p': 'r',
             'r2_FFA_a': 'y', 'l2_FFA_a': 'y',
             'r3_FFA_p': 'b', 'l3_FFA_p': 'b',
             'r3_FFA_a': 'g', 'l3_FFA_a': 'g'}
color2facecolor = {'b': 'blue',
                   'r': 'red',
                   'g': 'green',
                   'y': 'yellow',
                   'k': 'black'}

# predefine paths
project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
n_clusters_dir = pjoin(project_dir, '2mm_KM_init10_regress_left/3clusters')
fingerprint_files = pjoin(n_clusters_dir, '{}_func_fingerprint.csv')
# -----------------------


def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq'] / sum(aov['sum_sq'])
    return aov


def omega_squared(aov):
    mse = aov['sum_sq'][-1] / aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq'] - (aov[:-1]['df'] * mse)) / (sum(aov['sum_sq']) + mse)
    return aov


def curve_plot(show_errbar=False):
    fig, ax = plt.subplots()
    is_first_loop = True
    for roi in rois:
        fingerprint_file = fingerprint_files.format(roi)
        reader = CsvReader(fingerprint_file)
        fingerprints = np.array(reader.rows[1:], dtype=np.float64)

        x = np.arange(fingerprints.shape[1])
        fingerprints_mean = np.mean(fingerprints, axis=0)
        ax.plot(x, fingerprints_mean, '{}.-'.format(roi2color[roi]), label=roi)
        if show_errbar:
            fingerprints_std = np.std(fingerprints, axis=0)
            ax.fill_between(x, fingerprints_mean - fingerprints_std, fingerprints_mean + fingerprints_std,
                            alpha=0.5, facecolors=color2facecolor[roi2color[roi]])
        ax.legend()

        if is_first_loop:
            is_first_loop = False
            xticklabels = np.array(reader.rows[0])
            plt.xticks(x, xticklabels)
            # plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # fingerprint_file_1080 = '/nfs/t3/workingshop/chenxiayu/study/FFA_clustering/data/HCP_face-avg/label/' \
    #                         'lFFA_2mm_func_fingerprint.csv'
    # reader_1080 = CsvReader(fingerprint_file_1080)
    # fingerprints_1080 = np.array(reader_1080.rows[1:], dtype=np.float64)
    # x_1080 = np.arange(fingerprints_1080.shape[1])
    # fingerprints_1080_mean = np.mean(fingerprints_1080, axis=0)
    # ax.plot(x_1080, fingerprints_1080_mean, 'm*-', label='lFFA_2mm')
    # if show_errbar:
    #     fingerprints_1080_std = np.std(fingerprints_1080, axis=0)
    #     ax.fill_between(x_1080, fingerprints_1080_mean - fingerprints_1080_std,
    #                     fingerprints_1080_mean + fingerprints_1080_std,
    #                     alpha=0.5, facecolors='fuchsia')

    plt.tight_layout()
    plt.show()


def mds_plot():
    from sklearn.manifold import MDS
    from collections import OrderedDict

    only_mean = True

    data = OrderedDict()
    for roi in rois:
        fingerprint_file = fingerprint_files.format(roi)
        reader = CsvReader(fingerprint_file)
        fingerprints = np.array(reader.rows[1:], dtype=np.float64)
        fingerprints_mean = np.atleast_2d(np.mean(fingerprints, axis=0))
        if only_mean:
            data[roi] = fingerprints_mean
        else:
            data[roi] = np.r_[fingerprints_mean, fingerprints]

    X = np.zeros((0, list(data.values())[0].shape[1]))
    offsets = []
    counts = []
    for roi, fp in data.items():
        offsets.append(X.shape[0])
        counts.append(fp.shape[0])
        X = np.r_[X, fp]
    print(counts)

    embedding = MDS()
    X_transformed = embedding.fit_transform(X)

    fig, ax = plt.subplots()
    for idx, roi in enumerate(rois):
        positions = X_transformed[offsets[idx]:offsets[idx]+counts[idx]]
        ax.scatter(positions[0, 0], positions[0, 1], c=roi2color[roi], label=roi+'_mean', s=30)
        if positions.shape[0] > 1:
            ax.scatter(positions[1:, 0], positions[1:, 1], c=roi2color[roi], label=roi, s=1)
    ax.legend()
    ax.tick_params(bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    plt.show()


def anova():
    import pandas as pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    columns = ['z_stat', 'roi', 'cope']
    pd_dict = dict()
    for c in columns:
        pd_dict[c] = []
    for roi in rois:
        fingerprint_file = fingerprint_files.format(roi)
        csv_reader = CsvReader(fingerprint_file)
        csv_dict = csv_reader.to_dict()
        for cope, z_stats in csv_dict.items():
            pd_dict['roi'].extend([roi] * len(z_stats))
            pd_dict['cope'].extend([cope] * len(z_stats))
            pd_dict['z_stat'].extend(map(float, z_stats))
    data = pd.DataFrame(pd_dict, columns=columns)

    formula = 'z_stat ~ C(roi) + C(cope) + C(roi):C(cope)'
    model = ols(formula, data).fit()
    aov_table = anova_lm(model, typ=2)
    eta_squared(aov_table)
    omega_squared(aov_table)
    print(aov_table)


if __name__ == '__main__':
    curve_plot()
    mds_plot()
    anova()
