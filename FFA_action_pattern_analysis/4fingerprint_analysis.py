import numpy as np

from os.path import join as pjoin
from matplotlib import pyplot as plt
from commontool.io.io import CsvReader


# predefine some variates
# -----------------------
# predefine parameters
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

    X = []
    for roi in rois:
        fingerprint_file = fingerprint_files.format(roi)
        reader = CsvReader(fingerprint_file)
        fingerprints = np.array(reader.rows[1:], dtype=np.float64)
        X.append(np.mean(fingerprints, axis=0))

    X = np.array(X)
    embedding = MDS()
    X_transformed = embedding.fit_transform(X)

    fig, ax = plt.subplots()
    for idx, pos in enumerate(X_transformed):
        ax.scatter(pos[0], pos[1], c=roi2color[rois[idx]], label=rois[idx])
    ax.legend()
    ax.tick_params(bottom=False, left=False,
                   labelbottom=False, labelleft=False)
    plt.show()


def anova():
    pass


if __name__ == '__main__':
    mds_plot()
