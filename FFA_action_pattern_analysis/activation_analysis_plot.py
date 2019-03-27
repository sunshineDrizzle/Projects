import os
import numpy as np

from os.path import join as pjoin
from matplotlib import pyplot as plt
from statsmodels.stats.multitest import multipletests
from commontool.io.io import CsvReader
from commontool.algorithm.statistics import calc_mean_sem, ttest_ind_pairwise, plot_compare, plot_mean_sem

project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
acti_analysis_dir = pjoin(project_dir, 's2_25_zscore/HAC_ward_euclidean/2clusters/activation/acti_of_FSR')

# column_name_file = pjoin(acti_analysis_dir, 'npy_column_name')
# column_names = open(column_name_file).read().split(',')
# data1_1 = np.load(pjoin(acti_analysis_dir, 'g1_intra_pattern_similarity.npy'))
# data1_2 = np.load(pjoin(acti_analysis_dir, 'g2_intra_pattern_similarity.npy'))
# data1 = np.r_[data1_1, data1_2]
# data2 = np.load(pjoin(acti_analysis_dir, 'g1_and_g2_inter_pattern_similarity.npy'))
# samples1 = []
# samples2 = []
# sample_names = []
# for idx, sample_name in enumerate(column_names):
#     if 'FFA' in sample_name and 'mask' not in sample_name:
#         continue
#     else:
#         samples1.append(data1[:, idx])
#         samples2.append(data2[:, idx])
#         sample_names.append(sample_name)

# ---mean sem---
# mean_sem_dir = pjoin(acti_analysis_dir, 'mean_sem')
# if not os.path.exists(mean_sem_dir):
#     os.makedirs(mean_sem_dir)
# mean_sem_file1 = pjoin(mean_sem_dir, 'g1_and_g2_intra_pattern_similarity')
# mean_sem_file2 = pjoin(mean_sem_dir, 'g1_and_g2_inter_pattern_similarity')
# calc_mean_sem(samples1, mean_sem_file1, sample_names.copy())
# calc_mean_sem(samples2, mean_sem_file2, sample_names.copy())
# plot_mean_sem([mean_sem_file1, mean_sem_file2], ['intra', 'inter'], ylabel='pattern similarity')
# ---mean sem---

# ---compare---
compare_dir = pjoin(acti_analysis_dir, 'compare')
if not os.path.exists(compare_dir):
    os.makedirs(compare_dir)
compare_file = pjoin(compare_dir, 'intra_vs_inter_pattern_similarity')
# ttest_ind_pairwise(samples1, samples2, compare_file, sample_names.copy())

multi_test_corrected = False
alpha = 1.1
compare_dict = CsvReader(compare_file).to_dict(1)
ps = np.array(list(map(float, compare_dict['p'])))
if multi_test_corrected:
    reject, ps, alpha_sidak, alpha_bonf = multipletests(ps, 0.05, 'fdr_bh')
sample_names = [name for idx, name in enumerate(compare_dict['sample_name']) if ps[idx] < alpha]
ps = [p for p in ps if p < alpha]
print('\n'.join(list(map(str, zip(sample_names, ps)))))
plot_compare(ps, sample_names, title=os.path.basename(compare_file))
# ---compare---

plt.show()
