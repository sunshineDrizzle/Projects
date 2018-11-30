if __name__ == '__main__':
    import os
    import random
    import numpy as np

    from os.path import join as pjoin

    random_num = 10
    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    cluster_num_dir = pjoin(project_dir, '2mm_KM_thr2.3_zscore/3clusters')
    subject_labels_file = pjoin(cluster_num_dir, 'subject_labels')
    subject_ids_file = pjoin(project_dir, 'data/HCP_face-avg/s2/subject_id')
    random_dir = pjoin(cluster_num_dir, 'random_from_subgroup')
    if not os.path.exists(random_dir):
        os.makedirs(random_dir)

    with open(subject_labels_file) as rf:
        subject_labels = np.array(rf.read().split(' '), dtype=np.uint16)
    with open(subject_ids_file) as rf:
        subject_ids = np.array(rf.read().splitlines())

    for label in sorted(set(subject_labels)):
        sub_subject_ids = subject_ids[subject_labels == label]
        subject_ids_selected = sorted(random.sample(sub_subject_ids.tolist(), random_num))
        with open(pjoin(random_dir, 'subject{}_id_random{}'.format(label, random_num)), 'w+') as wf:
            wf.writelines('\n'.join(subject_ids_selected))
