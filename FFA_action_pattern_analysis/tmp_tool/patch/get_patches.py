from community import best_partition
from commontool.algorithm.graph import connectivity_grow


def get_patch_by_crg(vertices, edge_list):
    patches = []
    while vertices:
        seed = vertices.pop()
        patch = connectivity_grow([[seed]], edge_list)[0]
        patches.append(list(patch))
        vertices.difference_update(patch)

    return patches


def get_patch_by_LV(graph):
    partition = best_partition(graph)
    patches_dict = dict()
    for label in partition.values():
        patches_dict[label] = []
    for vtx, label in partition.items():
        patches_dict[label].append(vtx)
    patches = [patches_dict[label] for label in sorted(patches_dict.keys())]

    return patches


if __name__ == '__main__':
    import os
    import numpy as np
    import nibabel as nib

    from networkx import Graph
    from os.path import join as pjoin
    from scipy.spatial.distance import pdist
    from commontool.io.io import CiftiReader, GiftiReader, save2nifti
    from commontool.algorithm.triangular_mesh import get_n_ring_neighbor

    threshold = 2.3
    method = 'crg'  # crg, LV_weighted, LV_unweighted

    project_dir = '/nfs/s2/userhome/chenxiayu/workingdir/study/FFA_clustering'
    lFFA_label_file = pjoin(project_dir, 'data/HCP_face-avg/label/lFFA_2mm_15.label')
    rFFA_label_file = pjoin(project_dir, 'data/HCP_face-avg/label/rFFA_2mm_15.label')
    maps_file = pjoin(project_dir, 'data/HCP_face-avg/s2/S1200.1080.FACE-AVG_level2_zstat_hp200_s2_MSMAll.dscalar.nii')
    subject_ids_file = pjoin(project_dir, 'data/HCP_face-avg/s2/subject_id')
    patch_dir = pjoin(project_dir, 'data/HCP_face-avg/s2/patches_15/{}{}'.format(method, threshold))
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    lFFA_vertices = nib.freesurfer.read_label(lFFA_label_file)
    lFFA_vertices = set(lFFA_vertices.astype(np.uint16))
    rFFA_vertices = nib.freesurfer.read_label(rFFA_label_file)
    rFFA_vertices = set(rFFA_vertices.astype(np.uint16))
    c_reader = CiftiReader(maps_file)
    lmaps = c_reader.get_data('CIFTI_STRUCTURE_CORTEX_LEFT', True)
    rmaps = c_reader.get_data('CIFTI_STRUCTURE_CORTEX_RIGHT', True)
    with open(subject_ids_file) as rf:
        subject_ids = rf.read().splitlines()

    g_reader_l = GiftiReader('/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/'
                             'HCP_S1200_GroupAvg_v1/S1200.L.white_MSMAll.32k_fs_LR.surf.gii')
    g_reader_r = GiftiReader('/nfs/p1/public_dataset/datasets/hcp/DATA/HCP_S1200_GroupAvg_v1/'
                             'HCP_S1200_GroupAvg_v1/S1200.R.white_MSMAll.32k_fs_LR.surf.gii')

    lFFA_patch_maps = np.zeros_like(lmaps)
    lFFA_patch_stats = []
    for idx, lmap in enumerate(lmaps):
        patch_stat = [subject_ids[idx]]
        vertices_thr = np.where(lmap > threshold)[0]
        vertices_thr_FFA = lFFA_vertices.intersection(vertices_thr)
        mask = np.zeros(g_reader_l.coords.shape[0])
        mask[list(vertices_thr_FFA)] = 1
        edge_list = get_n_ring_neighbor(g_reader_l.faces, mask=mask)

        if method == 'crg':
            patches = get_patch_by_crg(vertices_thr_FFA, edge_list)
        elif 'LV' in method:
            graph = Graph()
            graph.add_nodes_from(vertices_thr_FFA)
            edges = [(vtx, vtx_neighbor) for vtx in vertices_thr_FFA for vtx_neighbor in edge_list[vtx]]
            w = method.split('_')[1]
            if w == 'weighted':
                edge_data = [pdist(np.array([[lmap[i]], [lmap[j]]]))[0] for i, j in edges]
                max_dissimilar = np.max(edge_data)
                min_dissimilar = np.min(edge_data)
                edge_data = [(max_dissimilar - dist) / (max_dissimilar - min_dissimilar) for dist in edge_data]
                edges = np.array(edges)
                graph.add_weighted_edges_from(zip(edges[:, 0], edges[:, 1], edge_data))
            elif w == 'unweighted':
                graph.add_edges_from(edges)
            else:
                raise RuntimeError("invalid method: {}".format(method))
            patches = get_patch_by_LV(graph)
        else:
            raise RuntimeError("the method - {} is not supported at present!".format(method))

        for label, patch in enumerate(patches, 1):
            lFFA_patch_maps[idx, patch] = label
        patch_stat.append(str(len(patches)))
        patch_stat.extend([str(len(patch)) for patch in patches])
        lFFA_patch_stats.append(','.join(patch_stat))

        print('{}/{}'.format(idx+1, lmaps.shape[0]))

    rFFA_patch_maps = np.zeros_like(rmaps)
    rFFA_patch_stats = []
    for idx, rmap in enumerate(rmaps):
        patch_stat = [subject_ids[idx]]
        vertices_thr = np.where(rmap > threshold)[0]
        vertices_thr_FFA = rFFA_vertices.intersection(vertices_thr)
        mask = np.zeros(g_reader_r.coords.shape[0])
        mask[list(vertices_thr_FFA)] = 1
        edge_list = get_n_ring_neighbor(g_reader_r.faces, mask=mask)

        if method == 'crg':
            patches = get_patch_by_crg(vertices_thr_FFA, edge_list)
        elif 'LV' in method:
            from networkx import Graph
            from scipy.spatial.distance import pdist

            graph = Graph()
            graph.add_nodes_from(vertices_thr_FFA)
            edges = [(vtx, vtx_neighbor) for vtx in vertices_thr_FFA for vtx_neighbor in edge_list[vtx]]
            w = method.split('_')[1]
            if w == 'weighted':
                edge_data = [pdist(np.array([[rmap[i]], [rmap[j]]]))[0] for i, j in edges]
                max_dissimilar = np.max(edge_data)
                min_dissimilar = np.min(edge_data)
                edge_data = [(max_dissimilar - dist) / (max_dissimilar - min_dissimilar) for dist in edge_data]
                edges = np.array(edges)
                graph.add_weighted_edges_from(zip(edges[:, 0], edges[:, 1], edge_data))
            elif w == 'unweighted':
                graph.add_edges_from(edges)
            else:
                raise RuntimeError("invalid method: {}".format(method))
            patches = get_patch_by_LV(graph)
        else:
            raise RuntimeError("the method - {} is not supported at present!".format(method))

        for label, patch in enumerate(patches, 1):
            rFFA_patch_maps[idx, patch] = label
        patch_stat.append(str(len(patches)))
        patch_stat.extend([str(len(patch)) for patch in patches])
        rFFA_patch_stats.append(','.join(patch_stat))

        print('{}/{}'.format(idx + 1, rmaps.shape[0]))

    header = nib.Nifti2Header()
    header['descrip'] = 'FreeROI label'
    save2nifti(pjoin(patch_dir, 'lFFA_patch_maps.nii.gz'.format(threshold)), lFFA_patch_maps, header=header)
    save2nifti(pjoin(patch_dir, 'rFFA_patch_maps.nii.gz'.format(threshold)), rFFA_patch_maps, header=header)
    open(pjoin(patch_dir, 'lFFA_patch_stats'.format(threshold)), 'w+').writelines('\n'.join(lFFA_patch_stats))
    open(pjoin(patch_dir, 'rFFA_patch_stats'.format(threshold)), 'w+').writelines('\n'.join(rFFA_patch_stats))
