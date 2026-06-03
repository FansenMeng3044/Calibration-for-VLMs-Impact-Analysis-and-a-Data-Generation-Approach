"""
Minimal point-cloud helpers for datasets that import from here.

Full LAVIS also ships ULIP processors and depends on lavis.models.ulip_models;
this tree only needs farthest_point_sample / pc_normalize for ModelNet-style
builders to import. See salesforce/LAVIS lavis/processors/ulip_processors.py.
"""

import numpy as np


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    point: pointcloud [N, D]
    npoint: number of samples
    returns: sampled pointcloud [npoint, D]
    """
    N, _ = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return point[centroids.astype(np.int32)]
