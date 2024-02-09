import numpy as np
import open3d as o3d


def transform_pcd(pcd, T):

    pcd = np.concatenate([pcd, np.ones_like(pcd)[..., 0: 1]], axis=-1)
    pcd = pcd @ T.T
    pcd = pcd[..., :-1]
    return pcd


def main():

    source_path = "real_pcds.npy"
    target_path = "rlbench_pcd.npy"

    source_pcd = np.load(source_path)
    target_pcd = np.load(target_path)

    T = np.array([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 1.2],
        [0., 0., 0., 1.]
    ])
    source_pcd = transform_pcd(source_pcd, T)

    source_pcd_o3d = o3d.geometry.PointCloud()
    source_pcd_o3d.points = o3d.utility.Vector3dVector(source_pcd)
    red = np.zeros_like(source_pcd)
    red[:, 0] = 1.
    source_pcd_o3d.colors = o3d.utility.Vector3dVector(red)

    target_pcd_o3d = o3d.geometry.PointCloud()
    target_pcd_o3d.points = o3d.utility.Vector3dVector(target_pcd)
    blue = np.zeros_like(target_pcd)
    blue[:, 2] = 1.
    target_pcd_o3d.colors = o3d.utility.Vector3dVector(blue)

    origin_frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries([source_pcd_o3d, target_pcd_o3d, origin_frame])


main()
