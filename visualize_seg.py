import os
import open3d as o3d


def main():
    # PLY 파일 경로
    # ply_path = os.path.join(
    #     "data",
    #     "hm3dsem_scene_graphs",
    #     "hm3dsem",
    #     "00862-LT9Jq6dN3Ea",
    #     # "00890-6s7QHgap2fW",
    #     # "full_pcd.ply",
    #     "masked_pcd.ply"
    # )
    ply_path = os.path.join(
        "data",
        "Replica_seg",
        "room1",
        # "full_pcd.ply",
        "masked_pcd.ply"
    )

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    print(f"Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    # 포인트 개수 등 정보 출력
    print(pcd)

    # O3DVisualizer를 사용한 시각화
    o3d.visualization.gui.Application.instance.initialize()
    vis = o3d.visualization.O3DVisualizer("PointCloud Viewer", 1024, 768)
    vis.add_geometry("pcd", pcd)

    app = o3d.visualization.gui.Application.instance
    app.add_window(vis)
    app.run()
    
if __name__ == "__main__":
    main()