import os
import torch
import open3d as o3d


def main():
    # PLY 파일 경로
    ply_path = os.path.join(
        "data",
        "hm3dsem_scene_graphs",
        "00862-LT9Jq6dN3Ea",
        # "00890-6s7QHgap2fW",
        "full_pcd.ply"
    )

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    print(f"Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)

    print(pcd)  # 포인트 개수 등 정보 출력
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()



# full_feats = torch.load("./data/hm3dsem_scene_graphs/00890-6s7QHgap2fW/full_feats.pt")
# print("full_feats.shape:", full_feats.shape)

# mask_feats = torch.load("./data/hm3dsem_scene_graphs/00890-6s7QHgap2fW/mask_feats.pt")
# print("mask_feats.shape:", mask_feats.shape)

# import os
# import glob
# import open3d as o3d


# import os
# import glob
# import open3d as o3d


# def visualize_objects():
#     base_dir = "/Users/sang-yun/Desktop/ConceptLoc/HOV-SG-main/data/Replica_sem_seg/room1/replica/objects"
#     ply_paths = sorted(glob.glob(os.path.join(base_dir, "*.ply")))

#     if not ply_paths:
#         raise FileNotFoundError(f"No .ply files found in: {base_dir}")

#     print(f"Found {len(ply_paths)} .ply files in {base_dir}")
#     print("뷰어 창에서 키보드 q 를 누르면 종료합니다.")

#     vis = o3d.visualization.VisualizerWithKeyCallback()
#     vis.create_window(window_name="objects viewer")

#     state = {"idx": 0, "quit": False}

#     def load_current():
#         vis.clear_geometries()
#         ply_path = ply_paths[state["idx"]]
#         print(f"[{state['idx']+1}/{len(ply_paths)}] {ply_path}")
#         pcd = o3d.io.read_point_cloud(ply_path)
#         vis.add_geometry(pcd)
#         vis.reset_view_point(True)

#     # 다음 객체: n
#     def next_obj(vis_):
#         if state["idx"] < len(ply_paths) - 1:
#             state["idx"] += 1
#             load_current()
#         return False  # 계속 루프

#     # 이전 객체: b
#     def prev_obj(vis_):
#         if state["idx"] > 0:
#             state["idx"] -= 1
#             load_current()
#         return False

#     # 종료: q
#     def quit_cb(vis_):
#         state["quit"] = True
#         vis_.close()
#         return True  # 창 닫기

#     # 키 콜백 등록
#     vis.register_key_callback(ord("N"), next_obj)  # N 키
#     vis.register_key_callback(ord("n"), next_obj)
#     vis.register_key_callback(ord("B"), prev_obj)  # B 키
#     vis.register_key_callback(ord("b"), prev_obj)
#     vis.register_key_callback(ord("Q"), quit_cb)   # Q 키
#     vis.register_key_callback(ord("q"), quit_cb)

#     load_current()
#     vis.run()
#     vis.destroy_window()


# if __name__ == "__main__":
#     visualize_objects()