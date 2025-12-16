import os
import json
import glob
import numpy as np
from collections import defaultdict

import hydra
import open3d as o3d
import matplotlib.pyplot as plt
# 3D 시각화 라이브러리
import pyvista as pv
from tqdm import tqdm
# Hydra가 넘겨주는 Dict 객체 타입
from omegaconf import DictConfig



# # 각 객체에 해당하는 RGB 색상을 할당하는 함수
# def get_cmap(n, name="hsv"):
#     """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
#     RGB color; the keyword argument name must be a standard mpl colormap name."""
#     return plt.cm.get_cmap(name, n)

@hydra.main(version_base=None, config_path="./config", config_name="visualize_graph")
def main(params: DictConfig):
    # Initialize the PyVista plotter
    p = pv.Plotter()

    # Load paths to floor PLY files and corresponding JSON metadata
    floors_ply_paths = sorted(glob.glob(os.path.join(params.graph_path, "floors", "*.ply")))
    floors_info_paths = sorted(glob.glob(os.path.join(params.graph_path, "floors", "*.json")))

    # Initialize data structures for storing point clouds and metadata
    floor_pcds = {}
    floor_infos = {}
    # 계층 구조를 표현하기 위한 데이터 구조 (floor -> rooms -> object)
    hier_topo = defaultdict(dict)
    # 시각화 시 층을 구분하기 위한 평행 이동 값
    init_offset = np.array([7.0, 2.5, 4.0])  # Initial offset for visualization

    # Process each floor
    for counter, (ply_path, info_path) in enumerate(zip(floors_ply_paths, floors_info_paths)):
        with open(info_path, "r") as fp:
            floor_info = json.load(fp)
        # 딕셔너리 형태로 floor 정보를 저장
        # Store relevant floor metadata
        """
        예시
        floor_infos = {
            "0": {
                "floor_id": "0",
                "name": "floor_0",
                "rooms": ["0_0", "0_1"],
                "floor_height": 3.2,
                "floor_zero_level": 0.0,
                "vertices": [[...], ...],
                "viz_offset": [0.0, 0.0, 0.0] -> 이런식으로 추가 됨 (floor_id * init_offset)
            },
            "1": {
                "floor_id": "1",
                "name": "floor_1",
                "rooms": ["1_0"],
                "floor_height": 3.1,
                "floor_zero_level": 3.3,
                "vertices": [[...], ...],
                "viz_offset": [7.0, 2.5, 4.0] -> 이런식으로 추가 됨 (floor_id * init_offset)
            },
            ...
        }
        """
        floor_infos[floor_info["floor_id"]] = {
            k: v for k, v in floor_info.items() if k in ["floor_id", "name", "rooms", "floor_height", "floor_zero_level", "vertices"]
        }
        # Apply visualization offset to each floor
        floor_infos[floor_info["floor_id"]]["viz_offset"] = init_offset * counter
        for r_id in floor_info["rooms"]:
            hier_topo[floor_info["floor_id"]][r_id] = []

        # floor point cloud 로드
        # Load the floor point cloud
        floor_pcds[floor_info["floor_id"]] = o3d.io.read_point_cloud(ply_path)

    # --------------------------------------------------------------------------------------------------------------------------------------
    
    # Load paths to room PLY files and corresponding JSON metadata
    rooms_ply_paths = sorted(glob.glob(os.path.join(params.graph_path, "rooms", "*.ply")))
    rooms_info_paths = sorted(glob.glob(os.path.join(params.graph_path, "rooms", "*.json")))

    # Initialize data structures for storing room point clouds and metadata
    room_pcds = {}
    room_infos = {}

    # Process each room
    for ply_path, info_path in zip(rooms_ply_paths, rooms_info_paths):
        with open(info_path, "r") as fp:
            room_info = json.load(fp)
        # Store relevant room metadata
        room_infos[room_info["room_id"]] = {
            k: v for k, v in room_info.items() if k in ["room_id", "name", "floor_id", "room_height", "room_zero_level", "vertices"]
        }
        for o_id in room_info["objects"]:
            hier_topo[room_info["floor_id"]][room_info["room_id"]].append(o_id)

        """
        예시
        room_infos = {
            "0_0": {
                "room_id": "0_1",
                "name": "bedroom",
                "floor_id": "0",
                "room_height": 3.2,
                "room_zero_level": 0.0,
                "vertices": [[...], ...],
            },
            "0_1": {
                "room_id": "0_2",
                "name": "kichen",
                "floor_id": "1",
                "room_height": 3.1,
                "room_zero_level": 3.3,
                "vertices": [[...], ...],
            },
            ...
        }
        """

        # 시각화 할 room point cloud 로드
        # Load the room point cloud and apply filtering
        origin_room_cloud = o3d.io.read_point_cloud(ply_path)
        origin_room_cloud_xyz = np.asarray(origin_room_cloud.points)
        # 전체를 다 잘 보이게 하기 위해서 천장 부분을 잘라내기
        below_ceiling_filter = (
            # y좌표만 뽑은 1차원 배열
            origin_room_cloud_xyz[:, 1]
            < room_infos[room_info["room_id"]]["room_zero_level"]
            + room_infos[room_info["room_id"]]["room_height"]
            - 0.4
        )
        room_pcds[room_info["room_id"]] = origin_room_cloud.select_by_index(np.where(below_ceiling_filter)[0])
        cloud_xyz = np.asarray(room_pcds[room_info["room_id"]].points)
        # floor에 저장해둔 시각화를 위한 offset 값을 더해주고 cloud_xyz 갱신
        cloud_xyz += floor_infos[room_info["floor_id"]]["viz_offset"]
        cloud = pv.PolyData(cloud_xyz)
        # room_pcds[room_info["room_id"]].colors = o3d.utility.Vector3dVector(
        #     np.clip(np.array(room_pcds[room_info["room_id"]].colors) * 1.2, 0.0, 1.0)
        # )
        # p.add_mesh(
        #     cloud,
        #     scalars=np.asarray(room_pcds[room_info["room_id"]].colors),
        #     rgb=True,
        #     point_size=5,
        #     opacity=0.8,
        #     show_vertices=True,
        # )

    # --------------------------------------------------------------------------------------------------------------------------------------

    # Load paths to object PLY files and corresponding JSON metadata
    objects_ply_paths = sorted(glob.glob(os.path.join(params.graph_path, "objects", "*.ply")))
    objects_info_paths = sorted(glob.glob(os.path.join(params.graph_path, "objects", "*.json")))

    # Initialize data structures for storing object point clouds, metadata, and features
    object_pcds = {}
    object_infos = {}
    object_feats = {}

    # Process each object
    for ply_path, info_path in zip(objects_ply_paths, objects_info_paths):
        with open(info_path, "r") as fp:
            object_info = json.load(fp)
        # Store relevant object metadata
        object_infos[object_info["object_id"]] = {
            k: v for k, v in object_info.items() if k in ["object_id", "name", "room_id", "object_height", "object_zero_level"]
        }
        object_feats[object_info["object_id"]] = np.asarray(object_info["embedding"])
        hier_topo[room_infos[object_info["room_id"]]["floor_id"]][room_infos[object_info["room_id"]]["room_id"]].append(
            object_info["object_id"]
        )

        """
        예시
        object_infos = {
            "0_0_0": {
                "object_id": "0_0_0",
                "name": "chair",
                "room_id": "0_0",
                "object_height": 1.2,
                "object_zero_level": 0.0,
            },
            "0_0_1": {
                "object_id": "0_0_1",
                "name": "table",
                "room_id": "0_0",
                "object_height": 0.8,
                "object_zero_level": 0.0,
            },
            ...
        """
        
        """
        계층 정보 관리 딕셔너리
        hier_topo = {
            "0": {
                # floor_0 안의 room_id 들
                "0_0": ["0_0_0", "0_0_1", ...],    # room_0_0 안의 object_id 들
                "0_1": ["0_1_0", ...],
            },
            "1": {
                "1_0": ["1_0_0", ...],
            },
        }
        """

        # Load the object point cloud and apply visualization offset
        object_pcds[object_info["object_id"]] = o3d.io.read_point_cloud(ply_path)
        cloud_xyz = np.asarray(object_pcds[object_info["object_id"]].points)
        # object_infosdptj room_id -> room_infos에서 floor_id -> floor_infos에서 viz_offset 값을 찾아 더해줌
        cloud_xyz += floor_infos[room_infos[object_info["room_id"]]["floor_id"]]["viz_offset"]

    # Floor centroid 시각화 위치 계산
    # Calculate centroids for floors
    max_floor_id = list(hier_topo.keys())[-1]
    max_floor_centroid = np.mean(np.asarray(floor_pcds[max_floor_id].points), axis=0)
    floor_centroids = {floor_id: np.mean(np.asarray(floor_pcds[floor_id].points), axis=0) for floor_id in hier_topo.keys()}
    floor_centroids_viz = {floor_id: floor_centroids[floor_id] + floor_infos[floor_id]["viz_offset"] + [0.0, 4.0, 0.0]
                           for floor_id in hier_topo.keys()}

    # Root 노드 시각화 -> 사용X
    # Calculate the root node centroid for visualization
    root_offset = [
        np.mean(np.stack(list(floor_centroids_viz.values())).T, axis=1)[0],
        6.0,
        np.mean(np.stack(list(floor_centroids_viz.values())).T, axis=1)[2],
    ]
    root_node_centroid_viz = max_floor_centroid + floor_infos[max_floor_id]["viz_offset"] + root_offset

    # Floor 해당 노드 시각화
    # Visualize the centroids of floors
    for floor_id, floor_centroid_viz in floor_centroids_viz.items():
        p.add_mesh(pv.Sphere(center=tuple(floor_centroid_viz), radius=0.5), color="orange")

    # Room centroid 시각화 위치 계산
    # Calculate and visualize the centroids of rooms
    room_centroids = {room_id: np.mean(np.asarray(room_pcds[room_id].points), axis=0) for room_id in room_infos.keys()}
    room_centroids_viz = {room_id: room_centroids[room_id] + [0.0, 3.5, 0] for room_id in room_infos.keys()}
    for room_id, room_centroid_viz in room_centroids_viz.items():
        p.add_mesh(pv.Sphere(center=tuple(room_centroid_viz), radius=0.25), color="blue")
        p.add_mesh(
            pv.Line(tuple(floor_centroids_viz[room_infos[room_id]["floor_id"]]), tuple(room_centroid_viz)),
            line_width=4,
        )

    # Object 해당 노드 시각화
    # Calculate and visualize the centroids of objects
    obj_centroids = {obj_id: np.mean(np.asarray(object_pcds[obj_id].points), axis=0) for obj_id in object_infos.keys()}
    obj_centroids_viz = {obj_id: obj_centroids[obj_id] for obj_id in object_infos.keys()}
    for obj_id, obj_info in object_infos.items():
        if (
            not any(
                substring in obj_info["name"].lower()
                for substring in ["wall", "floor", "ceiling", "paneling", "banner", "overhang"]
            )
            and len(object_pcds[obj_id].points) > 100
        ):
            p.add_mesh(
                pv.Line(tuple(room_centroids_viz[obj_info["room_id"]]), tuple(obj_centroids_viz[obj_id])),
                line_width=1.5,
                opacity=0.5,
            )
            print("included object of category:", obj_info["name"])
            # add object point cloud
            # object_pcds[obj_id].paint_uniform_color(np.random.rand(3))
            cloud_xyz = np.asarray(object_pcds[obj_id].points)
            cloud = pv.PolyData(cloud_xyz)
            p.add_mesh(
                cloud,
                scalars=np.asarray(object_pcds[obj_id].colors),
                rgb=True,
                point_size=5,
                show_vertices=True,
                show_scalar_bar=False,
            )

    # Show the visualization
    try:
        p.remove_scalar_bar()
    except Exception:
        pass
    p.show()

if __name__ == "__main__":
    main()
