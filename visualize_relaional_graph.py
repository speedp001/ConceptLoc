
import os
import json
import glob
import numpy as np
from collections import defaultdict

import hydra
import open3d as o3d
import matplotlib.pyplot as plt
import pyvista as pv
from tqdm import tqdm
from omegaconf import DictConfig


def _find_relation_names_json(graph_path: str) -> str:
    """Try common layouts and return the first existing relation_names.json path."""
    candidates = [
        os.path.join(graph_path, "edges", "relation_names.json"),
        os.path.join(graph_path, "graph", "edges", "relation_names.json"),
        os.path.join(graph_path, "edges", "relation_names.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""


def _load_relation_edges(relation_json_path: str):
    """Load relation edges from relation_names.json.

    Supports keys formatted like "i_j" (mask_idx pair). Each value may contain:
      - name (str)
      - score (float)
      - frame_ids (list)
      - object_id ([object_id_i, object_id_j])

    Returns a list of dict edges:
      {"obj_i": int, "obj_j": int, "object_id_i": str|None, "object_id_j": str|None, "name": str|None, "score": float|None}
    """
    if not relation_json_path or not os.path.exists(relation_json_path):
        return []

    with open(relation_json_path, "r") as f:
        data = json.load(f)

    edges = []
    for k, v in data.items():
        try:
            a, b = k.split("_")
            obj_i, obj_j = int(a), int(b)
        except Exception:
            # Skip malformed keys
            continue

        obj_ids = v.get("object_id", None)
        object_id_i, object_id_j = None, None
        if isinstance(obj_ids, (list, tuple)) and len(obj_ids) >= 2:
            object_id_i, object_id_j = obj_ids[0], obj_ids[1]

        edges.append(
            dict(
                obj_i=obj_i,
                obj_j=obj_j,
                object_id_i=object_id_i,
                object_id_j=object_id_j,
                name=v.get("name", None),
                score=v.get("score", None),
                frame_ids=v.get("frame_ids", []),
            )
        )

    return edges


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

        floor_infos[floor_info["floor_id"]] = {
            k: v
            for k, v in floor_info.items()
            if k in ["floor_id", "name", "rooms", "floor_height", "floor_zero_level", "vertices"]
        }
        # Apply visualization offset to each floor
        floor_infos[floor_info["floor_id"]]["viz_offset"] = init_offset * counter
        for r_id in floor_info["rooms"]:
            hier_topo[floor_info["floor_id"]][r_id] = []

        # floor point cloud 로드
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

        room_infos[room_info["room_id"]] = {
            k: v
            for k, v in room_info.items()
            if k in ["room_id", "name", "floor_id", "room_height", "room_zero_level", "vertices"]
        }
        for o_id in room_info["objects"]:
            hier_topo[room_info["floor_id"]][room_info["room_id"]].append(o_id)

        # 시각화 할 room point cloud 로드
        origin_room_cloud = o3d.io.read_point_cloud(ply_path)
        origin_room_cloud_xyz = np.asarray(origin_room_cloud.points)

        # 전체를 다 잘 보이게 하기 위해서 천장 부분을 잘라내기
        below_ceiling_filter = (
            origin_room_cloud_xyz[:, 1]
            < room_infos[room_info["room_id"]]["room_zero_level"]
            + room_infos[room_info["room_id"]]["room_height"]
            - 0.4
        )
        room_pcds[room_info["room_id"]] = origin_room_cloud.select_by_index(np.where(below_ceiling_filter)[0])

        cloud_xyz = np.asarray(room_pcds[room_info["room_id"]].points)
        cloud_xyz += floor_infos[room_info["floor_id"]]["viz_offset"]
        _ = pv.PolyData(cloud_xyz)

        # NOTE:
        # 아래 add_mesh 블록은 room point cloud까지 같이 그리면 화면이 너무 복잡해지고,
        # object point cloud와 hierarchy 라인이 잘 안 보이기 때문에 주석 처리해둔 케이스가 많음.
        # 필요하면 주석을 풀어서 room pcd도 같이 렌더링하면 됨.
        # room_pcds[room_info["room_id"]].colors = o3d.utility.Vector3dVector(
        #     np.clip(np.array(room_pcds[room_info["room_id"]].colors) * 1.2, 0.0, 1.0)
        # )
        # p.add_mesh(
        #     _,
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

        object_infos[object_info["object_id"]] = {
            k: v
            for k, v in object_info.items()
            if k in ["object_id", "name", "room_id", "object_height", "object_zero_level"]
        }
        object_feats[object_info["object_id"]] = np.asarray(object_info["embedding"]) if object_info.get("embedding", "") != "" else None

        hier_topo[room_infos[object_info["room_id"]]["floor_id"]][room_infos[object_info["room_id"]]["room_id"]].append(
            object_info["object_id"]
        )

        # Load the object point cloud and apply visualization offset
        object_pcds[object_info["object_id"]] = o3d.io.read_point_cloud(ply_path)
        cloud_xyz = np.asarray(object_pcds[object_info["object_id"]].points)
        cloud_xyz += floor_infos[room_infos[object_info["room_id"]]["floor_id"]]["viz_offset"]

    # Floor centroid 시각화 위치 계산
    max_floor_id = list(hier_topo.keys())[-1]
    max_floor_centroid = np.mean(np.asarray(floor_pcds[max_floor_id].points), axis=0)
    floor_centroids = {floor_id: np.mean(np.asarray(floor_pcds[floor_id].points), axis=0) for floor_id in hier_topo.keys()}
    floor_centroids_viz = {
        floor_id: floor_centroids[floor_id] + floor_infos[floor_id]["viz_offset"] + [0.0, 4.0, 0.0]
        for floor_id in hier_topo.keys()
    }

    # Root 노드 시각화 -> 사용X
    root_offset = [
        np.mean(np.stack(list(floor_centroids_viz.values())).T, axis=1)[0],
        6.0,
        np.mean(np.stack(list(floor_centroids_viz.values())).T, axis=1)[2],
    ]
    _ = max_floor_centroid + floor_infos[max_floor_id]["viz_offset"] + root_offset

    # Floor 해당 노드 시각화
    for floor_id, floor_centroid_viz in floor_centroids_viz.items():
        p.add_mesh(pv.Sphere(center=tuple(floor_centroid_viz), radius=0.5), color="orange")

    # Room centroid 시각화 위치 계산
    room_centroids = {room_id: np.mean(np.asarray(room_pcds[room_id].points), axis=0) for room_id in room_infos.keys()}
    room_centroids_viz = {room_id: room_centroids[room_id] + [0.0, 3.5, 0] for room_id in room_infos.keys()}
    for room_id, room_centroid_viz in room_centroids_viz.items():
        p.add_mesh(pv.Sphere(center=tuple(room_centroid_viz), radius=0.25), color="blue")
        p.add_mesh(
            pv.Line(tuple(floor_centroids_viz[room_infos[room_id]["floor_id"]]), tuple(room_centroid_viz)),
            line_width=4,
        )

    # Object 해당 노드 시각화
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
        
    # Relation edges 시각화
    relation_json_path = _find_relation_names_json(params.graph_path)
    relation_edges = _load_relation_edges(relation_json_path)
    print(f"[visualize_relational_graph] relation_names.json: {relation_json_path if relation_json_path else 'NOT FOUND'}")
    print(f"[visualize_relational_graph] #relation_edges: {len(relation_edges)}")

    # 유효 edge만 모으기
    pts = []
    lines = []
    seen = set()
    drawn = 0
    skipped = 0
    pid = 0

    for e in relation_edges:
        oi = e.get("object_id_i", None)
        oj = e.get("object_id_j", None)
        if oi is None or oj is None:
            skipped += 1
            continue
        if oi not in obj_centroids_viz or oj not in obj_centroids_viz:
            skipped += 1
            continue

        k = tuple(sorted([oi, oj]))
        if k in seen:
            continue
        seen.add(k)

        # 두 endpoint를 points 배열에 추가
        p0 = obj_centroids_viz[oi]
        p1 = obj_centroids_viz[oj]
        pts.append(p0)
        pts.append(p1)

        # VTK line cell 포맷: [2, idx0, idx1]
        lines.append([2, pid, pid + 1])
        pid += 2
        drawn += 1

    print(f"[visualize_relational_graph] relation edges prepared={drawn}, skipped={skipped}")

    # 2) 한 번에 add_mesh
    if drawn > 0:
        pts = np.asarray(pts)
        lines = np.asarray(lines).ravel()  # (N*3,)
        rel_poly = pv.PolyData(pts)
        rel_poly.lines = lines

        p.add_mesh(
            rel_poly,
            color="red",
            line_width=2,
            opacity=0.6,
        )

    # Show the visualization
    try:
        p.remove_scalar_bar()
    except Exception:
        pass
    p.show()

if __name__ == "__main__":
    main()