from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple, List

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from hovsg.graph.graph import Graph
from open3dsg.blip_relation_extractor import BlipRelationExtractor, load_relation_embeddings


class RelationalGraph(Graph):
    """
    Graph를 상속해서, object-object 관계(edge) 관련 유틸을 추가한 확장 클래스.
    - 기존 Graph.__init__ / create_feature_map / segment_* / create_graph / create_nav_graph 등은 그대로 상속 사용
    - 여기서는 object-frame / object-object / relation-frame / BLIP relation embedding 추가
    """

    def compute_object_frame_bboxes(self, save_path: str):
        """
        각 object 인스턴스가 잘 보이는 frame과 그 때의 2D bbox를 계산.
        결과를 save_path/edges/object2frames.pkl 에 저장.
        """
        num_objs = len(self.mask_pcds)
        object2frames: List[List[dict]] = [[] for _ in range(num_objs)]

        # 전체 object point 배열
        obj_points = [np.asarray(pcd.points) for pcd in self.mask_pcds]

        for frame_id in tqdm(
            range(0, len(self.dataset), self.cfg.pipeline.skip_frames),
            desc="Computing object-frame bboxes",
        ):
            rgb, depth, pose, _, K = self.dataset[frame_id]

            # Intrinsic matrix
            K = np.array(K).reshape(3, 3)

            # pose: Twc (camera-to-world) → world->camera
            world_to_cam = np.linalg.inv(pose)

            # 이미지 크기
            if hasattr(depth, "size"):
                H, W = depth.size[1], depth.size[0]
            else:
                H, W = np.array(depth).shape

            # depth array
            depth_np = np.array(depth) / self.cfg.main.depth_scale

            for obj_id, pts_world in enumerate(obj_points):
                if len(pts_world) == 0:
                    continue

                # 1) world -> camera 좌표
                pts_h = np.concatenate(
                    [pts_world, np.ones((pts_world.shape[0], 1))], axis=1
                ).T  # (4, N)
                pts_cam = (world_to_cam @ pts_h)[:3].T  # (N, 3)

                # 2) 카메라 앞쪽(z>0)만 사용
                front_mask = pts_cam[:, 2] > 0.05
                if front_mask.sum() < 20:
                    continue
                pts_cam = pts_cam[front_mask]

                # 3) K로 projection
                uv = (K @ pts_cam.T).T  # (N, 3)
                uv[:, 0] /= uv[:, 2]
                uv[:, 1] /= uv[:, 2]
                u = uv[:, 0]
                v = uv[:, 1]

                # 4) 이미지 안에 있는 점만
                in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if in_img.sum() < 10:
                    continue
                u = u[in_img]
                v = v[in_img]
                z = pts_cam[in_img, 2]

                # 5) depth consistency check
                v_int = np.clip(v.astype(int), 0, H - 1)
                u_int = np.clip(u.astype(int), 0, W - 1)
                depth_sample = depth_np[v_int, u_int]
                valid_depth = depth_sample > 0

                if valid_depth.sum() < 10:
                    continue

                depth_err = np.abs(depth_sample[valid_depth] - z[valid_depth])
                vis_ratio = (
                    depth_err < self.cfg.pipeline.depth_consistency_thresh
                ).mean()

                if vis_ratio < self.cfg.pipeline.vis_ratio_thresh:
                    continue

                # 6) bbox 계산
                xmin, xmax = u.min(), u.max()
                ymin, ymax = v.min(), v.max()
                bbox_w, bbox_h = xmax - xmin, ymax - ymin

                num_pixels = bbox_w * bbox_h
                if num_pixels < self.cfg.pipeline.min_bbox_pixels:
                    continue

                bbox = [float(xmin), float(ymin), float(xmax), float(ymax)]
                object2frames[obj_id].append(
                    dict(
                        frame_id=frame_id,
                        bbox=bbox,
                        vis_ratio=float(vis_ratio),
                        num_pixels=float(num_pixels),
                    )
                )

        # 각 object 마다 상위 top_k 프레임만 남기기
        top_k = self.cfg.pipeline.top_k_frames
        for obj_id in range(num_objs):
            frames = object2frames[obj_id]
            # score = num_pixels * vis_ratio 기준으로 정렬
            frames.sort(
                key=lambda x: x["num_pixels"] * x["vis_ratio"],
                reverse=True,
            )
            object2frames[obj_id] = frames[:top_k]

        # 저장
        os.makedirs(os.path.join(save_path, "edges"), exist_ok=True)
        with open(
            os.path.join(save_path, "edges", "object2frames.pkl"), "wb"
        ) as f:
            pickle.dump(object2frames, f)

        print(
            f"Object-frame bboxes saved: {sum(len(x) for x in object2frames)} mappings"
        )

    def compute_object_pairs(self, save_path: str) -> Dict[Tuple[int, int], float]:
        """
        객체 쌍 후보를 생성. 3D 거리 기반으로 필터링.
        결과를 save_path/edges/object_pairs.pkl 에 저장.
        """
        num_objs = len(self.mask_pcds)
        obj_centers = []

        # 각 객체의 중심 좌표 계산
        for pcd in self.mask_pcds:
            if len(pcd.points) > 0:
                center = np.mean(np.asarray(pcd.points), axis=0)
            else:
                center = np.array([0.0, 0.0, 0.0])
            obj_centers.append(center)

        obj_centers = np.array(obj_centers)  # (N, 3)

        # 모든 쌍에 대한 거리 계산
        dist_matrix = cdist(obj_centers, obj_centers)  # (N, N)

        # 대각선(자기 자신)은 무한대로
        np.fill_diagonal(dist_matrix, np.inf)

        # relation_max_dist 이하인 쌍만 선택
        max_dist = self.cfg.pipeline.relation_max_dist
        valid_pairs = np.argwhere(dist_matrix < max_dist)

        # (i, j)와 (j, i) 중복 제거 (i < j만 유지)
        valid_pairs = valid_pairs[valid_pairs[:, 0] < valid_pairs[:, 1]]

        # dict로 저장: {(obj_i, obj_j): distance}
        object_pairs: Dict[Tuple[int, int], float] = {}
        for i, j in valid_pairs:
            object_pairs[(int(i), int(j))] = float(dist_matrix[i, j])

        # 저장
        os.makedirs(os.path.join(save_path, "edges"), exist_ok=True)
        with open(
            os.path.join(save_path, "edges", "object_pairs.pkl"), "wb"
        ) as f:
            pickle.dump(object_pairs, f)

        print(f"Object pairs saved: {len(object_pairs)} pairs")
        return object_pairs

    def compute_relation_frames(self, save_path: str):
        """
        각 객체 쌍에 대해 두 객체가 동시에 잘 보이는 프레임 찾기.
        object2frames.pkl과 object_pairs.pkl 로드 필요.
        결과를 save_path/edges/relation_frames.pkl 에 저장.
        """
        edges_dir = os.path.join(save_path, "edges")

        with open(os.path.join(edges_dir, "object2frames.pkl"), "rb") as f:
            object2frames = pickle.load(f)

        with open(os.path.join(edges_dir, "object_pairs.pkl"), "rb") as f:
            object_pairs = pickle.load(f)

        relation_frames = {}
        top_k = self.cfg.pipeline.top_k_rel_frames

        for (obj_i, obj_j), dist in tqdm(
            object_pairs.items(), desc="Computing relation frames"
        ):
            frames_i = object2frames[obj_i]
            frames_j = object2frames[obj_j]

            if len(frames_i) == 0 or len(frames_j) == 0:
                continue

            # 두 객체의 공통 프레임 찾기
            frame_ids_i = {f["frame_id"] for f in frames_i}
            frame_ids_j = {f["frame_id"] for f in frames_j}
            common_frames = frame_ids_i & frame_ids_j

            if len(common_frames) == 0:
                continue

            # 공통 프레임에서 score 계산
            candidates = []
            for fid in common_frames:
                info_i = next(f for f in frames_i if f["frame_id"] == fid)
                info_j = next(f for f in frames_j if f["frame_id"] == fid)

                score_i = info_i["num_pixels"] * info_i["vis_ratio"]
                score_j = info_j["num_pixels"] * info_j["vis_ratio"]
                score = min(score_i, score_j)

                candidates.append(
                    dict(
                        frame_id=fid,
                        bbox_i=info_i["bbox"],
                        bbox_j=info_j["bbox"],
                        score=score,
                    )
                )

            candidates.sort(key=lambda x: x["score"], reverse=True)
            relation_frames[(obj_i, obj_j)] = candidates[:top_k]

        with open(os.path.join(edges_dir, "relation_frames.pkl"), "wb") as f:
            pickle.dump(relation_frames, f)

        print(f"Relation frames saved: {len(relation_frames)} pairs")
        return relation_frames

    def compute_blip_relation_embeddings(self, save_path: str):
        """
        BLIP으로 객체 쌍의 관계 임베딩 계산.
        relation_frames.pkl 필요.
        """
        edges_dir = os.path.join(save_path, "edges")
        with open(os.path.join(edges_dir, "relation_frames.pkl"), "rb") as f:
            relation_frames = pickle.load(f)

        blip_extractor = BlipRelationExtractor(
            model_name="Salesforce/blip-image-captioning-large",
            device=self.device,
        )

        relation_embeddings = blip_extractor.compute_relation_embeddings(
            dataset=self.dataset,
            relation_frames=relation_frames,
            save_path=save_path,
            bbox_margin=self.cfg.pipeline.get("bbox_margin", 10),
        )

        print("BLIP relation embeddings computed successfully")
        return relation_embeddings
    
    def attach_relation_edges_to_nx(self, save_path: str):
        """
        pkl / npz 에 저장된 관계 정보들을 실제 self.graph (networkx)에 edge로 추가.
        - object2frames / object_pairs / relation_frames / relation_embeddings 사용
        """
        edges_dir = os.path.join(save_path, "edges")

        # 1) relation_frames: (obj_i, obj_j) -> [{frame_id, bbox_i, bbox_j, score}, ...]
        with open(os.path.join(edges_dir, "relation_frames.pkl"), "rb") as f:
            relation_frames = pickle.load(f)

        # 2) object_pairs: (obj_i, obj_j) -> distance_3d
        with open(os.path.join(edges_dir, "object_pairs.pkl"), "rb") as f:
            object_pairs = pickle.load(f)

        # 3) relation_embeddings: (obj_i, obj_j) -> np.ndarray(emb_dim,)
        emb_path = os.path.join(edges_dir, "relation_embeddings.npz")
        relation_embeddings = load_relation_embeddings(emb_path)

        num_added = 0

        for (obj_i, obj_j), frames in relation_frames.items():
            # safety: 인덱스 범위 체크
            if obj_i >= len(self.objects) or obj_j >= len(self.objects):
                continue

            obj_node_i = self.objects[obj_i]
            obj_node_j = self.objects[obj_j]

            # 가장 score가 높은 frame 하나 선택
            best_frame = frames[0] if len(frames) > 0 else None
            best_frame_id = None if best_frame is None else best_frame["frame_id"]

            # 3D 거리
            dist_ij = object_pairs.get((obj_i, obj_j), None)

            # 관계 임베딩
            rel_emb = relation_embeddings.get((obj_i, obj_j), None)
            if rel_emb is not None:
                rel_emb = np.asarray(rel_emb).astype(float)

            # networkx edge 추가
            self.graph.add_edge(
                obj_node_i,
                obj_node_j,
                type="relation",
                obj_i=int(obj_i),
                obj_j=int(obj_j),
                best_frame_id=best_frame_id,
                distance_3d=dist_ij,
                relation_emb=rel_emb,
            )
            num_added += 1

        print(f"[attach_relation_edges_to_nx] Added {num_added} relation edges to self.graph")

    def build_relational_graph(self, save_path: str | None = None):
        
        # 기본 그래프 생성
        super().build_graph(save_path)

        # 관계 그래프 생성
        if not self.cfg.pipeline.get("compute_relations", False):
            return

        print("\n=== Computing object relations ===")
        print("Step 1: Object-frame bbox mapping...")
        self.compute_object_frame_bboxes(save_path)

        print("Step 2: Object pair generation...")
        self.compute_object_pairs(save_path)

        print("Step 3: Relation frame selection...")
        self.compute_relation_frames(save_path)

        print("Step 4: BLIP relation embeddings...")
        self.compute_blip_relation_embeddings(save_path)
        
        print("Step 5: Attach relation edges to networkx graph...")
        self.attach_relation_edges_to_nx(save_path)
        
        print("=== Relation computation complete ===\n")