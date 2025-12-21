from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple, List

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist

from hovsg.graph.graph import Graph
from open3dsg.blip_relation_extractor import (
    BlipRelationExtractor, 
    load_relation_embeddings,
    load_relation_names,
)

# 기존 Graph 클래스를 상속받아 관계 그래프로 확장
class RelationalGraph(Graph):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 관계 그래프 중간 결과를 위한 전역 변수 선언
        self.object2frames = None
        self.object_pairs = None
        self.relation_embeddings = None
        self.save_dir = None
        self.mask_idx_to_object = {}

    # 객체 인스턴스가 잘 보이는 프레임 추출과 공통영역 바운딩 박스 계산
    def compute_object_frame_bboxes(self):

        num_objs = len(self.mask_pcds)
        object2frames: List[List[dict]] = [[] for _ in range(num_objs)]

        # 전체 object point 배열
        # open3d point cloud 객체를 연산하기 위해 numpy 배열로 변환
        obj_points = [np.asarray(pcd.points) for pcd in self.mask_pcds]

        # 프레임 순회
        for frame_id in tqdm(
            range(0, len(self.dataset), self.cfg.pipeline.relation_skip_frames),
            desc="Computing object-frame bboxes",
        ):
            # 데이터셋 경로는 hm3dsem_walks 내부
            rgb, depth, pose, _, K = self.dataset[frame_id]

            # Intrinsic matrix
            K = np.array(K).reshape(3, 3)

            # pose: Twc (camera-to-world) → world->camera
            world_to_cam = np.linalg.inv(pose)

            # 이미지 크기
            # Hm3dsem과 ScanNet, Replica의 데이터 형식이 다르므로 분기 처리
            if hasattr(depth, "size"):
                H, W = depth.size[1], depth.size[0]
            else:
                H, W = np.array(depth).shape

            # depth array
            depth_np = np.array(depth) / 1000.0

            # 프레임에서 각 객체를 투영 
            for obj_id, pts_world in enumerate(obj_points):
                # 포인트가 없는 객체는 생략
                if len(pts_world) == 0:
                    continue

                # 1단계 world -> camera 좌표
                """
                월드 좌표 (의자 위치)          카메라 좌표 (카메라 기준)
                    y↑                              y↑
                    │    ◆ 의자                      │
                    │   /                           │    ◆ 의자
                    │  /                            │   /
                    └────► x                        └────► z (카메라가 바라보는 방향)
                """
                pts_h = np.concatenate(
                    [pts_world, np.ones((pts_world.shape[0], 1))], axis=1
                ).T  # (4, N)
                pts_cam = (world_to_cam @ pts_h)[:3].T  # (N, 3)

                # 2단계 카메라 앞쪽(z>0)만 사용
                front_mask = pts_cam[:, 2] > 0.05
                if front_mask.sum() < 30:
                    continue
                pts_cam = pts_cam[front_mask]

                # 3단계 K로 projection -> 각 카메라 좌표계에서 3차원 포인트 클라우드를 이미지 프레임으로 투영
                """
                3D 카메라 좌표 (x, y, z)
                        │
                        ▼ K (intrinsic matrix)
                        │
                2D 이미지 좌표 (u, v)

                        ┌─────────────────┐
                        │    (u, v)       │
                        │       ●         │  ← 이미지 평면
                        │                 │
                        └─────────────────┘
                """
                # K: 내부 인트린직 파라미터
                uv = (K @ pts_cam.T).T  # (N, 3)
                uv[:, 0] /= uv[:, 2]
                uv[:, 1] /= uv[:, 2]
                u = uv[:, 0]
                v = uv[:, 1]

                # 4단계 이미지 안에 있는 점만 고려
                """
                이미지 경계 (640 x 480)
                ┌─────────────────────┐
                │  ●  ●               │  ← 이미지 안 (유효)
                │      ●   ●          │
                │              ●      │
                └─────────────────────┘
                        ●  ← 이미지 밖 (제거)
                    ●        ← 이미지 밖 (제거)
                """
                in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                if in_img.sum() < 30:
                    continue
                u = u[in_img]
                v = v[in_img]
                z = pts_cam[in_img, 2]

                # 5단계 depth consistency check -> 가려짐 여부 판단 (depth 픽셀의 깊이 값이 실제 투영된 점과 일치하는지 확인)
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

                # 오차가 threshold 이하인 점의 비율
                if vis_ratio < self.cfg.pipeline.vis_ratio_thresh:
                    continue

                # 6단계 bbox 계산
                """
                이미지 (640 x 480)
                ┌───────────────────────────┐
                │                           │
                │   ┌─────────┐             │
                │   │ ● ● ●   │ bbox        │
                │   │  ● ● ●  │             │
                │   │   ● ●   │             │
                │   └─────────┘             │
                │ (xmin,ymin)  (xmax,ymax)  │
                └───────────────────────────┘

                num_pixels = bbox_w × bbox_h = 얼마나 크게 보이는지
                """
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

        # 인스턴스 변수로 저장
        self.object2frames = object2frames
        print(f"Object-frame bboxes computed: {sum(len(x) for x in object2frames)} mappings")

    # 객체 쌍 후보를 생성 (3D 거리 기반 필터링)
    # 최종으로 매칭된 쌍을 딕셔너리 형태로 저장
    def compute_object_pairs(self) -> Dict[Tuple[int, int], float]:
        
        # 각 객체의 중심 좌표 계산
        obj_centers = []
        for pcd in self.mask_pcds:
            if len(pcd.points) > 0:
                center = np.mean(np.asarray(pcd.points), axis=0)
            else:
                center = np.array([0.0, 0.0, 0.0])
            obj_centers.append(center)
        """
        N x 3
        obj_centers = [
            [ 1.2, 0.8, -3.1],   # obj 0 중심
            [ 1.8, 0.8, -2.7],   # obj 1 중심
            [ 7.5, 0.9, -1.2],   # obj 2 중심
            [ 1.3, 0.7, -3.0],   # obj 3 중심
            [ 0.2, 0.8, -3.3],   # obj 4 중심
        ]
        """

        obj_centers = np.array(obj_centers)

        # 모든 쌍에 대한 거리 계산
        dist_matrix = cdist(obj_centers, obj_centers)
        """
        N x N
        dist_matrix =
        [
            [0.00, 0.68, 6.50, 0.14, 1.10],
            [0.68, 0.00, 5.90, 0.70, 1.80],
            [6.50, 5.90, 0.00, 6.40, 7.20],
            [0.14, 0.70, 6.40, 0.00, 1.20],
            [1.10, 1.80, 7.20, 1.20, 0.00],
        ]
        """

        # 대각선(자기 자신)은 쌍을 만들지 않기 위해 무한대로 처리 -> 거리 기반이므로 자연스럽게 제외
        np.fill_diagonal(dist_matrix, np.inf)
        """
        [
            [inf, 0.68, 6.50, 0.14, 1.10],
            [0.68, inf, 5.90, 0.70, 1.80],
            [6.50, 5.90, inf, 6.40, 7.20],
            [0.14, 0.70, 6.40, inf, 1.20],
            [1.10, 1.80, 7.20, 1.20, inf],
        ]
        """

        # relation_max_dist 이하인 쌍만 선택
        max_dist = self.cfg.pipeline.relation_max_dist
        valid_pairs = np.argwhere(dist_matrix < max_dist)
        """
        valid_pairs =
        [
            [0, 1],
            [0, 3],
            [0, 4],
            [1, 0],
            [1, 3],
            [1, 4],
            [3, 0],
            [3, 1],
            [3, 4],
            [4, 0],
            [4, 1],
            [4, 3],
        ]
        """

        # (i, j)와 (j, i) 중복 제거 (i < j만 유지)
        valid_pairs = valid_pairs[valid_pairs[:, 0] < valid_pairs[:, 1]]
        """
        valid_pairs =
        [
            [0, 1],
            [0, 3],
            [0, 4],
            [1, 3],
            [1, 4],
            [3, 4],
        ]
        """

        # dict로 저장: {(obj_i, obj_j): distance}
        object_pairs: Dict[Tuple[int, int], float] = {}
        for i, j in valid_pairs:
            object_pairs[(int(i), int(j))] = float(dist_matrix[i, j])
            
        """
        object_pairs =
        {
        (0, 1): 0.68,
        (0, 3): 0.14,
        (0, 4): 1.10,
        (1, 3): 0.70,
        (1, 4): 1.80,
        (3, 4): 1.20,
        }
        """

        # 인스턴스 변수로 저장
        self.object_pairs = object_pairs
        print(f"Object pairs computed: {len(object_pairs)} pairs")
        return object_pairs

    # 각 객체 쌍에 대해 두 객체가 동시에 잘 보이는 프레임 찾기
    # 앞에서 구한 object2frames, object_pairs을 사용
    # object2frames: 각 객체가 잘 보이는 프레임 리스트
    # object_pairs: 3D 거리 기반으로 필터링된 객체 쌍
    def compute_relation_frames(self):

        object2frames = self.object2frames
        object_pairs = self.object_pairs
        top_k = self.cfg.pipeline.top_k_frames
        # 두 객체 쌍이 담긴 프레임 리스트
        relation_frames = {}
        
        for (obj_i, obj_j), dist in tqdm(object_pairs.items(), desc="Computing relation frames"):
            frames_i = object2frames[obj_i]
            frames_j = object2frames[obj_j]

            # 한 쪽이라도 프레임이 없으면 건너뜀
            if len(frames_i) == 0 or len(frames_j) == 0:
                continue

            # 두 객체의 공통 프레임 찾기
            frame_ids_i = {f["frame_id"] for f in frames_i}
            frame_ids_j = {f["frame_id"] for f in frames_j}
            common_frames = frame_ids_i & frame_ids_j

            # 공통 프레임이 없으면 건너뜀
            if len(common_frames) == 0:
                continue

            # 공통 프레임에서 score 계산
            # 공통 프레임에서 두 객체의 bbox가 얼마나 크게 잘 보이는지 평가
            candidates = []
            for fid in common_frames:
                info_i = next(f for f in frames_i if f["frame_id"] == fid)
                info_j = next(f for f in frames_j if f["frame_id"] == fid)

                score_i = info_i["num_pixels"] * info_i["vis_ratio"]
                score_j = info_j["num_pixels"] * info_j["vis_ratio"]
                # 둘 중 가장 낮은 score를 사용 -> 하나의 객체가 잘 보이더라도 다른 객체가 잘 안보이면 의미가 없음
                score = min(score_i, score_j)

                candidates.append(
                    dict(
                        frame_id=fid,
                        bbox_i=info_i["bbox"],
                        bbox_j=info_j["bbox"],
                        score=score,
                    )
                )

            # 후보 프레임 중 가장 score 좋은 프레임만 남기기
            candidates.sort(key=lambda x: x["score"], reverse=True)
            relation_frames[(obj_i, obj_j)] = candidates[:top_k]
            """
            relation_frames =
            {
                (0, 1): [
                    {
                        frame_id: 120,
                        bbox_i: [xmin, ymin, xmax, ymax],
                        bbox_j: [xmin, ymin, xmax, ymax],
                        score: 3456.7,
                    },
                    {
                        frame_id: 45,
                        bbox_i: [xmin, ymin, xmax, ymax],
                        bbox_j: [xmin, ymin, xmax, ymax],
                        score: 2987.3,
                    }
                ],
                (0, 2): [
                    {
                        frame_id: 85,
                        bbox_i: [xmin, ymin, xmax, ymax],
                        bbox_j: [xmin, ymin, xmax, ymax],
                        score: 2987.3,
                    },
                ],
                ... 
            }
            """
        self.relation_frames = relation_frames
        print(f"Relation frames computed: {len(relation_frames)} pairs")
        return relation_frames

    # BLIP 모델을 사용해 관계 임베딩 계산
    def compute_blip_relation_embeddings(self):

        relation_frames = self.relation_frames
        blip_extractor = BlipRelationExtractor(
            self.device,
            model_name="Salesforce/blip-itm-base-coco",
        )
        relation_embeddings = blip_extractor.compute_relation_embeddings(
            dataset=self.dataset,
            relation_frames=relation_frames,
            save_path=self.save_dir,
            bbox_margin=self.cfg.pipeline.get("bbox_margin", 10),
            infer_relation_names=self.cfg.pipeline.get("infer_relation_names", True)
        )
        
        self.relation_embeddings = relation_embeddings
        print("BLIP relation embeddings computed successfully")
        
        return relation_embeddings
    
    def attach_relation_edges(self):
        """
        graph/edges/relation_names.json + relation_embeddings.json을 읽어서
        self.graph(networkx)에 object-object relation edge를 추가
        key 포맷 -> "obj_i_obj_j" (obj_i, obj_j는 mask_idx)
        """
        save_path = self.save_dir

        # 관계 정의 Json 파일 로드
        relation_names = load_relation_names(save_path)
        relation_embeddings = load_relation_embeddings(save_path)

        # mask_idx -> Object 매핑 정보를 pkl 파일에서 로드 (Dict[int, Object 형태로 저장되어 있다는 전제)
        if not self.mask_idx_to_object:
            pkl_path = os.path.join(save_path, "graph", "objects", "mask_idx_to_object_id.pkl")
            if not os.path.exists(pkl_path):
                raise FileNotFoundError(f"mask_idx_to_object_id.pkl not found: {pkl_path}")

            with open(pkl_path, "rb") as f:
                self.mask_idx_to_object = pickle.load(f)  # Dict[int, Object]

        # 실제 그래프에 추가된 관계 edge 개수
        added = 0
        # 중복 edge 방지
        seen = set()

        for (obj_i, obj_j), info in relation_names.items():
            # 무방향 그래프 중복 방지
            key = (min(obj_i, obj_j), max(obj_i, obj_j))
            if key in seen:
                continue
            seen.add(key)

            # Object 객체를 그대로 가져옴 (node_i/node_j는 Object 인스턴스)
            node_i = self.mask_idx_to_object.get(int(obj_i))
            node_j = self.mask_idx_to_object.get(int(obj_j))
            if node_i is None or node_j is None:
                continue

            # 관계 임베딩
            rel_emb = relation_embeddings.get((obj_i, obj_j), None)
            if rel_emb is None:
                # 반대 방향도 확인
                rel_emb = relation_embeddings.get((obj_j, obj_i), None)

            # 임베딩 추출한 frame ids
            frame_ids = info.get("frame_ids", [])

            # edge 추가 + edge attribute 저장
            self.graph.add_edge(
                node_i, node_j,
                type="relation",
                mask_idx_i=int(obj_i),
                mask_idx_j=int(obj_j),
                relation_name=info.get("name", None),
                relation_score=info.get("score", None),
                relation_emb=rel_emb,      # np.ndarray or list 둘 다 가능(저장 포맷만 일관되게)
                frame_ids=frame_ids,
                object_id=info.get("object_id", None),  # [object_id_i, object_id_j]
            )
            added += 1

        print(f"Added {added} relation edges")
        
    # Object 매핑이 맞는지 확인
    # obj_pcd 중심 vs mask_pcds[mask_idx] 중심 비교
    def debug_check_maskidx_object_alignment(self, tol=1e-6, max_print=30):

        if hasattr(self, "mask_idx_to_object") and self.mask_idx_to_object:
            mapping = self.mask_idx_to_object
        else:
            mapping = {}
            for o in self.objects:
                if hasattr(o, "mask_idx"):
                    mapping[int(o.mask_idx)] = o

        ok, bad, out = 0, 0, 0
        printed = 0

        for k, obj in mapping.items():
            if k >= len(self.mask_pcds):
                out += 1
                if printed < max_print:
                    print("[BAD] mask_idx out of range:", k, "len(mask_pcds)=", len(self.mask_pcds))
                    printed += 1
                continue

            c1 = np.mean(np.asarray(obj.pcd.points), axis=0)
            c2 = np.mean(np.asarray(self.mask_pcds[k].points), axis=0)
            d = np.linalg.norm(c1 - c2)

            if d < tol:
                ok += 1
            else:
                bad += 1
                if printed < max_print:
                    print("[MISMATCH]", k, getattr(obj, "object_id", None), "center_dist=", float(d))
                    printed += 1

        print(f"[debug_check_maskidx_object_alignment] ok={ok}, bad={bad}, out_of_range={out}, mapping_size={len(mapping)}")


    # object pairs 매핑이 맞는지 확인
    # (i, j)가 실제로 mask_pcds[i], mask_pcds[j]를 가리키는지
    # i/j가 매핑된 Object와도 일치하는지
    def debug_check_pairs_match_objects(self, tol_center=1e-6, tol_dist=1e-3, max_print=30):
        if self.object_pairs is None:
            print("[debug_check_pairs_match_objects] self.object_pairs is None")
            return

        # mask_idx -> object 매핑 확보
        if hasattr(self, "mask_idx_to_object") and self.mask_idx_to_object:
            mapping = self.mask_idx_to_object
        else:
            mapping = {}
            for o in self.objects:
                if hasattr(o, "mask_idx"):
                    mapping[int(o.mask_idx)] = o

        ok, bad = 0, 0
        printed = 0

        for (i, j), dist in self.object_pairs.items():
            # mask_pcds 인덱스 범위 체크
            if i >= len(self.mask_pcds) or j >= len(self.mask_pcds):
                bad += 1
                if printed < max_print:
                    print("[BAD] pair index out of range:", (i, j), "len(mask_pcds)=", len(self.mask_pcds))
                    printed += 1
                continue

            # 거리 값이 실제 중심거리와 맞는지 체크 (pair 인덱스가 mask_pcds를 가리키는지 검증)
            ci = np.mean(np.asarray(self.mask_pcds[i].points), axis=0)
            cj = np.mean(np.asarray(self.mask_pcds[j].points), axis=0)
            dist2 = float(np.linalg.norm(ci - cj))
            if abs(dist2 - dist) > tol_dist:
                bad += 1
                if printed < max_print:
                    print("[BAD] dist mismatch:", (i, j), "stored=", float(dist), "recomputed=", dist2)
                    printed += 1
                continue

            # i/j가 실제 Object로도 존재하고, Object.pcd와 mask_pcds[i/j]가 같은지 체크
            oi = mapping.get(int(i), None)
            oj = mapping.get(int(j), None)
            if oi is not None:
                c1 = np.mean(np.asarray(oi.pcd.points), axis=0)
                d1 = float(np.linalg.norm(c1 - ci))
                if d1 > tol_center:
                    bad += 1
                    if printed < max_print:
                        print("[BAD] obj_i mapping mismatch:", i, getattr(oi, "object_id", None), "center_dist=", d1)
                        printed += 1
                    continue
                
            if oj is not None:
                c2 = np.mean(np.asarray(oj.pcd.points), axis=0)
                d2 = float(np.linalg.norm(c2 - cj))
                if d2 > tol_center:
                    bad += 1
                    if printed < max_print:
                        print("[BAD] obj_j mapping mismatch:", j, getattr(oj, "object_id", None), "center_dist=", d2)
                        printed += 1
                    continue

            ok += 1

        print(f"[debug_check_pairs_match_objects] ok={ok}, bad={bad}, num_pairs={len(self.object_pairs)}")
        
        
        
        
        
    ##### Relation graph build    
    def build_relational_graph(self, path):
        # 기본 그래프 생성
        super().build_graph(save_path=path)
        self.save_dir = path

        # 1차 segment_objects 이후 매핑 검증
        self.debug_check_maskidx_object_alignment()
        
        # 관계 그래프 생성
        if not self.cfg.pipeline.get("compute_relations", False):
            return

        print("\n=== Computing object relations ===")
        
        # 3차원 포인트 클라우드 2차원 프레임으로 역투영
        print("Step 1: Object-frame bbox mapping...")
        self.compute_object_frame_bboxes()

        # 검출된 객체 쌍 생성
        print("Step 2: Object pair generation...")
        self.compute_object_pairs()

        # 2차 pair 인덱스가 실제 mask/object와 맞는지 검증
        self.debug_check_pairs_match_objects()

        # 공통 프레임 찾기
        print("Step 3: Relation frame selection...")
        self.compute_relation_frames()

        # 공통 프레임에서의 관계 임베딩 계산 및 저장
        print("Step 4: BLIP relation embeddings...")
        self.compute_blip_relation_embeddings()
        
        # Graph에 관계 edge 추가
        print("Step 5: Attach relation edges to graph...")
        self.attach_relation_edges()
        
        print("=== Relation computation complete ===\n")