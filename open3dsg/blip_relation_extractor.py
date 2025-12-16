import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple
from transformers import AutoProcessor, BlipForImageTextRetrieval

from open3dsg.relation_label_utils import get_relation_label_feats, identify_relation


# BLIP-2를 활용해 객체 쌍의 관계 임베딩 추출
class BlipRelationExtractor:
    
    def __init__(self,  device, model_name = "Salesforce/blip-itm-base-coco"):

        self.device = device
        print(f"Loading BLIP model: {model_name}")
        
        # 이미지를 텐서로 변환하고 텍스트는 토큰 형식으로 변환하는 processor
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("BLIP model loaded successfully")
        
        # 라벨 텍스트 임베딩 캐시 파일
        self.relation_text_feats = None
        self.relation_labels = None

    # 두 객체의 바운딩 박스 합집합 영역
    def crop_union_region(
        self, 
        rgb_image, 
        bbox_i, 
        bbox_j,
        margin
    ):
        # 두 bbox의 union bbox 계산
        xmin = int(min(bbox_i[0], bbox_j[0]) - margin)
        ymin = int(min(bbox_i[1], bbox_j[1]) - margin)
        xmax = int(max(bbox_i[2], bbox_j[2]) + margin)
        ymax = int(max(bbox_i[3], bbox_j[3]) + margin)
        
        # 음수 방지
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        
        # PIL crop: (left, upper, right, lower)
        cropped = rgb_image.crop((xmin, ymin, xmax, ymax))
        return cropped
    
    # BLIP 모델로 임베딩 값 계산
    @torch.no_grad()
    def compute_image_embedding(self, pil_image):
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        vision_out = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        pooled = vision_out.pooler_output  # (1, 512)
        # 차원 자동 변환
        img_feat = self.model.vision_proj(pooled)  # (1, D)
        embedding = img_feat[0].float().cpu().numpy()
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    # 모든 객체 쌍에 대해서 BLIP 관계 임베딩 생성
    def compute_relation_embeddings(
        self,
        dataset,
        relation_frames: Dict[Tuple[int, int], List[Dict]],
        save_path,
        bbox_margin,
        infer_relation_names
    ):

        relation_embeddings = {}
        relation_names = {}

        # 관계 라벨 임베딩 생성
        # 라벨 임베딩을 csv에서 불러와 해당 text를 captioning에 사용
        if infer_relation_names:
            if (getattr(self, "relation_labels", None) is None) or (getattr(self, "relation_text_feats", None) is None):
                this_dir = os.path.dirname(os.path.abspath(__file__))
                relation_csv_path = os.path.join(this_dir, "labels", "relation_labels.csv")
            
                # 라벨 임베딩 로드
                text_feats, labels = get_relation_label_feats(
                    processor=self.processor,
                    model=self.model,
                    csv_path=relation_csv_path,
                    cache_path=os.path.dirname(relation_csv_path),
                    device=self.device,
                )
                
                # text_feats에는 csv에서 순차대로 읽은 텍스트의 임베딩 값이 리스트 형태로 저장
                self.relation_text_feats = text_feats
                # relation_labels에는 csv에서 순차대로 읽은 텍스트 라벨이 리스트 형태로 저장
                self.relation_labels = labels

            else:
                raise ValueError("relation_labels or relation_text_feats is None")

        # obj_i와 obj_j 쌍으로 object id 인덱싱 파일
        mask_idx2object_path = os.path.join(save_path, "graph", "objects", "mask_idx2object.json")
        if os.path.exists(mask_idx2object_path):
            with open(mask_idx2object_path, "r") as f:
                mask_idx2object = json.load(f)
        else:
            raise ValueError(f"Mask idx to object file not found: {mask_idx2object_path}")

        # relation_frames의 리스트 딕셔너리 순회
        pbar = tqdm(relation_frames.items(), desc="Computing BLIP relation embeddings")
        """
        예시
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
            ...
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
        # 리스트 내 딕셔너리에서 값 추출 (두 객체가 잘 보이는 상위 5개의 프레임 정보)
        for (obj_i, obj_j), frame_list in pbar:
            per_frame_embs = []
            frame_ids = []

            for frame_info in frame_list:
                frame_id = frame_info["frame_id"]
                bbox_i = frame_info["bbox_i"]
                bbox_j = frame_info["bbox_j"]

                # dataset[frame_id] -> (rgb_image, depth, pose, _, K)
                rgb_image, _, _, _, _ = dataset[frame_id]

                # rgb_image에서 bbox 합집합 영역 추출
                # Crop union region and compute embedding
                crop = self.crop_union_region(rgb_image, bbox_i, bbox_j, margin=bbox_margin)
                emb = self.compute_image_embedding(crop)
                per_frame_embs.append(emb)
                frame_ids.append(frame_id)

            # 여러 프레임에서 추출한 임베딩 평균화
            avg_emb = np.mean(per_frame_embs, axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
            relation_embeddings[(obj_i, obj_j)] = avg_emb
            """
            relation_embeddings = 
            {
                (0, 1): np.array([0.12, 0.34, -0.11, ..., 0.08]),  # shape: (768,)
                (0, 2): np.array([0.01, 0.22, -0.09, ..., 0.13]),
            ...
            }
            """

            # 라벨 이름과 선택된 프레임에서 읽은 평균 임베딩을 비교하여 가장 유사한 라벨 텍스트 및 매칭 점수 반환
            rel_name, rel_score = None, None
            if infer_relation_names:
                # 평균 임베딩에 해당하는 텍스트 문자와 그 때의 score
                rel_name, rel_score = identify_relation(avg_emb, self.relation_text_feats, self.relation_labels)

            # object id도 함께 저장
            object_id_i = mask_idx2object.get(str(obj_i), None)
            object_id_j = mask_idx2object.get(str(obj_j), None)

            relation_names[(obj_i, obj_j)] = {
                "name": rel_name,
                "score": float(rel_score),
                "relation_emb": avg_emb.tolist(),
                "frame_ids": frame_ids,
                "object_id": [object_id_i, object_id_j],
            }

            pbar.set_postfix({"pair": f"({object_id_i}, {object_id_j})"})

        """
        relation_names =
        {
            "(0, 1)": {
                "name": "on",
                "score": 0.92,
                "relation_emb": [0.12, 0.34, ...],    # 평균 임베딩 벡터
                "frame_ids": [120, 45],               # 사용된 프레임 id 리스트
                "object_id": ["0_0_0", "0_0_2"]
            },
            "(0, 2)": {
                "name": "next to",
                "score": 0.85,
                "relation_emb": [0.11, 0.22, ...],
                "frame_ids": [85],
                "object_id": ["0_0_0", "0_0_3"]
            },
            ...
        }
        """
        
        # edges 디렉토리에 relation_names&relation_embeddings 정보 저장
        if save_path is not None:
            edges_dir = os.path.join(save_path,"graph", "edges")
            os.makedirs(edges_dir, exist_ok=True)

            # Save relation_embeddings in JSON
            relation_embeddings_serializable = {
                f"{k[0]}_{k[1]}": v.tolist() for k, v in relation_embeddings.items()
            }
            with open(os.path.join(edges_dir, "relation_embeddings.json"), "w") as f:
                json.dump(relation_embeddings_serializable, f, indent=2)

            # Save relation_names in JSON
            relation_names_serializable = {
                f"{k[0]}_{k[1]}": v for k, v in relation_names.items()
            }
            with open(os.path.join(edges_dir, "relation_names.json"), "w") as f:
                json.dump(relation_names_serializable, f, indent=2)

        print(f"Relation embeddings saved: {len(relation_embeddings)} pairs")
        print(f"Relation names saved: {len(relation_names)} pairs")

        return relation_embeddings


def load_relation_embeddings(save_path: str) -> Dict[Tuple[int, int], np.ndarray]:

    json_file = os.path.join(save_path,"graph", "edges", "relation_embeddings.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Relation embeddings not found: {json_file}")

    with open(json_file, "r") as f:
        data = json.load(f)

    relation_embeddings = {}
    for key, value in data.items():
        obj_i, obj_j = map(int, key.split("_"))
        relation_embeddings[(obj_i, obj_j)] = np.array(value)
    print(f"Loaded {len(relation_embeddings)} relation embeddings")
    return relation_embeddings


def load_relation_names(save_path: str) -> Dict[Tuple[int, int], Dict]:

    json_file = os.path.join(save_path,"graph", "edges", "relation_names.json")
    
    if not os.path.exists(json_file):
        print(f"Warning: Relation names not found: {json_file}")
        return {}
    
    with open(json_file, "r") as f:
        data = json.load(f)
    
    relation_names = {}
    for key, value in data.items():
        obj_i, obj_j = map(int, key.split("_"))
        relation_names[(obj_i, obj_j)] = value
    
    print(f"Loaded {len(relation_names)} relation names")
    return relation_names