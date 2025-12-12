
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple
from transformers import BlipProcessor, BlipForConditionalGeneration


class BlipRelationExtractor:
    """
    BLIP-2를 활용해 객체 쌍의 관계 임베딩 추출
    """
    
    def __init__(
        self, 
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = "cuda"
    ):
        """
        Args:
            model_name: Hugging Face 모델 이름
            device: "cuda" or "cpu"
        """
        self.device = device
        print(f"Loading BLIP model: {model_name}")
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()
        
        print("BLIP model loaded successfully")
    
    def extract_union_bbox(
        self, 
        bbox_i: List[float], 
        bbox_j: List[float],
        margin: int = 10
    ) -> List[int]:
        """
        두 bbox의 union bbox 계산 (margin 포함)
        
        Args:
            bbox_i: [xmin, ymin, xmax, ymax]
            bbox_j: [xmin, ymin, xmax, ymax]
            margin: bbox 확장 픽셀 수
            
        Returns:
            union_bbox: [xmin, ymin, xmax, ymax]
        """
        xmin = int(min(bbox_i[0], bbox_j[0]) - margin)
        ymin = int(min(bbox_i[1], bbox_j[1]) - margin)
        xmax = int(max(bbox_i[2], bbox_j[2]) + margin)
        ymax = int(max(bbox_i[3], bbox_j[3]) + margin)
        
        # 음수 방지
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        
        return [xmin, ymin, xmax, ymax]
    
    def crop_union_region(
        self, 
        rgb_image: Image.Image, 
        bbox_i: List[float], 
        bbox_j: List[float],
        margin: int = 10
    ) -> Image.Image:
        """
        RGB 이미지에서 두 객체의 union bbox 영역 crop
        
        Args:
            rgb_image: PIL Image
            bbox_i, bbox_j: 각 객체의 bbox
            margin: bbox 확장 픽셀
            
        Returns:
            cropped_img: PIL Image
        """
        union_bbox = self.extract_union_bbox(bbox_i, bbox_j, margin)
        xmin, ymin, xmax, ymax = union_bbox
        
        # PIL crop: (left, upper, right, lower)
        cropped = rgb_image.crop((xmin, ymin, xmax, ymax))
        return cropped
    
    @torch.no_grad()
    def compute_image_embedding(self, pil_image: Image.Image) -> np.ndarray:
        """
        BLIP vision encoder로 이미지 임베딩 추출
        
        Args:
            pil_image: PIL Image
            
        Returns:
            embedding: (D,) numpy array
        """
        # BLIP processor로 전처리
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Vision encoder 통과
        vision_outputs = self.model.vision_model(**inputs)
        
        # [CLS] 토큰 또는 pooled output 사용
        # BLIP의 경우 vision_outputs.pooler_output 사용
        if hasattr(vision_outputs, "pooler_output"):
            embedding = vision_outputs.pooler_output
        else:
            # 없으면 마지막 hidden state의 평균
            embedding = vision_outputs.last_hidden_state.mean(dim=1)
        
        # (1, D) -> (D,)
        embedding = embedding.squeeze(0).cpu().numpy()
        
        # L2 정규화
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def compute_relation_embeddings(
        self,
        dataset,  # HOV-SG dataset object
        relation_frames: Dict[Tuple[int, int], List[Dict]],
        save_path: str,
        bbox_margin: int = 10,
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        모든 객체 쌍에 대해 관계 임베딩 계산
        
        Args:
            dataset: self.dataset (HM3DSemDataset 등)
            relation_frames: {(obj_i, obj_j): [frame_info_list]}
            save_path: 저장 경로
            bbox_margin: union bbox margin
            
        Returns:
            relation_embeddings: {(obj_i, obj_j): embedding (D,)}
        """
        relation_embeddings = {}
        
        pbar = tqdm(relation_frames.items(), desc="Computing BLIP relation embeddings")
        
        for (obj_i, obj_j), frame_list in pbar:
            embeddings_list = []
            
            for frame_info in frame_list:
                frame_id = frame_info["frame_id"]
                bbox_i = frame_info["bbox_i"]
                bbox_j = frame_info["bbox_j"]
                
                # RGB 이미지 로드
                rgb_image, _, _, _, _ = dataset[frame_id]
                
                # Union bbox crop
                cropped_img = self.crop_union_region(
                    rgb_image, bbox_i, bbox_j, margin=bbox_margin
                )
                
                # BLIP embedding 추출
                embedding = self.compute_image_embedding(cropped_img)
                embeddings_list.append(embedding)
            
            if len(embeddings_list) > 0:
                # 여러 프레임의 임베딩 평균
                avg_embedding = np.mean(embeddings_list, axis=0)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
                relation_embeddings[(obj_i, obj_j)] = avg_embedding
            
            pbar.set_postfix({"pair": f"({obj_i}, {obj_j})"})
        
        # 저장
        np.savez(
            os.path.join(save_path, "edges", "relation_embeddings.npz"),
            **{f"{k[0]}_{k[1]}": v for k, v in relation_embeddings.items()}
        )
        
        print(f"Relation embeddings saved: {len(relation_embeddings)} pairs")
        return relation_embeddings


def load_relation_embeddings(save_path: str) -> Dict[Tuple[int, int], np.ndarray]:
    """
    저장된 관계 임베딩 로드
    
    Args:
        save_path: graph 저장 경로
        
    Returns:
        relation_embeddings: {(obj_i, obj_j): embedding}
    """
    npz_file = os.path.join(save_path, "edges", "relation_embeddings.npz")
    
    if not os.path.exists(npz_file):
        raise FileNotFoundError(f"Relation embeddings not found: {npz_file}")
    
    data = np.load(npz_file)
    relation_embeddings = {}
    
    for key in data.keys():
        obj_i, obj_j = map(int, key.split("_"))
        relation_embeddings[(obj_i, obj_j)] = data[key]
    
    print(f"Loaded {len(relation_embeddings)} relation embeddings")
    return relation_embeddings