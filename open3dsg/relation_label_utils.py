import os
import csv
import numpy as np
import torch
from typing import List, Tuple
import torch.nn.functional as F


# 관계 라벨 CSV 파일 로드
def load_relation_labels(csv_path: str):
    
    relation_labels = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 스킵
        for row in reader:
            if row:
                relation_labels.append(row[0].strip())
    return relation_labels

# BLIP 텍스트 인코더로 관계 라벨 임베딩 생성
@torch.no_grad()
def compute_relation_text_embeddings(relation_labels, processor, model, device):
    text_embeddings = []
    for label in relation_labels:
        prompt = f"object A is {label} object B"
        inputs = processor(text=prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        text_out = model.text_encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            return_dict=True
        )

        # pooler_output 대신 CLS 사용
        # pooler_output은 CLS에서 활성 함수를 통과한 값인데 BLIP text encoder에서는 사용하지 않음
        # CLS를 직접 뽑아서 차원을 .text_proj에 통과시켜 이미지 임베딩과 차원을 맞춤
        cls = text_out.last_hidden_state[:, 0, :]   # (1, hidden=256)
        txt_feat = model.text_proj(cls) # (1, projection_dim=512)
        txt_feat = F.normalize(txt_feat, dim=-1)

        text_embeddings.append(txt_feat[0].float().cpu().numpy())

    return np.stack(text_embeddings, axis=0)


# 관계 라벨 임베딩 로드 또는 생성 (캐싱 지원)
def get_relation_label_feats(
    processor,
    model,
    csv_path,
    cache_path,
    device):

    cache_file = os.path.join(cache_path, "relation_text_feats.npy")
    labels_file = os.path.join(cache_path, "relation_labels.txt")
    
    # 캐시가 있으면 로드
    if os.path.exists(cache_file) and os.path.exists(labels_file):
        print("Loading cached relation label embeddings...")
        text_feats = np.load(cache_file)
        with open(labels_file, "r") as f:
            relation_labels = [line.strip() for line in f.readlines()]
            """
            "on\n"      → "on"
            "under\n"   → "under"
            "next to\n" → "next to"
            """
        return text_feats, relation_labels
    
    # 캐시 파일이 없으면 생성
    print("Computing relation label embeddings...")
    relation_labels = load_relation_labels(csv_path)
    text_feats = compute_relation_text_embeddings(
        relation_labels, processor, model, device
    )
    
    # 캐시 저장
    os.makedirs(cache_path, exist_ok=True)
    np.save(cache_file, text_feats)
    with open(labels_file, "w") as f:
        for label in relation_labels:
            f.write(label + "\n")
    
    print(f"Relation label embeddings saved: {len(relation_labels)} labels")
    return text_feats, relation_labels

# 관계 임베딩과 텍스트 라벨 임베딩 비교해서 가장 유사한 관계 이름 반환
# (CLIP에서 객체 이름 매칭한 것과 동일한 방식)
def identify_relation(
    relation_emb,
    text_feats,
    relation_labels
):
    # 코사인 유사도 계산
    # 둘 다 정규화가 되어있음
    relation_emb = relation_emb / (np.linalg.norm(relation_emb) + 1e-8)
    similarities = text_feats @ relation_emb
    
    best_idx = np.argmax(similarities)
    best_label = relation_labels[best_idx]
    best_score = float(similarities[best_idx])
    
    return best_label, best_score
