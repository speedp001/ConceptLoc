### Last Update: 2026.01.23

# ConceptLoc

> This repository contains the official implementation of **“Open-vocabulary Relational Scene Graph Generation for Large-scale Scene”**,
> submitted to **IPIU 2026 (제38회 영상처리 및 이해에 관한 워크샵)**.
> 컨퍼런스는 2026년 2월 4–6일, 제주도에서 개최되었습니다.

- **IWAIT 2026 공식 웹사이트**: http://www.ipiu.or.kr/
- **논문 링크**: 

## Index

- [Project Introduction](#project-introduction)  
- [System Overview](#system-overview)
- [Applications](#applications)
- [Requirements](#requirements)

---

## Project Introduction

We propose an Open-vocabulary relational scene graph for large-scale indoor environments.  
Unlike conventional scene graphs relying on predefined object categories and fixed relation sets,  
our method represents both objects and their relationships using open-vocabulary semantic embeddings.  
The scene is organized hierarchically into **floor–room–object** levels, enabling scalable spatial reasoning,  
robust visual localization, and language-driven object retrieval in complex indoor spaces.

---

## System Overview

<p align="center">
  <img src="https://i.imgur.com/MeelM71.jpeg" width="90%">
</p>

Our framework constructs a large-scale indoor scene representation as a **hierarchical open-vocabulary relational scene graph**, where both objects and their relationships are encoded with semantic embeddings and organized in a Floor–Room–Object structure. This design enables scalable reasoning, efficient search, and robust matching in complex indoor environments.

The system consists of three key components:

### 1. Hierarchical Scene Graph Construction
The global 3D point cloud is first aligned to a canonical coordinate system and partitioned along the height axis to separate individual floors.  
Each floor is then projected into a bird’s-eye-view occupancy map, where wall structures are detected and used to segment the space into rooms via density-based clustering and watershed partitioning.  
Object nodes extracted from RGB-D observations are anchored to their corresponding room and floor nodes, forming a hierarchical **Floor → Room → Object** graph structure.  
This hierarchy allows the search space to be progressively reduced and supports coarse-to-fine reasoning in large-scale environments.

### 2. Open-vocabulary Object Embedding
For each detected object, multi-modal visual cues are encoded using CLIP:
- Full-image context
- Bounding box region
- Segmentation mask region

The resulting embeddings are aggregated into a unified object representation that captures both local appearance and global semantic context.  
This enables recognition of objects beyond a closed-set label space and supports free-form language queries.

### 3. Open-vocabulary Relation Embedding
For each object pair, relation-aware visual-language embeddings are extracted using BLIP by jointly observing the two objects within the same frame.  
The resulting relational edges encode not only geometric configurations (e.g., on, inside, next to) but also semantic interactions and contextual relationships, forming a rich relational graph.

<p align="center">
  <img src="https://i.imgur.com/UblqzCe.png" width="90%">
</p>

The final scene graph integrates the hierarchical layout with open-vocabulary embeddings for objects and relations.  
Yellow nodes denote floor-level anchors, blue nodes represent room-level partitions, and object nodes are connected within each room by relation edges.  
Red edges visualize semantically meaningful object–object relations, illustrating how spatial structure and relational context are jointly encoded.  
This unified representation enables robust cross-modal matching, relation-aware localization, and language-driven object retrieval in large-scale indoor environments.

---

## Applications

| Scene Graph-based Visual Localization | Open-vocabulary Query-based Object Retrieval |
|--------------------------------------|----------------------------------------------|
| <p align="center"><img src="https://i.imgur.com/uCYEZoP.png" width="80%"></p> | <p align="center"><img src="https://i.imgur.com/7XNCWIl.png" width="100%"></p> |
| **Pipeline**<br/>1. Extract object and relation embeddings.<br/>2. Perform cosine similarity matching with global scene graph nodes.<br/>3. Validate matches using relation-edge consistency.<br/>4. Enforce room-level hierarchical constraints.<br/>5. Estimate camera pose using PnP with matched 2D–3D correspondences. | **Pipeline**<br/>1. Convert free-form text query into CLIP embedding.<br/>2. Apply coarse-to-fine hierarchical filtering (Floor → Room → Object).<br/>3. Refine candidates using relation-aware matching. |

## Applications

| Visual Localization | Object Retrieval |
|---------------------|------------------|
| <img src="https://i.imgur.com/uCYEZoP.png" width="100%"> | <img src="https://i.imgur.com/7XNCWIl.png" width="100%"> |

### Scene Graph-based Visual Localization
1. Extract object and relation embeddings.
2. Perform cosine similarity matching with global scene graph nodes.
3. Validate matches using relation-edge consistency.
4. Enforce room-level hierarchical constraints.
5. Estimate camera pose using PnP.

### Open-vocabulary Query-based Object Retrieval
1. Convert free-form text query into CLIP embedding.
2. Apply coarse-to-fine hierarchical filtering (Floor → Room → Object).
3. Refine candidates using relation-aware matching.
---

## Requirements

This project builds upon the implementation and environment settings of **HOV-SG**.  
Please follow the installation instructions and dependency setup provided in the official repository:

🔗 https://github.com/hovsg/HOV-SG?tab=readme-ov-file

All credits for the base infrastructure and environment configuration belong to the original authors of HOV-SG.  
Users are required to prepare the runtime environment according to the guidelines in the above repository before running ConceptLoc.

---
