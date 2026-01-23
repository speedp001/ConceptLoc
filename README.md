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

The framework consists of three main components:

1. **Hierarchical Scene Graph Construction**
   - The global point cloud is partitioned into floors via height clustering.
   - Each floor is subdivided into rooms using BEV projection and wall-based segmentation.
   - Objects are anchored to room nodes, forming a Floor–Room–Object hierarchy.

2. **Open-vocabulary Object Embedding**
   - Multi-modal CLIP embeddings from:
     - Whole image
     - Bounding box region
     - Segmentation mask region
   - Aggregated into a unified object node embedding.

3. **Open-vocabulary Relation Embedding**
   - Object pairs are processed using BLIP to extract relation-aware visual-language embeddings.
   - Relation edges encode both geometric and semantic interactions.

---

## Applications

| Scene Graph-based Visual Localization | Open-vocabulary Query-based Object Retrieval |
|--------------------------------------|----------------------------------------------|
| <p align="center"><img src="https://i.imgur.com/uCYEZoP.png" width="80%"></p> | <p align="center"><img src="https://i.imgur.com/7XNCWIl.png" width="100%"></p> |
| **Pipeline**<br/>1. Extract object and relation embeddings.<br/>2. Perform cosine similarity matching with global scene graph nodes.<br/>3. Validate matches using relation-edge consistency.<br/>4. Enforce room-level hierarchical constraints.<br/>5. Estimate camera pose using PnP with matched 2D–3D correspondences. | **Pipeline**<br/>1. Convert free-form text query into CLIP embedding.<br/>2. Apply coarse-to-fine hierarchical filtering (Floor → Room → Object).<br/>3. Refine candidates using relation-aware matching. |

---

## Requirements

This project builds upon the implementation and environment settings of **HOV-SG**.  
Please follow the installation instructions and dependency setup provided in the official repository:

🔗 https://github.com/hovsg/HOV-SG?tab=readme-ov-file

All credits for the base infrastructure and environment configuration belong to the original authors of HOV-SG.  
Users are required to prepare the runtime environment according to the guidelines in the above repository before running ConceptLoc.

---
