### Last Update: 2026.01.23

# Open-vocabulary Relational Scene Graph Generation for Large-scale Scene

> This repository contains the official implementation of **“Open-vocabulary Relational Scene Graph Generation for Large-scale Scene”**,
> submitted to **IPIU 2026 (제38회 영상처리 및 이해에 관한 워크샵)**.
> 컨퍼런스는 2026년 2월 4–6일, 제주도에서 개최되었습니다.

- **IWAIT 2026 공식 웹사이트**: http://www.ipiu.or.kr/
- **논문 링크**: 

## Index

- [Project Introduction](#project-introduction)  
- [System Overview](#system-overview)
- [Modules Overview](#modules-overview)
- [Experiments](#experiments)
- [Requirements](#requirements)
- [Demo Video](#demo-video)
<br></br>

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
  <img src="/Users/sang-yun/Downloads/IPIU_figure1.drawio.png" width="90%">
</p>

The framework consists of three main components:

1. **Hierarchical Scene Graph Construction**
   - Global point cloud is partitioned into floors via height clustering.
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

## 🧠 Applications

### 1. Scene Graph-based Visual Localization

<p align="center">
  <img src="figures/Fig2_localization.png" width="90%">
</p>

Given a query image:

1. Extract object and relation embeddings.
2. Perform cosine similarity matching with global scene graph nodes.
3. Validate matches using relation-edge consistency.
4. Enforce room-level hierarchical constraints.
5. Estimate camera pose using PnP with matched 2D–3D correspondences.

**Figure 2.**  
*Relation-aware visual localization via joint matching of object nodes and relational edges in the hierarchical scene graph.*

---

### 2. Open-vocabulary Query-based Object Retrieval

<p align="center">
  <img src="figures/Fig3_retrieval.png" width="90%">
</p>

1. Convert free-form text query into CLIP embedding.
2. Apply coarse-to-fine hierarchical filtering (Floor → Room → Object).
3. Refine candidates using relation-aware matching.

**Figure 3.**  
*Open-vocabulary object retrieval using hierarchical pruning and relation-consistent matching in large-scale indoor scenes.*

---

## 📊 Experimental Results

- **Datasets**: HM3D, Replica
- **Hierarchical Accuracy**:
  - Floor classification: 100%
  - Room classification: robust except in open-boundary spaces
- **Object Embedding Evaluation**:
  - Metrics: mAcc, pAcc
- **Relation Embedding Evaluation**:
  - Metric: Confidence score
  - Stable across diverse semantic and geometric relations

---

## 📁 Repository Structure
