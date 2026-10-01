# Dynamic Knowledge Guided Reasoning for Human-Object Interaction Detection

This repository contains our Capstone Project-1 on Human-Object Interaction (HOI) Detection.

The project is based on QPIC (Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information) and extends the pipeline toward graph-based relational reasoning.

## Project Pipeline

Image
→ QPIC HOI Detection
→ Structured HOI Output
→ Hybrid Graph Construction
→ GNN-based Reasoning

## Current Progress

- QPIC inference implemented
- Single-image inference tested
- Multi-image inference developed
- HOI outputs structured for downstream processing
- QPIC evaluation scripts developed
- Hybrid graph construction initiated
- Graph structure and KNN-based relationships analyzed
- Graph construction parameters experimentally investigated

---

# QPIC: Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information

by [Masato Tamura](https://scholar.google.co.jp/citations?user=IbPzCocAAAAJ), [Hiroki Ohashi](https://scholar.google.com/citations?user=GKC6bbYAAAAJ), and Tomoaki Yoshinaga.

This repository contains the official implementation of the paper "[QPIC: Query-Based Pairwise Human-Object Interaction Detection with Image-Wide Contextual Information](https://arxiv.org/abs/2103.05399)", which is accepted to CVPR2021.

<div align="center">
  <img src=".github/overview.png" width="900px" />
</div>

QPIC is implemented by extending the recently proposed object detector, DETR. QPIC leverages the query-based detection and attention mechanism in the transformer, and as a result, achieves high HOI detection performance with simple detection heads.

<div align="center">
  <img src=".github/attention.png" width="900px" />
  <p>Example attention maps.</p>
</div>
