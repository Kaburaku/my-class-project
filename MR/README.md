
# 🎬 Movie Recommender System with PaddlePaddle

基于百度飞桨的协同过滤电影推荐系统。使用矩阵分解学习用户与电影的隐向量，提供模型训练、个性化推荐和离线评估三大功能。

## 📋 项目简介

本项目使用 MovieLens 100K 数据集，构建一个轻量级的电影推荐引擎。通过训练矩阵分解模型，可以为任意用户生成 Top-K 电影推荐，并使用 Hit Rate 和 NDCG 指标评估推荐效果。代码易读、注释完整，是飞桨入门推荐系统的极佳实践。

## ✨ 功能特性

- **模型训练 (train.py)**：自动下载 MovieLens 数据集，训练矩阵分解模型并保存。
- **在线推荐 (recommend.py)**：为指定用户生成推荐列表，并打印电影标题。
- **效果评估 (evaluate.py)**：计算 HR@K 和 NDCG@K，量化推荐质量。

## 🚀 快速开始

### 环境要求
- Python >= 3.7
- PaddlePaddle >= 2.4
- NumPy

### 安装依赖
```bash
    pip install paddlepaddle numpy