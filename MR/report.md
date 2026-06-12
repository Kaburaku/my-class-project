# 基于百度飞桨的协同过滤电影推荐系统 —— 机器学习实战项目报告

## 一、项目背景

在信息过载的互联网时代，推荐系统已成为连接用户与海量内容的核心技术。无论是视频网站、电商平台还是新闻资讯，个性化推荐都极大提升了用户体验与平台收益。电影推荐作为推荐系统领域的经典场景，其技术演进从基于内容的过滤、协同过滤，发展到深度学习模型，始终是学术界和工业界的研究热点。

百度飞桨（PaddlePaddle）提供了灵活的神经网络构建能力，非常适合实现各类推荐算法。本项目利用飞桨实现一个基于矩阵分解的协同过滤推荐系统，旨在让开发者掌握推荐模型的训练、在线推荐生成以及离线评估的完整流程，同时理解隐语义模型在捕捉用户兴趣和物品属性方面的核心思想。

## 二、需求分析

本项目需要构建一个功能完整的电影推荐引擎，核心需求包含三个模块：

1.  **模型训练模块**：读取 MovieLens 100K 评分数据，实现矩阵分解模型，通过梯度下降优化用户和电影的隐向量以及偏置项，最终保存模型参数。
2.  **Top-K 推荐模块**：加载训练好的模型，针对指定用户，预测其对所有未观看电影的评分，并返回得分最高的K部电影，同时展示电影标题，验证推荐的可解释性。
3.  **离线评估模块**：在测试集上计算经典推荐指标 Hit Rate (HR@K) 和 Normalized Discounted Cumulative Gain (NDCG@K)，从召回率和排序质量两个维度量化模型性能。

非功能性需求方面，系统需支持数据集自动下载，代码结构清晰，保证可复现性和可扩展性。

## 三、关键代码实现说明

### 3.1 数据加载与预处理

项目采用 GroupLens 提供的 MovieLens 100K 数据集，包含943个用户对1682部电影的10万条评分记录。代码通过 `urllib` 自动下载并解压，之后读取 `u.data` 文件构建评分矩阵。用户ID和电影ID均从0开始重新索引，以适配 Embedding 层。数据按8:2随机划分为训练集和测试集，保证评估的独立性。

### 3.2 矩阵分解模型设计

本系统基于经典的 Latent Factor Model，模型结构如下：
```python
class MatrixFactorization(nn.Layer):
    def __init__(self, num_users, num_items, embed_dim=50):
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = self.create_parameter(shape=[1])