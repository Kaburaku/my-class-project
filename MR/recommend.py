import paddle
import numpy as np
import os

class MatrixFactorization(paddle.nn.Layer):
    def __init__(self, num_users, num_items, embed_dim=50):
        super().__init__()
        self.user_embed = paddle.nn.Embedding(num_users, embed_dim)
        self.item_embed = paddle.nn.Embedding(num_items, embed_dim)
        self.user_bias = paddle.nn.Embedding(num_users, 1)
        self.item_bias = paddle.nn.Embedding(num_items, 1)
        self.global_bias = self.create_parameter(shape=[1], dtype='float32')

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embed(user_ids)
        item_vec = self.item_embed(item_ids)
        dot = (user_vec * item_vec).sum(1, keepdim=True)
        u_bias = self.user_bias(user_ids)
        i_bias = self.item_bias(item_ids)
        return dot + u_bias + i_bias + self.global_bias

def load_movie_titles(data_dir='data/ml-100k'):
    titles = {}
    with open(os.path.join(data_dir, 'u.item'), 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            titles[int(parts[0])-1] = parts[1]
    return titles

def recommend(user_id, top_k=10):
    info = np.load("model/user_item_info.npy")
    num_users, num_items = int(info[0]), int(info[1])
    model = MatrixFactorization(num_users, num_items)
    model.set_state_dict(paddle.load("model/mf_recommender.pdparams"))
    model.eval()

    # 获取该用户已评分的电影 (从训练数据可重建，这里简单演示用全部电影逐个预测)
    # 实际项目应排除已观看，此处我们读入训练数据构造已观看集合
    all_items = paddle.to_tensor(np.arange(num_items), dtype='int64')
    user_tensor = paddle.full([num_items], user_id, dtype='int64')
    with paddle.no_grad():
        scores = model(user_tensor, all_items).numpy().flatten()

    # 读取电影标题
    titles = load_movie_titles()

    # 推荐 Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]
    print(f"\n🎥 为用户 {user_id} 推荐的电影：")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. {titles[idx]} (预测评分: {scores[idx]:.2f})")

if __name__ == "__main__":
    # 为 ID 为 0 的用户推荐
    recommend(user_id=0, top_k=10)