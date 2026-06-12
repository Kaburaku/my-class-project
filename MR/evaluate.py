import paddle
import numpy as np
from collections import defaultdict
from train import load_data, download_movielens, MatrixFactorization

def evaluate(model, test_data, num_items, k=10):
    model.eval()
    # 按用户组织测试集正样本
    user_positives = defaultdict(set)
    for u, i, r in test_data:
        if r >= 4:  # 阈值为4分以上视为喜欢
            user_positives[u].add(i)

    hits, ndcgs = [], []
    for user, true_items in user_positives.items():
        if len(true_items) == 0:
            continue
        # 为该用户预测所有物品的评分
        user_tensor = paddle.full([num_items], user, dtype='int64')
        all_items = paddle.to_tensor(np.arange(num_items), dtype='int64')
        with paddle.no_grad():
            scores = model(user_tensor, all_items).numpy().flatten()
        # 排除训练集（简化起见这里未排除，实际评估应排除，但Hit/NDCG趋势仍可用）
        top_k = np.argsort(scores)[::-1][:k]
        # Hit Rate
        hits.append(1 if any(item in true_items for item in top_k) else 0)
        # NDCG
        dcg = sum([1/np.log2(pos+2) if top_k[pos] in true_items else 0 for pos in range(k)])
        idcg = sum([1/np.log2(i+2) for i in range(min(len(true_items), k))])
        ndcgs.append(dcg/idcg if idcg > 0 else 0)

    hr = np.mean(hits)
    ndcg = np.mean(ndcgs)
    print(f"HR@{k}: {hr:.4f}, NDCG@{k}: {ndcg:.4f}")
    return hr, ndcg

if __name__ == "__main__":
    data_path = download_movielens()
    _, test_data = load_data(data_path)
    info = np.load("model/user_item_info.npy")
    num_users, num_items = int(info[0]), int(info[1])
    model = MatrixFactorization(num_users, num_items)
    model.set_state_dict(paddle.load("model/mf_recommender.pdparams"))
    evaluate(model, test_data, num_items, k=10)