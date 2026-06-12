import paddle
import paddle.nn as nn
import numpy as np
import os
import urllib.request
import zipfile

# -------------------- 自动下载 MovieLens 100K --------------------
def download_movielens(data_dir='data'):
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = os.path.join(data_dir, 'ml-100k.zip')
    if not os.path.exists(os.path.join(data_dir, 'ml-100k')):
        os.makedirs(data_dir, exist_ok=True)
        print("Downloading MovieLens 100K...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as f:
            f.extractall(data_dir)
        print("Download complete.")
    return os.path.join(data_dir, 'ml-100k')

# -------------------- 数据加载 --------------------
def load_data(data_path):
    ratings = []
    with open(os.path.join(data_path, 'u.data'), 'r') as f:
        for line in f:
            user, item, rating, _ = line.strip().split('\t')
            ratings.append([int(user)-1, int(item)-1, float(rating)])
    ratings = np.array(ratings)
    # 划分训练/测试 (按时间或随机，这里简单随机取20%作为测试)
    np.random.seed(42)
    np.random.shuffle(ratings)
    split = int(0.8 * len(ratings))
    return ratings[:split], ratings[split:]

# -------------------- 矩阵分解模型 --------------------
class MatrixFactorization(nn.Layer):
    def __init__(self, num_users, num_items, embed_dim=50):
        super().__init__()
        self.user_embed = nn.Embedding(num_users, embed_dim)
        self.item_embed = nn.Embedding(num_items, embed_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = self.create_parameter(shape=[1], dtype='float32')

    def forward(self, user_ids, item_ids):
        user_vec = self.user_embed(user_ids)
        item_vec = self.item_embed(item_ids)
        dot = (user_vec * item_vec).sum(1, keepdim=True)
        u_bias = self.user_bias(user_ids)
        i_bias = self.item_bias(item_ids)
        return dot + u_bias + i_bias + self.global_bias

def train():
    data_path = download_movielens()
    train_data, test_data = load_data(data_path)
    num_users = int(max(train_data[:,0].max(), test_data[:,0].max())) + 1
    num_items = int(max(train_data[:,1].max(), test_data[:,1].max())) + 1
    print(f"Users: {num_users}, Items: {num_items}, Train: {len(train_data)}, Test: {len(test_data)}")

    # Dataset
    train_dataset = paddle.io.TensorDataset(
        [paddle.to_tensor(train_data[:,0], dtype='int64'),
         paddle.to_tensor(train_data[:,1], dtype='int64'),
         paddle.to_tensor(train_data[:,2], dtype='float32')]
    )
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = MatrixFactorization(num_users, num_items, embed_dim=50)
    criterion = nn.MSELoss()
    optimizer = paddle.optimizer.Adam(learning_rate=0.005, parameters=model.parameters())

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            u, i, r = batch
            pred = model(u, i)
            loss = criterion(pred, r.unsqueeze(1))
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            total_loss += loss.numpy()[0]
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    os.makedirs("model", exist_ok=True)
    paddle.save(model.state_dict(), "model/mf_recommender.pdparams")
    np.save("model/user_item_info.npy", np.array([num_users, num_items]))
    print("Model saved.")
    return model, test_data

if __name__ == "__main__":
    train()