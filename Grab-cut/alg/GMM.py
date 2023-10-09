
import numpy as np
from sklearn.cluster import KMeans

def score_formula(mult,mat):
    score = np.exp(-.5 * mult) / np.sqrt(2 * np.pi)/np.sqrt(np.linalg.det(mat))
    return score

class Model:
    def __init__(self, X, component=5):
        self.n_components = component  # 成分
        self.n_features = X.shape[1]  # 每个成分的维度
        self.n_samples = np.zeros(self.n_components)  # 每个类别包含的数据个数

        self.coefs = np.zeros(self.n_components)  # 每个成分在GMM中的权值
        self.means = np.zeros((self.n_components, self.n_features))  # 每个成分的均值
        self.covariances = np.zeros(
            (self.n_components, self.n_features, self.n_features))  # 每个成分的协方差
        self.init_with_kmeans(X)  # 初始化K-Means模型
        print(X.shape)  # (79663, 3)，属于前景/背景的像素，每个像素维度为3
        print('每个成分的维度：', self.n_features)

    # 初始化KMeans
    def init_with_kmeans(self, X):
        # GMM有5个成分，KMeans中将X聚为5类，label为每个数据的标签
        label = KMeans(n_clusters=self.n_components, n_init=1).fit(X).labels_
        self.fit(X, label)

    def calc_prob(self, X):
        prob = []
        for ci in range(self.n_components):  # 遍历5个成分
            score = np.zeros(X.shape[0])  # 74128个像素对应的score
            if self.coefs[ci] > 0:  # 如果该类别的权重不为0
                diff = X - self.means[ci]
                Tdiff = diff.T
                inv_cov = np.linalg.inv(self.covariances[ci])
                dot = np.dot(inv_cov, Tdiff)
                Tdot = dot.T
                mult = np.einsum('ij,ij->i', diff, Tdot)
                score = score_formula(mult,self.covariances[ci])
            prob.append(score)
        ans = np.dot(self.coefs, prob)  # 权重*概率
        return ans

    def fit(self, X, labels):
        assert self.n_features == X.shape[1]
        self.n_samples[:] = 0
        self.coefs[:] = 0
        # 返回类别以及每个类别包含的数据个数
        uni_labels, count = np.unique(labels, return_counts=True)
        self.n_samples[uni_labels] = count
        variance = 0.01
        # 遍历每个类别
        for ci in uni_labels:
            n = self.n_samples[ci]  # 获取该类别的个数
            # 计算该类在整个分布中的的权重
            sum = np.sum(self.n_samples)
            self.coefs[ci] = n / sum
            # 计算该类别的均值和协方差矩阵
            self.means[ci] = np.mean(X[ci == labels], axis=0)
            if self.n_samples[ci] <= 1:
                self.covariances[ci] = 0
            else:
                self.covariances[ci] = np.cov(X[ci == labels].T)  # 该类的协方差
            det = np.linalg.det(self.covariances[ci])  # 协方差矩阵的行列式
            # 如果行列式小于0再做处理
            if det <= 0:
                self.covariances[ci] += np.eye(self.n_features) * variance
                det = np.linalg.det(self.covariances[ci])


