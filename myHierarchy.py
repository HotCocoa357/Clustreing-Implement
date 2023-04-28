# 自定义凝聚式层次聚类对象 可选连通方式和距离度量
# 随机生成数据点进行测试 聚类结果可视化
import numpy as np
import matplotlib.pyplot as plt
from seaborn import color_palette


class myHierarchicalClustering:
    # 初始化 目标类别数、 连通方式、 距离度量
    def __init__(self, clusterNum=2, linkage='single', distanceType='euclidean'):
        self.clusterNum = clusterNum
        self.linkage = linkage
        self.distanceType = distanceType
        self.clusters = None
        self.distMatrix = None
        self.labels_ = None

    def fit(self, X):
        sampleNum, dimensionNum = X.shape
        # 将单个点初始化为簇
        self.clusters = [[i] for i in range(sampleNum)]
        # 距离矩阵
        self.distMatrix = np.zeros((sampleNum, sampleNum))
        for i in range(sampleNum):
            for j in range(i + 1, sampleNum):
                dist = self.distance(X[i], X[j])
                self.distMatrix[i][j] = dist
                self.distMatrix[j][i] = dist

        # 持续合并 直到达到指定数量
        while len(self.clusters) > self.clusterNum:
            # 寻找最近簇
            min_dist = float('inf')
            indexToMerge = (-1, -1)
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    dist = self.clusterDistance(self.clusters[i], self.clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        indexToMerge = (i, j)

            # 合并最近簇
            self.clusters[indexToMerge[0]] += self.clusters[indexToMerge[1]]
            del self.clusters[indexToMerge[1]]

        # 分配标签
        self.labels_ = np.zeros(sampleNum, dtype=int)
        for i, cluster in enumerate(self.clusters):
            for j in cluster:
                self.labels_[j] = i
        return self.labels_

    # 根据连通方式 计算簇距离
    def clusterDistance(self, cluster1, cluster2):
        dists = [self.distMatrix[i][j] for i in cluster1 for j in cluster2]
        if self.linkage == 'single':
            return min(dists)
        elif self.linkage == 'complete':
            return max(dists)
        elif self.linkage == 'average':
            return sum(dists) / len(dists)
        else:
            raise ValueError('Invalid linkage type')

    # 根据距离类型 计算点距离
    def distance(self, x, y):
        if self.distanceType == 'euclidean':
            return np.sqrt(np.sum((x - y) ** 2))
        elif self.distanceType == 'manhattan':
            return np.sum(abs(x - y))
        else:
            raise ValueError('Invalid distance type')


# 测试部分
np.random.seed(2051518)
X = np.random.rand(40, 2)
myAHC = myHierarchicalClustering(clusterNum=4, linkage='average', distanceType = 'euclidean')
labels = myAHC.fit(X)

# 分配不同颜色
colors = color_palette('husl', n_colors=len(np.unique(labels))).as_hex()

for i in range(myAHC.clusterNum):
    points = X[myAHC.labels_ == i]
    plt.scatter(points[:, 0], points[:, 1], color=colors[i], label=f'Cluster {i + 1}')


plt.title('Hierarchical Clustering Example')
plt.legend(loc='center right', bbox_to_anchor=(1.1, 1))
plt.show()
