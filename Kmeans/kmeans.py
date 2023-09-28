import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Pixel:
    def __init__(self, loc_x, loc_y, index, color):
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.index = index
        self.color = color

    def length(self, other):
        return distance(self.color, other.color)

    def min_length(self, _clusters):
        return min(list(map(lambda x: self.length(x), _clusters))), self.index

    def put_into_clusters(self, centers, _clusters):
        _clusters[np.argmin(np.array(list(map(lambda x: self.length(x), centers))))].append(self)


# 计算欧式距离
def distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def new_central(cluster):
    color_cluster = list(map(lambda x: x.color, cluster))
    return Pixel(-1, -1, -1, np.mean(color_cluster, axis=0))


# kmeans核心步骤
def kmeans(centers, p_list):
    origin_clusters = []
    for _i in range(n):
        origin_clusters.append([])
    for _p in p_list:
        _p.put_into_clusters(centers, origin_clusters)
    _new_centrals = list(map(lambda x: new_central(x), origin_clusters))
    return _new_centrals, origin_clusters


# 判断是否继续迭代
def continue_iteration(old, new):
    for order in range(len(old)):
        for _p in old[order]:
            if _p.index not in map(lambda x: x.index, new):
                return False


if __name__ == '__main__':
    # 数据读入与处理
    image = cv2.imread('9.jpg', 1)
    cluster_image = image.copy()
    pixel_list = []
    index_ = 0
    n = int(input('cluster number:'))
    for i in range(len(cluster_image)):
        for j in range(len(cluster_image[0])):
            pixel_list.append(Pixel(i, j, index_, cluster_image[i][j]))
            index_ += 1
    # 使用Kmeans++算法寻找初始的n个中心点
    init_pixel = np.random.randint(0, len(pixel_list))
    centrals = [pixel_list[init_pixel]]
    for i in range(1, n):
        cent_len_list = []
        cent_list = []
        for pixel in pixel_list:
            if pixel.index not in map(lambda x: x.index, centrals):
                pixel_min_len, index = pixel.min_length(centrals)
                cent_len_list.append(pixel_min_len)
                cent_list.append(index)
        p = cent_len_list / np.sum(cent_len_list)
        num = np.random.choice(cent_list, p=p)
        next_cent = pixel_list[num]
        centrals.append(next_cent)
    # 获得第一次聚类的结果
    centrals, clusters = kmeans(centrals, pixel_list)
    new_centrals, new_clusters = kmeans(centrals, pixel_list)
    i = 0
    # 聚类迭代直到满足条件或者上限
    while continue_iteration(clusters, new_clusters) and i < 10000000:
        centrals = new_centrals
        clusters = new_clusters
        new_centrals, new_clusters = kmeans(centrals, pixel_list)
        i += 1
    # 绘制图表
    if n == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([x.color[0] for x in new_clusters[0]], [x.color[1] for x in new_clusters[0]],
                   [x.color[2] for x in new_clusters[0]], c='red', marker='o', label='color 1')
        ax.scatter([x.color[0] for x in new_clusters[1]], [x.color[1] for x in new_clusters[1]],
                   [x.color[2] for x in new_clusters[1]], c='green', marker='o', label='color 2')
        ax.scatter([x.color[0] for x in new_clusters[2]], [x.color[1] for x in new_clusters[2]],
                   [x.color[2] for x in new_clusters[2]], c='blue', marker='o', label='color 3')
        ax.set_xlabel('R-axis')
        ax.set_ylabel('G-axis')
        ax.set_zlabel('B-axis')
        # 添加图例
        ax.legend()
        plt.show()
    # 绘制图像
    for i in range(len(new_centrals)):
        for p in new_clusters[i]:
            cluster_image[p.loc_x][p.loc_y] = new_centrals[i].color
    plt.imsave('9-cluster-' + str(n) + '.png', cluster_image)
