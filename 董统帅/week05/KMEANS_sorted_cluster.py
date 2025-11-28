import pprint

import numpy as np
import random
import sys

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        max_interval = 0
        while True:
            results = [[] for _ in range(len(self.points))]
            for ele in self.ndarray:
                min_index = -1
                min_distance = float('inf')
                for p_index, point in enumerate(self.points):
                    distance = self.__distance(ele, point)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = p_index
                results[min_index].append(ele)

            new_center = []
            for group_ele_list in results:
                center = np.mean(group_ele_list, axis=0)
                new_center.append(center)
            max_interval += 1
            if np.allclose(new_center, self.points, atol=1e-8) or max_interval > 100:
                break
            else:
                self.points = new_center
        return self.__build_clusters(results)

    def __build_clusters(self, point_list_arr):
        clusters = []
        for center_index, center in enumerate(self.points):
            eles = point_list_arr[center_index]
            sum_dis = self.__sumdis(eles, center)
            clusters.append((eles, sum_dis / len(eles), center))
        return clusters

    def __sumdis(self,result, center = None):
        #计算总距离和
        sum_dis=0
        if center is None:
            for p_index, point in enumerate(self.points):
                for res in result[p_index]:
                    sum_dis += self.__distance(point, res)
        else:
            for res in result:
                sum_dis += self.__distance(center, res)
        return sum_dis

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = p1 - p2
        tmp = np.sum(tmp ** 2) #np.square(arr)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if ndarray.shape[0] < cluster_num:
            return None
        sample_index = random.sample(np.arange(0, ndarray.shape[0], 1).tolist(), cluster_num)
        points = []
        # sample_index = [1, 3, 5, 7 ,9]
        for index in sample_index:
            points.append(ndarray[index])
        return np.array(points)

x = np.random.rand(20, 3)
# x = [[0.81460478, 0.19307407, 0.55477152],
#      [0.98610379, 0.40427675, 0.12331253],
#      [0.27574473, 0.72484532, 0.35902661],
#      [0.23469346, 0.20869655, 0.88249604],
#      [0.34933146, 0.87746318, 0.89811059],
#      [0.86787867, 0.94955277, 0.13763939],
#      [0.3024464 , 0.58560248, 0.18087878],
#      [0.55627955, 0.13669477, 0.27098444],
#      [0.14809008, 0.3022801 , 0.53105441],
#      [0.59574658, 0.80423808, 0.39700644],
#      [0.20358816, 0.832475  , 0.39141121],
#      [0.29899968, 0.45133605, 0.9636294 ],
#      [0.1837001 , 0.18902515, 0.80083918],
#      [0.99627635, 0.55970567, 0.29952509],
#      [0.50575089, 0.82507891, 0.80496458],
#      [0.51227069, 0.42801355, 0.76252769],
#      [0.68653356, 0.86982582, 0.80367491],
#      [0.52951824, 0.82743063, 0.55740846],
#      [0.30979678, 0.15306602, 0.62329344],
#      [0.66416692, 0.45264477, 0.3241089 ]]
# x = np.array(x)
kmeans = KMeansClusterer(x, 5)
clusters = kmeans.cluster()
sorted_cluster = sorted(clusters, key = lambda cls : cls[1])
for index, c in enumerate(sorted_cluster):
    print(index, ' ', c)
