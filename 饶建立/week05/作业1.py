import numpy as np
import random
import sys
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

    def get_cluster_distances(self, result):
        """
        计算每个簇的类内距离统计信息
        返回: 列表，每个元素为 (簇索引, 质心, 样本数量, 总距离, 平均距离, 最大距离)
        """
        cluster_stats = []

        for i in range(len(result)):
            if len(result[i]) == 0:
                # 处理空簇的情况
                cluster_stats.append((i, self.points[i], 0, 0, 0, 0))
                continue

            cluster_points = np.array(result[i])
            centroid = self.points[i]

            # 计算每个点到质心的距离
            distances = []
            for point in cluster_points:
                dist = self.__distance(point, centroid)
                distances.append(dist)

            total_distance = sum(distances)
            avg_distance = total_distance / len(distances) if distances else 0
            max_distance = max(distances) if distances else 0

            cluster_stats.append((i, centroid, len(cluster_points), total_distance, avg_distance, max_distance))

        return cluster_stats

    def sort_clusters_by_intra_distance(self, result, method='average'):
        """
        根据类内距离对簇进行排序

        参数:
        result: 聚类结果
        method: 排序依据
            'average' - 按平均距离排序
            'total'   - 按总距离排序
            'max'     - 按最大距离排序
            'size'    - 按簇大小排序

        返回: 排序后的簇统计信息
        """
        cluster_stats = self.get_cluster_distances(result)

        # 根据选择的方法确定排序键
        if method == 'average':
            sort_key = 4  # 平均距离在元组中的索引
            reverse = True  # 平均距离越大，排序越靠前
        elif method == 'total':
            sort_key = 3  # 总距离在元组中的索引
            reverse = True
        elif method == 'max':
            sort_key = 5  # 最大距离在元组中的索引
            reverse = True
        elif method == 'size':
            sort_key = 2  # 簇大小在元组中的索引
            reverse = True
        else:
            raise ValueError("method参数必须是 'average', 'total', 'max' 或 'size'")

        # 按指定方法排序
        sorted_stats = sorted(cluster_stats, key=lambda x: x[sort_key], reverse=reverse)

        return sorted_stats


    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)


# 测试代码
if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(42)  # 设置随机种子以便复现结果
    x = np.random.rand(100, 8)

    print("数据点数量:", len(x))
    print("数据维度:", x.shape)

    # 进行KMeans聚类
    kmeans = KMeansClusterer(x, 10)
    result, centers, total_distance = kmeans.cluster()

    print("\n=== 原始聚类结果 ===")
    for i in range(len(result)):
        print(f"簇 {i}: {len(result[i])} 个样本")

    print(f"\n总距离和: {total_distance:.4f}")

    # 按不同方法排序并显示结果
    methods = ['average', 'total', 'max', 'size']

    for method in methods:
        print(f"\n=== 按{method}距离排序 ===")
        sorted_clusters = kmeans.sort_clusters_by_intra_distance(result, method=method)

        print(f"{'簇索引':<8} {'样本数':<8} {'总距离':<12} {'平均距离':<12} {'最大距离':<12}")
        print("-" * 60)

        for cluster_info in sorted_clusters:
            idx, centroid, size, total_dist, avg_dist, max_dist = cluster_info
            print(f"{idx:<8} {size:<8} {total_dist:<12.4f} {avg_dist:<12.4f} {max_dist:<12.4f}")

    # 获取排序后的详细簇信息
    print(f"\n=== 按平均距离排序的详细结果 ===")
    sorted_by_avg = kmeans.sort_clusters_by_intra_distance(result, method='average')

    for cluster_info in sorted_by_avg:
        idx, centroid, size, total_dist, avg_dist, max_dist = cluster_info
        print(f"\n簇 {idx} (样本数: {size}):")
        print(f"  质心: {centroid}")
        print(f"  总距离: {total_dist:.4f}")
        print(f"  平均距离: {avg_dist:.4f}")
        print(f"  最大距离: {max_dist:.4f}")