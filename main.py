import glob
import os

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.utils import shuffle


def main(input_image_path, output_image_path, n_clusters):
    """
    功能描述： 该方法可通过 KMean 聚类算法实现对图像像素压缩，其优势是 无需改变图像尺度即可压缩图像
    :param input_image_path: 输入图像路径
    :param output_image_path: 输出图像路径
    :param n_clusters: 压缩图像像素 簇 的大小
    :return:
    """
    images_path_list = glob.glob(f"{input_image_path}/*.jpg")
    for image_path in images_path_list:
        image_base_name = os.path.basename(image_path)

        origin_image = np.array(Image.open(image_path), dtype=np.float32) / 255
        wide, height, depth = tuple(origin_image.shape)

        origin_image_flattened = np.reshape(origin_image, (wide * height, depth))

        # 75% 的像素点拥有训练
        image_array_sample = shuffle(origin_image_flattened, random_state=42)[:int(wide*height*0.75)]
        # 像素点去重
        unique_image_array_sample = np.unique(image_array_sample, axis=0)

        estimator = KMeans(n_clusters=n_clusters, random_state=42)
        estimator.fit(unique_image_array_sample)

        # 预测像素点类别
        cluster_assignments = estimator.predict(origin_image_flattened)
        # 每个簇的质心
        compressed_palette = estimator.cluster_centers_
        compressed_image = np.zeros((wide, height, compressed_palette.shape[1]))

        label_idx = 0

        for i in range(wide):
            for j in range(height):
                # 将每个新像素点的簇的质心代替原像素点
                compressed_image[i][j] = compressed_palette[cluster_assignments[label_idx]]
                label_idx += 1

        Image.fromarray(np.uint8(compressed_image * 255)).convert('RGB').save(f"{output_image_path}/{image_base_name}")


if __name__ == '__main__':
    input_image_path = "./datasets/input_data"
    output_image_path = "./datasets/output_data"
    n_clusters = 128

    main(input_image_path, output_image_path, n_clusters)