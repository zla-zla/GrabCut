import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import pandas as pd

def color_quantization(image, num_bins):
    quantized_image = np.floor(image / (256 / num_bins)).astype(int)
    return quantized_image

def plot_histogram(image):
    histogram = np.zeros((12, 12, 12), dtype=int)
    height, width, _ = image.shape

    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            histogram[r, g, b] += 1

    flattened_histogram = histogram.flatten()

    plt.bar(range(len(flattened_histogram)), flattened_histogram)
    plt.xlabel('Color Bin')
    plt.ylabel('Frequency')
    plt.show()
    return histogram

# 读取图像
image = cv2.imread('messi5.jpg')  # 替换为你的图像路径
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_array = image_rgb.reshape(-1, 3)
df = pd.DataFrame(image_array, columns=['Red', 'Green', 'Blue'])

plt.figure(1)
sns.kdeplot(data=df['Red'], color='red')
sns.kdeplot(data=df['Green'], color='green')
sns.kdeplot(data=df['Blue'], color='blue')
plt.xlabel('Intensity')
plt.ylabel('Density')
plt.title('RGB Probability Distribution')


image_rgb = (11*(image_rgb/256)).round()
image_array = image_rgb.reshape(-1, 3)
df = pd.DataFrame(image_array, columns=['Red', 'Green', 'Blue'])

plt.figure(2)
sns.kdeplot(data=df['Red'], color='red')
sns.kdeplot(data=df['Green'], color='green')
sns.kdeplot(data=df['Blue'], color='blue')
# sns.histplot(data=df['Red'], bins=12, kde=True, color='red')
# sns.histplot(data=df['Green'], bins=12, kde=True, color='green')
# sns.histplot(data=df['Blue'], bins=12, kde=True, color='blue')
plt.xlabel('Intensity')
plt.ylabel('Density')
plt.title('RGB Probability Distribution')

plt.show()





# 将每个通道的颜色量化为12个不同的值
quantized_image = color_quantization(image, 12)

# 统计直方图
h = plot_histogram(quantized_image)
print()
