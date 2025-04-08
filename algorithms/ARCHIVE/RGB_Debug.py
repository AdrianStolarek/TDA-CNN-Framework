from tda_pipelines import SplitRGBChannels
from data_preprocessing import load_cifar10_batch
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf


batch_path = "E:\\ZoryaDyn_Benchmarks\\Objects (CIFAR-10)\\cifar-10-batches-py\\data_batch_1"
raw_data, labels = load_cifar10_batch(batch_path)
print(labels)

print("GPU is", "available" if tf.config.list_physical_devices(
    'GPU') else "not available")

print(raw_data.shape)

data_point = raw_data[368]
print(data_point.shape)

plt.figure(figsize=(5, 5))
plt.imshow(data_point.astype("uint8"))
plt.title("RGB image - Label: {}".format(labels[368]))
plt.axis("off")
plt.show()

data_point2 = raw_data[368, :, :, 1]
lib_split_R, lib_split_G, lib_split_B = cv2.split(data_point)

spliting = SplitRGBChannels()

splited = spliting.transform(data_point)

final1 = splited[0]
final2 = splited[1]
final3 = splited[2]

plt.subplot(1, 3, 1)
plt.imshow(final1.astype("uint8"))
plt.title("RGB image RED - Label: {}".format(labels[368]))
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(final2.astype("uint8"))
plt.title("RGB image GREEN - Label: {}".format(labels[368]))
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(final3.astype("uint8"))
plt.title("RGB image BLUE - Label: {}".format(labels[368]))
plt.axis("off")
plt.show()
