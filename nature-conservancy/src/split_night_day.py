# https://www.kaggle.com/qiubit/the-nature-conservancy-fisheries-monitoring/detecting-night-photos

import numpy as np
import glob
import random
import matplotlib.pyplot as plt
import os

from scipy._lib.six import xrange
from sklearn import cluster
from sklearn import neighbors

from scipy.misc import imread, imsave

# load images
imgs_to_load = 20

preview_files = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:imgs_to_load]
preview = np.array([imread(img) for img in preview_files])

def show_loaded_with_means(imgs):
    rows_total = int(len(preview) / 4)
    for i in range(rows_total):
        _, img_ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
        _, imgmean_ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
        for j in range(4):
            img = preview[i * 4 + j]
            img_mean = np.mean(img, axis=(0, 1))
            # calculate squared means to amplify green dominance effect
            img_mean = np.power(img_mean, 2)

            # show plots
            img_ax[j].axis('off')
            img_ax[j].imshow(img)
            imgmean_ax[j].bar(range(3), img_mean, width=0.3, color='blue')
            imgmean_ax[j].set_xticks(np.arange(3) + 0.3 / 2)
            imgmean_ax[j].set_xticklabels(['R', 'G', 'B'])


show_loaded_with_means(preview)
plt.show()

imgs_to_load = 20

preview_files = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:imgs_to_load]
preview = np.array([imread(img) for img in preview_files])


def show_loaded_with_mean_differences(imgs):
    rows_total = int(len(preview) / 4)
    for i in range(rows_total):
        _, img_ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
        _, imgmean_ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
        for j in range(4):
            # calculate features of an image
            img = preview[i * 4 + j]
            img_mean = np.mean(img, axis=(0, 1))
            img_features = np.zeros(3)
            img_features[0] = (img_mean[0] - img_mean[1]) + (img_mean[0] - img_mean[2])
            img_features[1] = (img_mean[1] - img_mean[0]) + (img_mean[1] - img_mean[2])
            img_features[2] = (img_mean[2] - img_mean[0]) + (img_mean[2] - img_mean[1])

            # display plots
            img_ax[j].axis('off')
            img_ax[j].imshow(img)
            imgmean_ax[j].bar(range(3), img_features, width=0.3, color='blue')
            imgmean_ax[j].set_xticks(np.arange(3) + 0.3 / 2)
            imgmean_ax[j].set_xticklabels(['R', 'G', 'B'])


show_loaded_with_mean_differences(preview)
plt.show()

# one cluster will be day photos, the other one night photos
knn_cls = 2
# increase this number while training locally for better results
training_imgs = 50

training_files = sorted(glob.glob('../input/train/*/*.jpg'), key=lambda x: random.random())[:training_imgs]
training = np.array([imread(img) for img in training_files])
training_means = np.array([np.mean(img, axis=(0, 1)) for img in training])
training_features = np.zeros((training_imgs, 3))
for i in range(training_imgs):
    training_features[i][0] = (training_means[i][0] - training_means[i][1])
    training_features[i][0] += (training_means[i][0] - training_means[i][2])
    training_features[i][1] = (training_means[i][1] - training_means[i][0])
    training_features[i][1] += (training_means[i][1] - training_means[i][2])
    training_features[i][2] = (training_means[i][2] - training_means[i][0])
    training_features[i][2] += (training_means[i][2] - training_means[i][1])

kmeans = cluster.KMeans(n_clusters=knn_cls).fit(training_features)
print(np.bincount(kmeans.labels_))

def show_four(imgs, title):
    _, ax = plt.subplots(1, 4, sharex='col', sharey='row', figsize=(8, 2))
    plt.suptitle(title, size=8)
    for i, img in enumerate(imgs[:4]):
        ax[i].axis('off')
        ax[i].imshow(img)

for i in range(knn_cls):
    cluster_i = training[np.where(kmeans.labels_ == i)]
    show_four(cluster_i[:4], 'cluster' + str(i))

batch = 100

# now load all training examples and cluster them
CLUSTER_FOLDER = os.path.abspath('../input/train/clustered')
training_filenames = sorted(glob.glob('../input/train/*/*.jpg'))

# make directories if they doesn't exist
if not os.path.isdir(CLUSTER_FOLDER):
    os.makedirs(CLUSTER_FOLDER)

for cluster_num in xrange(knn_cls):
    single_cluster_folder = os.path.join(CLUSTER_FOLDER, str(cluster_num))
    if not os.path.isdir(single_cluster_folder):
        os.mkdir(single_cluster_folder)

print (training_filenames)
saved_files = 0
while saved_files < len(training_filenames):
    training_files = training_filenames[saved_files:saved_files+batch]
    training = np.array([imread(img) for img in training_files])
    training_means = np.array([np.mean(img, axis=(0, 1)) for img in training])
    training_features = np.zeros((training_imgs, 3))
    for i in xrange(len(training)):
        training_features[i][0] = (training_means[i][0] - training_means[i][1])
        training_features[i][0] += (training_means[i][0] - training_means[i][2])
        training_features[i][1] = (training_means[i][1] - training_means[i][0])
        training_features[i][1] += (training_means[i][1] - training_means[i][2])
        training_features[i][2] = (training_means[i][2] - training_means[i][0])
        training_features[i][2] += (training_means[i][2] - training_means[i][1])

    img_cls = kmeans.predict(training_features)

    for i, img in enumerate(training):
        cluster = img_cls[i]
        save_path = os.path.join(CLUSTER_FOLDER, str(cluster))
        class_name = os.path.basename(os.path.dirname(training_files[i]))
        save_path = os.path.join(save_path, class_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, os.path.basename(training_files[i]))
        # print save_path
        imsave(save_path, img)
        saved_files += 1

    print(str(saved_files) + "/" + str(len(training_filenames)))