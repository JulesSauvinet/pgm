import numpy as np
import glob
import random
import matplotlib.pyplot as plt
import os

from scipy._lib.six import xrange
from sklearn import cluster
from sklearn import neighbors

from scipy.misc import imread, imsave

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

batch = 100

# now load all training examples and cluster them
CLUSTER_FOLDER = os.path.abspath('./data/train/clustered')
training_filenames = sorted(glob.glob('./data/train/*/*.jpg'))

# make directories if they doesn't exist
if not os.path.isdir(CLUSTER_FOLDER):
    os.makedirs(CLUSTER_FOLDER)

for cluster_num in xrange(knn_cls):
    single_cluster_folder = os.path.join(CLUSTER_FOLDER, str(cluster_num))
    if not os.path.isdir(single_cluster_folder):
        os.mkdir(single_cluster_folder)

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