from torchvision import transforms, datasets
import torch
import sys
import numpy as np
import pickle
from skimage.feature import hog
import cv2 as cv
from sklearn.cluster import KMeans
from concurrent.futures import ThreadPoolExecutor

eps = sys.float_info.epsilon
torch.cuda.empty_cache()

# Class that performs the SIFT feature extraction, employing the SIFT alghorithm to extract a variable number of keypoints from each frame and then obtaining 
# a fixed-size feature vector by exploiting the Bag of Keypoints technique
class BagOfKeypoints():
    def __init__(self, pretrained, n_clusters, max_keypoints_per_frame, sift_edge_threshold):
        self.n_clusters = n_clusters
        self.max_keypoints_per_frame = max_keypoints_per_frame
        self.sift_edge_threshold = sift_edge_threshold
        if (pretrained == True):
            print("Loading pretrained kmeans")
            with open('./pretrained/kmeans_model.pkl', 'rb') as f:
                self.sift_kmeans = pickle.load(f)
        else:    
            self.sift_kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init='auto', max_iter=300, tol=0.0001,
                                  verbose=0, random_state=None, copy_x=True, algorithm='lloyd')
            print("Creating kmeans model")
    
    # Function to extract SIFT features
    def extract_sift_features_batch(self, images):
        def extract_sift_features(image, nfeatures=self.max_keypoints_per_frame, nOctaveLayers=4, contrastThreshold=0.03, edgeThreshold=self.sift_edge_threshold,
                                  sigma=1.2):
            # nfeatures=500,        Limit to 500 keypoints
            # nOctaveLayers=4,      Use 4 layers per octave
            # contrastThreshold=0.03,  Lower threshold for detecting more features
            # edgeThreshold=15,     Increase threshold to filter out more edge-like keypoints
            # sigma=1.2             Use a slightly lower sigma for the Gaussian blur

            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            sift = cv.SIFT_create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma)
            kp, des = sift.detectAndCompute(gray, None)
            return des

        with ThreadPoolExecutor() as executor:
            descriptors = list(
                executor.map(extract_sift_features, [np.array(img).transpose((1, 2, 0)) for img in images]))
        return descriptors

    # Function that computes the keypoints descriptors of all the frames and than trains a kmeans model on them
    def fit(self, loader):
        sift_descriptors = []
        print("Starting sift features extraction")
        for (imgs, targets) in loader:
            keypoints = self.extract_sift_features_batch(imgs)
            sift_descriptors.extend([des for des in keypoints])
        sift_descriptors = np.vstack(sift_descriptors)

        self.sift_kmeans = self.sift_kmeans.fit(sift_descriptors)

        with open('./pretrained/kmeans_model.pkl', 'wb') as f:
            pickle.dump(self.sift_kmeans, f)

    # Function that obtains a fixed-size feature vector for each frame by computing a normalized histogram representing the distribution of the keypoints extracted from that frame with respecto to the clusters
    def predict(self, loader, labeled):
        all_sift_features = []

        if labeled:
            for imgs, _ in loader:
                sift_batch_vectors = []
                sift_desc_batch = self.extract_sift_features_batch(imgs)
                for des in sift_desc_batch:
                    new_feature_vector = torch.zeros(self.n_clusters)
                    if des is not None:
                        clusters = self.sift_kmeans.predict(des)
                        for cluster in clusters:
                            new_feature_vector[cluster] += 1 / des.shape[0]
                    sift_batch_vectors.append(new_feature_vector)
                all_sift_features.append(torch.stack(sift_batch_vectors))
        else:
            for imgs in loader:
                sift_batch_vectors = []
                sift_desc_batch = self.extract_sift_features_batch(imgs)
                for des in sift_desc_batch:
                    new_feature_vector = torch.zeros(self.n_clusters)
                    if des is not None:
                        clusters = self.sift_kmeans.predict(des)
                        for cluster in clusters:
                            new_feature_vector[cluster] += 1 / des.shape[0]
                    sift_batch_vectors.append(new_feature_vector)
                all_sift_features.append(torch.stack(sift_batch_vectors))

        return all_sift_features

# Class that creates a new loader containing HOG feature vectors, SIFT feature vectors, the frames to be processed by the CNN and the labels for each batch of frames
class GeometricFeatureExtraction():
    def __init__(self, pretrained, n_clusters, max_keypoints_per_frame, sift_edge_threshold):
        self.data_transforms_cnn= transforms.Compose([transforms.Resize((112, 112), antialias=True),])
        self.BoK = BagOfKeypoints(pretrained, n_clusters, max_keypoints_per_frame, sift_edge_threshold)

    # Function to extract HOG features
    def extract_hog_features_batch(self, images):
        def extract_hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
            fd, hog_image = hog(image,
                                pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block,
                                orientations=orientations,
                                block_norm='L2-Hys',
                                visualize=True,
                                channel_axis=2)
            return torch.tensor(fd, dtype=torch.float32)

        with ThreadPoolExecutor() as executor:
            descriptors = list(
                executor.map(extract_hog_features, [np.array(img).transpose((1, 2, 0)) for img in images]))
        return torch.tensor(np.array(descriptors), dtype=torch.float32)

    # Function that builds the new loader
    def get_features(self, loader, labeled = True):
        all_sift_features = self.BoK.predict(loader, labeled)
        new_loader = []
        batch = 0
        if labeled:
            print("Processing labeled data")
            for (imgs, targets) in loader:
                print(f"Batch number: {batch}, starting with hog extraction")
                hog_imgs = self.extract_hog_features_batch(imgs)
                print(f"Exit from: {batch} batch")
                cnn_imgs = self.data_transforms_cnn(imgs)
                new_loader.append((hog_imgs, all_sift_features[batch], cnn_imgs, targets))
                batch += 1
        else:
            print("Processing unlabeled data")
            for imgs in loader:
                print(f"Batch number: {batch}, starting with hog extraction")
                hog_imgs = self.extract_hog_features_batch(imgs)
                print(f"Exit from: {batch} batch")
                cnn_imgs = self.data_transforms_cnn(imgs)
                new_loader.append((hog_imgs, all_sift_features[batch], cnn_imgs))
                batch += 1

        return new_loader