"""
Author: Zihan Li
Date Created: 2021/11/24
Last Modified: 2021/11/24
Python Version: Anaconda 2021.05 (Python 3.8)
"""
import os, sys
from warnings import warn
import matplotlib.pyplot as plt
import numpy as npy
import pandas as pd
import scipy.spatial.distance
from sklearn import cluster, manifold, metrics
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min as argmin_min

class MLChemLab4(object):
    """Template class for Lab4 -*- Data Clustering -*-

    Properties:
        model: KMeans model to fit.
        featurization_mode: Keyword to choose feature methods.
    """
    def __init__(self):
        """Initialize class with empty model and default featurization mode to identical"""
        self.model = None

    def add_model(self, kw : str = "kmeans", cluster_num: int = 8, **kwargs):
        """Add model before fitting and prediction

        Args:
            kw: Keyword that indicates which model to build.
            kwargs: Keyword arguments passed to 
        """
        if kw == "kmeans":
            self.model = cluster.KMeans(n_clusters = cluster_num, random_state = 114514)
        else:
            # Catch incorrect keywords
            raise NotImplementedError("Got incorrect model keyword " + kw)

    def dim_reduction(self, X, dim_reduction_mode: str = "identical",
                      dist_mode: str = "euclidean"):
        """Reduce the dimension of input X data to 2 using preset mode"""

        if dim_reduction_mode == "identical":
            # Do nothing, returns raw X data
            return X
        elif dim_reduction_mode == "pca":
            # Put your dimension reduction code HERE
            pca = PCA(n_components = 2)
            return pca.fit_transform(X)
        elif dim_reduction_mode == "tsne":
            tsne = manifold.TSNE(n_components = 2, random_state = 1919810, 
                                                metric = dist_mode, square_distances = True, n_jobs = -1)
            return tsne.fit_transform(X)
        elif dim_reduction_mode == "isomap":
            isomap = manifold.Isomap(n_components = 2, n_neighbors = 8, 
                                                        metric = dist_mode, n_jobs = -1)
            return isomap.fit_transform(X)
        else:
                # Catch incorrect keywords
            raise NotImplementedError("Got incorrect dimension reduction keyword " 
                                                            + dim_reduction_mode)

    def fit(self, X):
        """Feature input X using given mode and fit model with featurized X
        
        Args:
            X: Input X data.

        Returns:
            Trained model using given X, or None if no model is specified
        """
        # Catch empty model, a model should be added earlier
        if (self.model is None):
            warn("No model to fit. Nothing Returned.")
            return None
        # Preprocess X
        self.model.fit(X)
        return self.model

    def tanimoto_dist(x, y):
        return 1 - (npy.dot(x, y) / (npy.linalg.norm(x, ord = 2) ** 2 
                        + npy.linalg.norm(y, ord = 2) ** 2 - npy.dot(x, y)))

    def cosine_dist(x, y):
        return 1 - (npy.dot(x, y) / (npy.linalg.norm(x, ord = 2) 
                        * npy.linalg.norm(y, ord = 2)))

    def dice_similarity(x, y):
        return 1 - (2 * npy.dot(x, y) / (npy.linalg.norm(x, ord = 1) 
                        + npy.linalg.norm(y, ord = 1)))

    def silhouette_eval(self, X):
        silhouette_euclidean = metrics.silhouette_score(X, self.model.labels_, 
                                                                                        metric = "euclidean")
        silhouette_cosine = metrics.silhouette_score(X, self.model.labels_, 
                                                                                    metric = "cosine")
        return silhouette_euclidean, silhouette_cosine

def search_k_best(model_ml: MLChemLab4, 
                                X: npy.ndarray, tag: npy.ndarray, 
                                k_min: int = 2, k_max: int =10):
    dist_mode_list = ["cosine", "dice", "jaccard"]
    dist_name_list = ["cosine", "dice", "tanimoto"]
    k_best_euc = k_best_cos = max(k_min, 2)
    sse = npy.zeros(k_max - k_min + 1)
    se = npy.zeros(k_max - k_min + 1)
    sc = npy.zeros(k_max - k_min + 1)
    for i in range(0, len(dist_mode_list)):
        output_file = open(".\\tsne_" + dist_name_list[i] 
                                        + "_kmeans.dat", mode = "w")
        X_trans = model_ml.dim_reduction(X, "tsne", dist_mode_list[i]) 
        for k in range(k_min, k_max + 1):
            model_ml.add_model("kmeans", k)
            model_ml.fit(X_trans)
            output_file.write("K = " + str(k) + "\n")
            sse[k - k_min] = model_ml.model.inertia_
            output_file.write("Sum of squared errors = " + str(sse[k - k_min]) + "\n")
            if (k >= 2):
                se[k - k_min], sc[k - k_min] = model_ml.silhouette_eval(X_trans)
                output_file.write("Silhouette score using euclidean distance = " 
                                            + str(se[k - k_min]) + "\n")
                output_file.write("Silhouette score using cosine distance = " 
                                            + str(sc[k - k_min]) + "\n")
                if (se[k - k_min] > se[k_best_euc - k_min]): 
                    k_best_euc = k
                    closest_euc, _ = argmin_min(model_ml.model.cluster_centers_, 
                                                                    X_trans, metric = "euclidean")
                if (sc[k - k_min] > sc[k_best_cos - k_min]): 
                    k_best_cos = k
                    closest_cos, _ = argmin_min(model_ml.model.cluster_centers_, 
                                                                    X_trans, metric = "cosine")
            output_file.write("#" * 20 + "\n")
        output_file.write("Cluster center using euclidean distance: \n")
        for t in range(0, k_best_euc):
            output_file.write("Cluster" + str(t + 1) + ": " + tag[closest_euc[t]][0] + "\n")
        output_file.write("#" * 20 + "\n")
        output_file.write("Cluster center using cosine distance: \n")
        for t in range(0, k_best_cos):
            output_file.write("Cluster" + str(t + 1) + ": " + tag[closest_cos[t]][0] + "\n")
        output_file.close()

        fig = plt.figure(figsize = (9, 6), dpi = 300)
        ax = fig.add_subplot(1, 1, 1)
        pic = ax.plot(npy.linspace(k_min, k_max, k_max - k_min + 1), 
                                sse, marker = "o", label = "tsne_" + dist_name_list[i])  
        ax.set_xlabel("Value of K"); ax.set_ylabel("Distortion")
        ax.legend()
        plt.title("The Elbow Method Using Distortion")
        plt.tight_layout()
        plt.savefig(".\\tsne_" + dist_name_list[i] + "_kmeans_elbow_method.png")

def main():
    os.chdir(sys.path[0])
    zinc_folder_path = "..\\data\\"
    zinc_fp = pd.read_csv(zinc_folder_path + "zinc_fp.csv", index_col = 0)
    zinc_SMILES = pd.read_csv(zinc_folder_path + "zinc_SMILES.csv", index_col = 0)

    my_model = MLChemLab4()
    search_k_best(my_model, zinc_fp.values, zinc_SMILES.values, 1, 10)
  
if __name__ == '__main__':
    main()