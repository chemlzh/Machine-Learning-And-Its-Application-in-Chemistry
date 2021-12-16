"""
Author: Zihan Li
Date Created: 2021/11/24
Last Modified: 2021/11/24
Python Version: Anaconda 2021.05 (Python 3.8)
"""
import os
import sys
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import numpy as npy
from sklearn import cluster, datasets, manifold, metrics, random_projection
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.decomposition import PCA

def plot2D(X, labels, images, normalized = False, showdigits = False, 
           title = "", legend_title = "", xlabel = "", ylabel = "", 
           save = ".\\2D-plot.png"):
    NCLASS = 10
    # Color for each category
    category_colors = plt.get_cmap("tab10")(npy.linspace(0., 1., NCLASS))
    digit_styles = {"weight": "bold", "size": 8}

    fig = plt.figure(figsize = (7.5, 7.5), dpi = 300)
    ax = fig.add_subplot(1, 1, 1)
    
    if (normalized == True):
        X_trans = MinMaxScaler().fit_transform(X)
    else: X_trans = X
    x_min = X_trans[:,0].min(); x_max = X_trans[:,0].max()
    y_min = X_trans[:,1].min(); y_max = X_trans[:,1].max()
    delta_max = max(x_max - x_min, y_max - y_min)

    # for xy, l in zip(X_trans, labels):
    #     ax.text(*xy, str(l), color = category_colors[l], **digit_styles)
    pic = ax.scatter(X_trans[:,0], X_trans[:,1], c = labels, 
                cmap = "tab10", label = labels, marker = "o") 

    if (showdigits == True):
        image_locs = npy.ones((1, 2), dtype = float)
        for xy, img in zip(X_trans, images):
            dist = npy.sqrt(npy.sum(npy.power(image_locs - xy, 2), axis = 1))
            if npy.min(dist) < 0.05 * delta_max: continue
            thumbnail = offsetbox.OffsetImage(img, zoom = .8, cmap = plt.cm.gray_r)
            imagebox = offsetbox.AnnotationBbox(thumbnail, xy)
            ax.add_artist(imagebox)
            image_locs = npy.vstack([image_locs, xy])
    
    # ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(*pic.legend_elements(), title = legend_title)
    plt.title(title); plt.tight_layout(); plt.savefig(save)

def rand_proj_proc(X, digits, normalized = False, 
                   showdigits = False, file_name = ".\\2D_random_projection"):
    # Project X to 2 random components
    RP = random_projection.SparseRandomProjection(
                    n_components = 2, random_state = 1919810)
    X_projected = RP.fit_transform(X)
    # Labels: digits.target; Images: digits.images
    plot2D(X_projected, digits.target, digits.images, normalized, showdigits,
        title = "Random Projection",
        legend_title = "label", xlabel = "RP1", ylabel = "RP2", 
        save = file_name + ".png")
    return X_projected

def pca_proc(X, digits, normalized = False, 
             showdigits = False, file_name = ".\\2D_pca"):
    # PCA to 2D
    pca = PCA(n_components = 20)
    x_pca_2 = pca.fit_transform(digits['data'])[:,0:2]
    plot2D(x_pca_2, digits.target, digits.images, normalized, showdigits,
        title = "Principal Component Analysis", 
        legend_title = "label", xlabel = "PC1", ylabel = "PC2", 
        save = file_name + ".png")
    return x_pca_2

def tsne_proc(X, digits, normalized = False, 
              showdigits = False, file_name = ".\\2D_tsne"):
    # tSNE to 2D
    tsne = manifold.TSNE(n_components = 2)
    x_tsne_2 = tsne.fit_transform(digits['data'])
    plot2D(x_tsne_2, digits.target, digits.images, normalized, showdigits,
        title = "t-distributed Stochastic Neighbor Embedding", 
        legend_title = "label", xlabel = "tSNE1", ylabel = "tSNE2", 
        save = file_name + ".png")
    return x_tsne_2

def kmeans_proc(X, digits, red_dim = 0, normalized = False, 
                showdigits = False, file_name = ".\\2D_kmeans"):
    if (red_dim == 0):
        # KMeans clustering (No Dim Red)
        kmeans = cluster.KMeans(n_clusters = 10, 
                            random_state = 114514).fit(digits['data'])
        plot2D(X, kmeans.labels_, digits.images, normalized, showdigits,
            title = "KMeans clustering without dimension reduction", 
            legend_title = "kmeans", xlabel = "tSNE1", ylabel = "tSNE2", 
            save = file_name + ".png")
    else:
        # KMeans clustering (Dim Red)
        pca = PCA(n_components = red_dim)
        kmeans = cluster.KMeans(n_clusters = 10, 
                            random_state = 114514).fit(pca.fit_transform(
                            digits['data'])[:,0: red_dim])
        plot2D(X, kmeans.labels_, digits.images, normalized, showdigits,
            title = "KMeans clustering with dimension reduced to " + str(red_dim), 
            legend_title = "kmeans", xlabel = "tSNE1", ylabel = "tSNE2", 
            save = file_name + ".png")

    output_file = open(file_name + ".dat", mode = "w")
    silhouette = metrics.silhouette_score(X, kmeans.labels_, 
                            sample_size = len(X), metric = "euclidean")
    output_file.write("Silhouette score = " + str(silhouette) + "\n")

    output_file.write(f"Homogeneity = {metrics.homogeneity_score(digits['target'], kmeans.labels_):.3f}\n")
    output_file.write(f"Completeness = {metrics.completeness_score(digits['target'], kmeans.labels_):.3f}\n")
    output_file.write(f"V-measure = {metrics.v_measure_score(digits['target'], kmeans.labels_):.3f}\n")

    conf_mat = metrics.confusion_matrix(digits['target'], kmeans.labels_)
    output_file.write("KMeans Confusion Matrix: \n")
    output_file.write(str(conf_mat))

def main():
    os.chdir(sys.path[0])
    NCLASS = 10
    # prepare data
    digits = datasets.load_digits(n_class = NCLASS) 
    # digits datasets. a dict. 2 important keys: 'data' and 'target'
    X = digits.data    # Pixel data from dataset

    x_rand_proj = rand_proj_proc(X, digits, file_name = ".\\2D_random_projection")
    x_pca_2 = pca_proc(X, digits, file_name = ".\\2D_pca")
    x_tsne_2 = tsne_proc(X, digits, file_name = ".\\2D_tsne")

    kmeans_proc(x_tsne_2, digits, red_dim = 0, 
                            file_name = ".\\2D_kmeans")
    for i in range(0, 6):
        kmeans_proc(x_tsne_2, digits, red_dim = 2 ** i, 
                                file_name = ".\\2D_kmeans_red_dim_" + str(2 ** i))

if __name__ == '__main__':
    main()