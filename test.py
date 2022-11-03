import matplotlib.pyplot as plt 
import numpy as np
import umap

from sklearn import datasets, svm, metrics, neighbors, cluster

if __name__ == '__main__':
    digits = datasets.load_digits() #assignment 2
    reducer = umap.UMAP(random_state=42)
    reducer.fit(digits.data)
    embedding = reducer.transform(digits.data)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('UMAP projection of the Digits dataset', fontsize=24)
    mean_imgs = np.array([digits.data[digits.target==i].mean(axis=0) for i in range(10)])
    A = np.array(digits.data) #1797x64
    B = mean_imgs #10x64
    raw_l2_norm = np.linalg.norm(A) #||A||
    imgs_l2_norm = np.linalg.norm(B.T) #||B||
    cosine_sim  = np.dot(A,B.T)/(raw_l2_norm*imgs_l2_norm)
    """ for i in range(0, 10):

        min_idx = np.argmin(cosine_sim[:,i])
        max_idx = np.argmax(cosine_sim[:,i])

        fig, axs = plt.subplots(1,3,figsize=(10,10))
        
        # reference image
        axs[0].set_title(f"{i}")        
        axs[0].imshow(mean_imgs[i].reshape(8,8))
        
        # image and data with highest similarity to reference
        axs[1].set_title(f"{max_idx}, sim:{cosine_sim[max_idx, i]:.3f}, y:{digits.target[max_idx]}")
        axs[1].imshow(digits.data[max_idx].reshape(8,8))
        
        # image and data with lowest similarity to reference
        axs[2].set_title(f"{min_idx}, sim:{cosine_sim[min_idx, i]:.3f}, y:{digits.target[min_idx]}")
        axs[2].imshow(digits.data[min_idx].reshape(8,8))
    """
    num_examples = len(digits.data)
    num_split = int(0.7*num_examples)
    train_features = digits.data[:num_split]
    train_labels =  digits.target[:num_split]
    test_features = digits.data[num_split:]
    test_labels = digits.target[num_split:]
    #KNN
    knn = neighbors.KNeighborsClassifier(n_neighbors=5, algorithm="brute")
    knn.fit(train_features, train_labels)
    predicted_labels = knn.predict(test_features)
    result = metrics.classification_report(test_labels, predicted_labels)
    confusion = metrics.confusion_matrix(test_labels, predicted_labels)
    print(result)
    print(confusion)
    distances, neighbor = knn.kneighbors(X=test_features)
    for i,j in zip(neighbor,range(3)):
        fig = plt.figure(figsize=(10,10))
        outer_grid = fig.add_gridspec(2,1)
        ax0 = fig.add_subplot(outer_grid[0,0])
        ax0.set_title(f"Label:{test_labels[j]}, predicted as:{predicted_labels[j]}")
        ax0.imshow(test_features[j].reshape(8,8))
        inner_grid = outer_grid[1,0].subgridspec(ncols=5, nrows=1, wspace=2)
        for k,subplot,num in zip(i, inner_grid.subplots(),range(5)):
            subplot.set_title(f"Label:{train_labels[i][num]}")
            subplot.imshow(train_features[k].reshape(8,8))
    plt.tight_layout()
    plt.show()
    #Kµ
    kmean = cluster.KMeans(n_clusters = 10)
    clusters = kmean.fit(train_features)
    cluster_centers = kmean.cluster_centers_
    cluster_labels = kmean.predict(train_features)
    print(f"completeness: {metrics.completeness_score( train_labels, cluster_labels)}")
    print(f"homogeneity: {metrics.homogeneity_score( train_labels, cluster_labels)}")
    print(f"adjusted mutual information: {metrics.adjusted_mutual_info_score(train_labels, cluster_labels)}")
    predicted_labels = kmean.predict(test_features)
    result = metrics.classification_report(test_labels, predicted_labels)
    #plt.imshow(cluster_centers)