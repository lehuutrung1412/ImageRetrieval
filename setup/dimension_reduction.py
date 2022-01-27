from sklearn.decomposition import PCA


def perform_pca_on_single_vector(ft_np_vector, n_components=2, reshape_dim=2048):
    """
    Given a feature vector, perform dimension reduction using PCA.
    Args:
        ft_np_vector    : numpy feature vector
        n_components    : number of principal components
        reshape_dim     : height of reshaped matrix

    Returns
        PCA performed vector
    """
    pca = PCA(n_components=n_components, whiten=True)
    file_fts = ft_np_vector.reshape(reshape_dim, -1)
    pca.fit(file_fts)
    x = pca.transform(file_fts)
    return x.flatten()
