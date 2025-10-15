# Proyecto 2: Reducción de la dimensionalidad

import pandas as pd
import matplotlib.pyplot as plt
import gdown
import numpy as np
import os
import struct



def download_data(file_id, name_file):
    # si no existe el archivo file_name
    if os.path.exists(name_file):
        return name_file
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, name_file, quiet=False)
    return name_file


def read_labels(file_path):
    class_names = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }

    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    df = pd.DataFrame(labels, columns=["label"])
    df["class_name"] = df["label"].map(class_names)
    return df


def extrar_feature_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        print("Número de imágenes:", num_images)
        print("Dimensiones de cada imagen:", rows, "x", cols)
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, rows, cols)
        X = images.reshape(num_images, rows * cols)
        print("Forma de la matriz final:", X.shape)
    return X


def Show_Image(X, nro_imagen):
    if nro_imagen < 0 or nro_imagen >= X.shape[0]:
        raise IndexError(f"El índice {nro_imagen} está fuera de rango. Debe estar entre 0 y {X.shape[0] - 1}")

    img = X[nro_imagen].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    plt.title(f"Imagen #{nro_imagen}")
    plt.axis('off')
    plt.show()


# Descargando la data solo la primera vez
file_train_X = download_data("1enziBIpqiv_t95KQcifsclNH2BdR8lAd", "train_X")
file_test_X = download_data("1Jeax6tnQ6Nmr2PTNXdQqzKnN0YqtrLe4", "test_X")
file_train_Y = download_data("1MZtn2iA5cgiYT1i3O0ECuR01oD0kGHh7", "train_Y")
file_test_Y = download_data("1K5pxwk2s3RDYsYuwv8RftJTXZ-RGR7K4", "test_Y")

train_X = extrar_feature_images(file_train_X)
test_X = extrar_feature_images(file_test_X)
train_Y = read_labels(file_train_Y)
test_Y = read_labels(file_test_Y)

print("Data train : ", train_X.shape)
print("Label train : ", train_Y.shape)
print("Data test : ", test_X.shape)
print("Label test : ", test_Y.shape)

image_number = 43
Show_Image(train_X, image_number)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = Preprocess()

    # Dimensionality Reduction with PCA
    X_train_PCA = PCA(X_train, 0.95)
    X_test_PCA = PCA(X_test, 0.95)

    # Dimensionality Reduction with NMF
    X_train_NMF = NMF(X_train, 0.95)
    X_test_NMF = NMF(X_test, 0.95)

    # Dimensionality Reduction with t-SNE
    X_train_tSNE = tSNE(X_train, 2)
    X_test_tSNE = tSNE(X_test, 2)

    # Dimensionality Reduction with UMAP
    X_train_UMAP = UMAP(X_train, 2)
    X_test_UMAP = UMAP(X_test, 2)

    # Dimensionality Reduction with Spectral Embedding
    X_train_SE = SpectralEmbedding(X_train, 2)
    X_test_SE = SpectralEmbedding(X_test, 2)

    # Dimensionality Reduction with Isomap
    X_train_ISO = Isomap(X_train, 2)
    X_test_ISO = Isomap(X_test, 2)

    # Visualize the results
    visualize(X_train_PCA, y_train, "PCA")
    visualize(X_train_NMF, y_train, "NMF")
    visualize(X_train_tSNE, y_train, "t-SNE")
    visualize(X_train_UMAP, y_train, "UMAP")
    visualize(X_train_SE, y_train, "Spectral Embedding")
    visualize(X_train_ISO, y_train, "Isomap")


    # Train and evaluate models
    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNN(),
        "Support Vector Machine": SVM(),
        "Random Forest": RandomForest()
    }

    data = {
        "No Reduction": (X_train, X_test),
        "PCA": (X_train_PCA, X_test_PCA),
        "NMF": (X_train_NMF, X_test_NMF),
        "t-SNE": (X_train_tSNE, X_test_tSNE),
        "UMAP": (X_train_UMAP, X_test_UMAP),
        "Spectral Embedding": (X_train_SE, X_test_SE),
        "Isomap": (X_train_ISO, X_test_ISO)
    }

    results = {}
    for model_name, model in models.items():
        results[model_name] = {}
        for reduction_name, (X_tr, X_te) in data.items():
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            get_result(y_test, y_pred, model_name, reduction_name)










