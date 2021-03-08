from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import multiprocessing as mp
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

DATA_DIR = "aps360_project_db/DATASET"
TRAIN_DATA_DIR = f"{DATA_DIR}/TRAIN"
TEST_DATA_DIR = f"{DATA_DIR}/TEST"


def gen_features_for_img(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128), Image.ANTIALIAS)
    img = np.array(img, dtype=np.uint8)

    # Some images do not have color channels, normally these images can be ignored
    if len(img.shape) != 3:
        return None

    grey_img = rgb2gray(img)
    hog_features = hog(grey_img, block_norm="L2-Hys", pixels_per_cell=(8, 8))

    color_features = img.flatten()
    flat_features = np.hstack((color_features, hog_features))
    return flat_features


def generate_features_for_folder(folder_path):
    images = glob.glob(f"{folder_path}/*")
    final_results = []

    print(f"\nStarting feature generation for {folder_path} \n")
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(gen_features_for_img, images), total=len(images)))

        for i, result in enumerate(results):
            if result is not None:
                final_results.append(result)

    feature_matrix = np.array(final_results)
    feature_matrix = StandardScaler().fit_transform(feature_matrix)
    feature_matrix = PCA(n_components=500).fit_transform(feature_matrix)
    return feature_matrix


def run():
    organic_feature_matrix = generate_features_for_folder("aps360_project_db/DATASET/TRAIN/O")
    print(f"Organic feature matrix shape is {organic_feature_matrix.shape}")

    recyclable_feature_matrix = generate_features_for_folder("aps360_project_db/DATASET/TRAIN/R")
    print(f"Recyclable feature matrix shape is {recyclable_feature_matrix.shape}")

    training_features = np.vstack((organic_feature_matrix, recyclable_feature_matrix))
    training_labels = ['O'] * organic_feature_matrix.shape[0] + ['R'] * recyclable_feature_matrix.shape[0]

    print("Starting to train model")
    svc = LinearSVC(random_state=42, max_iter=10000, dual=False, fit_intercept=False, class_weight="balanced")
    svc.fit(training_features, training_labels)
    training_acc = svc.score(training_features, training_labels)
    print(f"Done training model, with training accuracy {training_acc}")

    organic_feature_matrix_test = generate_features_for_folder("aps360_project_db/DATASET/TEST/O")
    print(f"Organic testing feature matrix shape is {organic_feature_matrix_test.shape}")

    recyclable_feature_matrix_test = generate_features_for_folder("aps360_project_db/DATASET/TEST/R")
    print(f"Recyclable testing feature matrix shape is {recyclable_feature_matrix_test.shape}")

    testing_features = np.vstack((organic_feature_matrix_test, recyclable_feature_matrix_test))
    testing_labels = ['O'] * organic_feature_matrix_test.shape[0] + ['R'] * recyclable_feature_matrix_test.shape[0]

    predicted_labels = svc.predict(testing_features)
    accuracy = accuracy_score(testing_labels, predicted_labels)

    print(f"Model accuracy is: {int(accuracy * 100)}%")


if __name__ == '__main__':
    run()
