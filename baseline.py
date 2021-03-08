from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import multiprocessing as mp
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

'''
* Script assumes data located in DATA_DIR already
'''

DATA_DIR = "aps360_project_db/DATASET"
TRAIN_DATA_DIR = f"{DATA_DIR}/TRAIN"
TEST_DATA_DIR = f"{DATA_DIR}/TEST"

# Option when set to True will load precomputed feature matrices from disk
RELOAD_DATA = False


def gen_features_for_img(image_path):
    img = Image.open(image_path)
    img = img.resize((50, 50), Image.ANTIALIAS)
    img = np.array(img)

    # Some images do not have color channels, normally these images can be ignored
    if len(img.shape) != 3:
        return None

    grey_img = rgb2gray(img)
    hog_features = hog(grey_img, block_norm="L2-Hys", pixels_per_cell=(2, 2))

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
    feature_matrix = PCA(n_components=50).fit_transform(feature_matrix)
    return feature_matrix


def run():
    if not RELOAD_DATA:
        organic_feature_matrix = generate_features_for_folder("aps360_project_db/DATASET/TRAIN/O")
        print(f"Organic feature matrix shape is {organic_feature_matrix.shape}")

        recyclable_feature_matrix = generate_features_for_folder("aps360_project_db/DATASET/TRAIN/R")
        print(f"Recyclable feature matrix shape is {recyclable_feature_matrix.shape}")

        training_features = np.vstack((organic_feature_matrix, recyclable_feature_matrix))
        training_labels = ['1'] * organic_feature_matrix.shape[0] + ['0'] * recyclable_feature_matrix.shape[0]

        pickle.dump(training_features, open("training_features.pkl", "wb"))
        pickle.dump(training_labels, open("training_labels.pkl", "wb"))

        organic_feature_matrix_test = generate_features_for_folder("aps360_project_db/DATASET/TEST/O")
        print(f"Organic testing feature matrix shape is {organic_feature_matrix_test.shape}")

        recyclable_feature_matrix_test = generate_features_for_folder("aps360_project_db/DATASET/TEST/R")
        print(f"Recyclable testing feature matrix shape is {recyclable_feature_matrix_test.shape}")

        testing_features = np.vstack((organic_feature_matrix_test, recyclable_feature_matrix_test))
        testing_labels = ['1'] * organic_feature_matrix_test.shape[0] + ['0'] * recyclable_feature_matrix_test.shape[0]

        pickle.dump(testing_features, open("testing_features.pkl", "wb"))
        pickle.dump(testing_labels, open("testing_labels.pkl", "wb"))

    else:
        training_features = pickle.load(open("training_features.pkl", "rb"))
        training_labels = pickle.load(open("training_labels.pkl", "rb"))
        testing_features = pickle.load(open("testing_features.pkl", "rb"))
        testing_labels = pickle.load(open("testing_labels.pkl", "rb"))

    training_labels = list(map(lambda x: float(x), training_labels))
    testing_labels = list(map(lambda x: float(x), testing_labels))
    training_labels = np.array(training_labels)
    testing_labels = np.array(testing_labels)

    print("Starting to train model")
    svc = SVC(random_state=42, class_weight="balanced", gamma=1/55000, verbose=True, C=0.4, degree=3)
    svc.fit(training_features, training_labels)

    pickle.dump(svc, open("svc.pkl", "wb"))

    training_acc = svc.score(training_features, training_labels)
    print(f"\nDone training model, with training accuracy {int(training_acc*100)}%")

    accuracy = svc.score(testing_features, testing_labels)
    print(f"Model accuracy is: {int(accuracy * 100)}%")

    print(f"Naive model accuracy: {int(accuracy_score(testing_labels, np.array([1] * len(testing_labels))) * 100)}%")


if __name__ == '__main__':
    run()
