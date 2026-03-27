import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
)
from sklearn.decomposition import PCA
import joblib
import warnings

warnings.filterwarnings("ignore")

TRAIN_DIR = r"C:\Users\PRASHANTH\OneDrive\文档\Desktop\SCT_TASK_03\dataset\train"         
IMG_SIZE = (64, 64)          
MAX_SAMPLES = 2000           
TEST_SIZE = 0.2              
RANDOM_STATE = 42
USE_HOG = True               
N_PCA_COMPONENTS = 150       
TUNE_HYPERPARAMS = False     
MODEL_PATH = "svm_model.pkl"

def extract_hog_features(img: np.ndarray) -> np.ndarray:
    """Compute HOG (Histogram of Oriented Gradients) descriptor."""
    win_size = IMG_SIZE
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    return hog.compute(img).flatten()


def _imread_unicode(path: str):
    """
    Read an image safely even when the path contains Unicode / non-ASCII
    characters (common on Windows with Japanese/Chinese folder names).
    cv2.imread silently returns None for such paths; np.fromfile + imdecode
    is the correct workaround.
    """
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_dataset(train_dir: str, max_samples: int = None):
    """
    Load images from:
        <train_dir>/cat/  →  label 0 (cat)
        <train_dir>/dog/  →  label 1 (dog)

    Returns (X, y) numpy arrays.
    """
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")

    def collect(subdir):
        folder = os.path.join(train_dir, subdir)
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(folder, ext)))
        return sorted(paths)

    cat_paths = collect("cat")
    dog_paths = collect("dog")

    
    abs_dir = os.path.abspath(train_dir)
    print(f"\n[DEBUG] train_dir (resolved) : {abs_dir}")
    print(f"[DEBUG] cat images found     : {len(cat_paths)}")
    print(f"[DEBUG] dog images found     : {len(dog_paths)}")
    if cat_paths:
        print(f"[DEBUG] sample cat path      : {cat_paths[0]}")
    if dog_paths:
        print(f"[DEBUG] sample dog path      : {dog_paths[0]}")

    if not cat_paths or not dog_paths:
        raise FileNotFoundError(
            f"\nNo images found.\n"
            f"  Searched : {abs_dir}/cat/  and  {abs_dir}/dog/\n\n"
            f"  Fix options:\n"
            f"    1. Run the script from the folder that contains 'train/'\n"
            f"       e.g.  cd C:\\Users\\You\\Desktop\\SCT_TASK_03  then  python svm_cats_dogs.py\n"
            f"    2. Or set an absolute path at the top of the file:\n"
            f"       TRAIN_DIR = r'C:\\Users\\You\\Desktop\\SCT_TASK_03\\train'"
        )

   
    if max_samples:
        half = max_samples // 2
        cat_paths = cat_paths[:half]
        dog_paths = dog_paths[:half]

    all_paths  = cat_paths + dog_paths
    all_labels = [0] * len(cat_paths) + [1] * len(dog_paths)

    features, labels = [], []
    failed = 0

    for path, label in tqdm(
        zip(all_paths, all_labels), total=len(all_paths), desc="Loading images"
    ):
        img = _imread_unicode(path)
        if img is None:
            failed += 1
            continue

        img = cv2.resize(img, IMG_SIZE)

        if USE_HOG:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            feat = extract_hog_features(gray)
        else:
            feat = img.flatten().astype(np.float32) / 255.0

        features.append(feat)
        labels.append(label)

    if failed:
        print(f"[WARNING] Skipped {failed} unreadable image(s).")

    if not features:
        raise RuntimeError(
            "Zero images were loaded successfully.\n"
            "Possible causes:\n"
            "  1. TRAIN_DIR path is wrong.\n"
            "  2. Files are corrupted.\n"
            "  3. Extensions don't match (.jpg/.jpeg/.png supported)."
        )

    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.int32)
    print(f"\nLoaded {len(X)} images  |  Cats: {(y==0).sum()}  Dogs: {(y==1).sum()}")
    print(f"Feature vector length : {X.shape[1]}")
    return X, y

def build_pipeline(n_pca: int = None) -> Pipeline:
    """StandardScaler → (optional PCA) → RBF-SVM"""
    steps = [("scaler", StandardScaler())]
    if n_pca:
        steps.append(("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)))
    steps.append((
        "svm",
        SVC(kernel="rbf", C=10, gamma="scale",
            probability=True, random_state=RANDOM_STATE),
    ))
    return Pipeline(steps)

def tune_hyperparams(pipeline: Pipeline, X_train, y_train) -> Pipeline:
    param_grid = {
        "svm__C":      [0.1, 1, 10, 100],
        "svm__gamma":  ["scale", "auto", 0.001, 0.01],
        "svm__kernel": ["rbf", "poly"],
    }
    print("\nRunning GridSearchCV (this may take a while)…")
    gs = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1,
                      verbose=2, scoring="accuracy")
    gs.fit(X_train, y_train)
    print(f"Best params : {gs.best_params_}")
    print(f"Best CV acc : {gs.best_score_:.4f}")
    return gs.best_estimator_

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrix.png")


def plot_roc_curve(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — SVM Cats vs Dogs")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=150)
    plt.show()
    print("Saved: roc_curve.png")


def show_sample_predictions(model, X_test, y_test, n=10):
    """Show a grid of test images with predicted vs true labels."""
    indices  = np.random.choice(len(X_test), size=min(n, len(X_test)), replace=False)
    preds    = model.predict(X_test[indices])
    label_map = {0: "Cat", 1: "Dog"}

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Sample Predictions", fontsize=14)
    for ax, idx, pred in zip(axes.flat, indices, preds):
        true  = y_test[idx]
        color = "green" if pred == true else "red"
        ax.set_title(f"Pred: {label_map[pred]}\nTrue: {label_map[true]}",
                     color=color, fontsize=9)
        ax.imshow(np.ones((*IMG_SIZE, 3)) * 0.85, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("sample_predictions.png", dpi=150)
    plt.show()
    print("Saved: sample_predictions.png")

def main():
    print("=" * 55)
    print("  SVM Cats vs Dogs Classifier")
    print("=" * 55)

    X, y = load_dataset(TRAIN_DIR, max_samples=MAX_SAMPLES)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    pipeline = build_pipeline(n_pca=N_PCA_COMPONENTS)

    if TUNE_HYPERPARAMS:
        pipeline = tune_hyperparams(pipeline, X_train, y_train)
    else:
        print("\nFitting SVM pipeline…")
        pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{'─'*40}")
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"{'─'*40}\n")
    print(classification_report(y_test, y_pred, target_names=["Cat", "Dog"]))

    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=5, scoring="accuracy", n_jobs=-1)
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    show_sample_predictions(pipeline, X_test, y_test)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'")

def predict_single_image(image_path: str, model=None):
    """
    Run inference on one image.
    Usage:
        model = joblib.load('svm_cats_dogs.pkl')
        predict_single_image('my_photo.jpg', model)
    """
    if model is None:
        model = joblib.load(MODEL_PATH)

    img = _imread_unicode(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img  = cv2.resize(img, IMG_SIZE)
    feat = (
        extract_hog_features(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).reshape(1, -1)
        if USE_HOG
        else (img.flatten().astype(np.float32) / 255.0).reshape(1, -1)
    )

    pred  = model.predict(feat)[0]
    proba = model.predict_proba(feat)[0]
    label = "Dog" if pred == 1 else "Cat"

    print(f"Prediction : {label}")
    print(f"Confidence : Cat={proba[0]:.2%}  Dog={proba[1]:.2%}")
    return label, proba


if __name__ == "__main__":
    main()