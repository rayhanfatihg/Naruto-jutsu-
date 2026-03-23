"""
train_svm.py — Train & evaluate an SVM classifier on handsign CSV data
=======================================================================
Run this after collect_data.py has generated data/handsigns.csv.

Usage:
    python tools/train_svm.py

Outputs:
    models/svm_handsign.pkl   — trained SVM + scaler (for use in SequenceMatcher)
"""

import os
import sys
import csv
import pickle
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
_CSV_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')
_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'svm_handsign.pkl')

# Labels (must match collect_data.py SIGN_LABELS order)
SIGN_LABELS = [
    "Tiger", "Serpent", "Ram", "Monkey", "Dragon",
    "Rat", "Bird", "Ox", "Boar", "Horse", "Dog", "Hare",
]

# Feature columns (all except 'label' and 'label_name')
_NUM_HANDS = 2
_NUM_LM    = 21
_FEAT_COLS = [f"h{h}_lm{i}_{c}"
              for h in range(_NUM_HANDS)
              for i in range(_NUM_LM)
              for c in ('x', 'y', 'z')]


def load_csv(path):
    X, y = [], []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row[col]) for col in _FEAT_COLS]
                label = int(row['label'])
                X.append(features)
                y.append(label)
            except (KeyError, ValueError) as e:
                print(f"[train] Skipping malformed row: {e}")
    return np.array(X), np.array(y)


def main():
    print("=== Naruto Jutsu — SVM Handsign Trainer ===\n")

    if not os.path.exists(_CSV_PATH):
        print(f"[train] ERROR: CSV not found at {_CSV_PATH}")
        print("        Run tools/collect_data.py first to generate training data.")
        sys.exit(1)

    # 1. Load data
    X, y = load_csv(_CSV_PATH)
    print(f"[train] Loaded {len(X)} samples, {X.shape[1]} features")

    unique, counts = np.unique(y, return_counts=True)
    print("[train] Class distribution:")
    for label_idx, count in zip(unique, counts):
        name = SIGN_LABELS[label_idx] if label_idx < len(SIGN_LABELS) else str(label_idx)
        print(f"         [{label_idx}] {name:10s} — {count} samples")
    print()

    if len(unique) < 2:
        print("[train] ERROR: Need at least 2 classes. Collect more data!")
        sys.exit(1)

    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train SVM (RBF kernel works well for hand gestures)
    print("[train] Training SVM (RBF kernel)...")
    clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    clf.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean() * 100
    print(f"[train] Test accuracy: {acc:.2f}%\n")

    target_names = [SIGN_LABELS[i] if i < len(SIGN_LABELS) else str(i) for i in unique]
    print(classification_report(y_test, y_pred, target_names=target_names,
                                labels=unique))

    # 6. 5-fold cross-validation
    cv_scores = cross_val_score(clf, X_scaled, y, cv=5)
    print(f"[train] 5-fold CV accuracy: {cv_scores.mean()*100:.2f}% "
          f"(±{cv_scores.std()*100:.2f}%)")

    # 7. Save model (scaler + classifier bundled together)
    os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
    bundle = {'scaler': scaler, 'clf': clf, 'labels': SIGN_LABELS}
    with open(_MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"\n[train] Model saved → {_MODEL_PATH}")
    print("[train] Done! You can now run main.py and it will use this SVM.")


if __name__ == "__main__":
    main()
