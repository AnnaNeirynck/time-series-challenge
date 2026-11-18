"""HMS EEG classification: simple scikit-learn teaching baseline.

Run in a Kaggle notebook with the HMS competition dataset attached, or locally:
    python hms_sklearn_starter.py --data-dir /path/to/hms-harmful-brain-activity-classification

Requires: numpy, pandas, scipy, scikit-learn, pyarrow.
This is a teaching baseline, not a competitive Kaggle solution.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
from scipy.stats import kurtosis
from xgboost import XGBClassifier


TARGETS = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
FS = 200
# A small, interpretable subset of EEG electrodes (not the EKG channel).
CHANNELS = ["Fp1", "F7", "T3", "T5", "O1", "Fp2", "F8", "T4", "T6", "O2"]
BANDS = [ (0.5, 4),(4, 8), (8, 13), (13,30)]

def extract_features(eeg_path, offset_seconds=0.0):
    """Extract features from the central 10 s of a 50 s annotated segment.

    train.csv offsets refer to the START of each 50 s segment; the labels
    concern its central 10 s. Test EEG files are exactly 50 s long.
    """
    frame = pd.read_parquet(eeg_path)
    channels = [c for c in CHANNELS if c in frame.columns]
    if not channels:
        raise ValueError(f"No expected EEG channels in {eeg_path}")

    start = int(round((float(offset_seconds) + 20.0) * FS))
    end = start + 10 * FS
    if end > len(frame):
        raise ValueError(f"EEG window outside file: {eeg_path}, offset={offset_seconds}")
    segment = frame.iloc[start:end][channels].to_numpy(dtype=np.float64)

    features = []
    for j in range(len(channels)):    #ieder kanaal afzonderlijk afgaan
        x = segment[:, j]
        if not np.isfinite(x).any():
            x = np.zeros_like(x)
        else:
            fill = np.nanmedian(x[np.isfinite(x)])
            x = np.where(np.isfinite(x), x, fill)
        x = np.clip(x, -1000, 1000)
        features.extend([ np.mean(x), np.std(x)])
                #np.ptp is piek tot piek amplitude 

        ###NIEUWE HJORTH FEATURES
        activity= np.var(x)
        dx=np.diff(x)
        ddx=np.diff(dx)
        var_dx=np.var(dx)
        var_ddx=np.var(ddx)
        mobility = ( np.sqrt(var_dx / activity) if activity > 0 else 0)
        mobility_derivative = (np.sqrt(var_ddx / var_dx) if var_dx > 0 else 0)
        complexity = (mobility_derivative / mobility if mobility > 0 else 0)
        features.extend([activity,mobility,complexity])

        ###NIEUWE RELATIVE BAND POWER 
        frequencies, psd = welch(x, fs=FS, nperseg=min(512, len(x)))
        total_mask = (frequencies >= 0.5) & (frequencies < 30)
        total_power = np.sum(psd[total_mask])
        for low, high in BANDS:
            mask = (frequencies >= low) & (frequencies < high)
            # Absolute band power (existing feature)
            band_power = np.sum(psd[mask])
            features.append(np.log1p(band_power))

            # Relative band power (new feature)
            relative_power = ( band_power / total_power if total_power > 0 else 0)
            features.append(relative_power)
        
            
        #NIEUWE SPECTRAL ENTROPY
        if total_power > 0:
          normalized_psd = psd[total_mask] / total_power
          spectral_entropy = -np.sum(normalized_psd * np.log(normalized_psd + 1e-12))
          spectral_entropy /= np.log(len(normalized_psd))
        else:
          spectral_entropy = 0
        features.append(spectral_entropy)

        #NIEUWE KURTOSIS
        if np.std(x) < 1e-6:
            kurt = 0.0
        else:
            kurt = kurtosis(x)
        features.append(kurt)

            #features.append(np.log1p(np.sum(psd[mask])))          
    return np.asarray(features, dtype=np.float32)


def build_features(metadata, eeg_folder, is_train=True):           #1EEG fragment omgezet naar 70 features
    X = []                                                         #matrix --> rij is een trainingsvoorbeeld   
    for i, row in enumerate(metadata.itertuples(index=False)):     #           kolom is een feature
        eeg_path = eeg_folder / f"{row.eeg_id}.parquet"            # = input voor ons model 
        offset = row.eeg_label_offset_seconds if is_train else 0.0
        X.append(extract_features(eeg_path, offset))
        if (i + 1) % 200 == 0:
            print(f"  Extracted {i + 1}/{len(metadata)} recordings")
    X = np.vstack(X)

    print("Feature matrix shape:", X.shape)

    return X


def align_probabilities(model, X, n_classes=6):
    """Keep all six classes even when a small training subset omits one."""
    p = np.zeros((len(X), n_classes), dtype=float)
    p[:, model.classes_.astype(int)] = model.predict_proba(X)
    return np.clip(p, 1e-8, None) / np.clip(p, 1e-8, None).sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/kaggle/input/hms-harmful-brain-activity-classification"))
    parser.add_argument("--max-train", type=int, default=2000, help="Limit examples for a quick classroom run; 0 uses all rows")
    parser.add_argument("--submission", action="store_true", help="Also predict the provided test EEGs")
    args = parser.parse_args()

    train = pd.read_csv(args.data_dir / "train.csv")
    # Keep one annotated window per EEG for a fast, simple starting point.
    train = train.drop_duplicates(subset="eeg_id").reset_index(drop=True)
    if args.max_train > 0 and len(train) > args.max_train:
        train = train.sample(n=args.max_train, random_state=42).reset_index(drop=True)

    # Soft targets are the normalized expert votes; the simple RF fits hard labels.
    votes = train[TARGETS].to_numpy(dtype=float)
    soft_targets = votes / votes.sum(axis=1, keepdims=True)        #volledige verdeling van de expertstemmen
    hard_targets = np.argmax(soft_targets, axis=1)                 #zoekt de positie van de grootste waarde

    print("Extracting training EEG features...")
    X = build_features(train, args.data_dir / "train_eegs", is_train=True)

    # Split by patient, not EEG window, to prevent patient-level leakage.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_train, idx_val = next(splitter.split(X, hard_targets, groups=train["patient_id"]))

   # model = RandomForestClassifier(
   #     n_estimators=150, min_samples_leaf=2, random_state=42, n_jobs=-1)

   # model.fit(X[idx_train], hard_targets[idx_train])

   # val_prob = align_probabilities(model, X[idx_val])

    # XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=6,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    #model.fit(X[idx_train],hard_targets[idx_train])
    #training on soft targets 
    n = len(idx_train)
    X_rep = np.vstack([X[idx_train]] * 6)
    y_rep = np.repeat(np.arange(6), n)
    w_rep = soft_targets[idx_train].T.reshape(-1)
    mask = w_rep > 0
    model.fit(X_rep[mask], y_rep[mask], sample_weight=w_rep[mask])
    val_prob = align_probabilities(model, X[idx_val])

# Clip and normalize probabilities
    val_prob = np.clip(val_prob, 1e-8, None)
    val_prob = val_prob / val_prob.sum(axis=1, keepdims=True)

    val_pred = np.argmax(val_prob, axis=1)
    print(f"Validation accuracy (consensus/majority class): {accuracy_score(hard_targets[idx_val], val_pred):.4f}")
    print(f"Validation log loss (hard labels): {log_loss(hard_targets[idx_val], val_prob, labels=np.arange(6)):.4f}")
    # Kaggle uses KL divergence between expert-vote distributions and predictions.
    y_true = np.clip(soft_targets[idx_val], 1e-12, 1.0)
    kl = np.mean(np.sum(y_true * (np.log(y_true) - np.log(val_prob)), axis=1))
    print(f"Validation KL divergence (soft expert votes): {kl:.4f}")

    if args.submission:
        print("Refitting on all selected training recordings...")
        #model.fit(X, hard_targets)
        #test = pd.read_csv(args.data_dir / "test.csv")
        #print("Extracting test EEG features...")
        #X_test = build_features(test, args.data_dir / "test_eegs", is_train=False)
        #pred = align_probabilities(model, X_test)
        
        n = len(X)
        X_rep = np.vstack([X] * 6)
        y_rep = np.repeat(np.arange(6), n)
        w_rep = soft_targets.T.reshape(-1)
        mask = w_rep > 0
        model.fit(X_rep[mask], y_rep[mask], sample_weight=w_rep[mask])
        print("Classes:", model.classes_)
        test = pd.read_csv(args.data_dir / "test.csv")
        print("Extracting test EEG features...")
        X_test = build_features(test, args.data_dir / "test_eegs", is_train=False)
        # Predict probabilities
        pred = align_probabilities(model, X_test)
        submission = pd.DataFrame(pred, columns=TARGETS)
        submission.insert(0, "eeg_id", test["eeg_id"].to_numpy())
        submission.to_csv("submission.csv", index=False)
        print("Saved submission.csv")


if __name__ == "__main__":
    main()