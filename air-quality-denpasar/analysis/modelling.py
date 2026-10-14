import os

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


INPUT_FILE = os.path.join(
    "include",
    "data",
    "air_quality_documentation.csv",
)

OUTPUT_FILE = os.path.join(
    "include",
    "data",
    "model_results.csv",
)

FEATURE_COLUMNS = [
    "pm25",
    "pm1",
    "relativehumidity",
    "temperature",
    "um003",
    "pm25_lag_1",
    "pm25_lag_2",
    "pm25_lag_3",
    "pm25_rolling_mean_3",
    "pm25_rolling_std_3",
]

TARGET_COLUMN = "air_quality_label"

CLASS_LABELS = [
    "Baik",
    "Sedang",
    "Tidak Sehat",
    "Sangat Tidak Sehat",
    "Berbahaya",
]


def load_data():
    print("Membaca dataset dokumentasi...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"File tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Jumlah data: {len(df)}")
    return df


def validate_data(df):
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom yang dibutuhkan tidak ditemukan: "
            f"{missing_columns}"
        )

    print("Validasi kolom berhasil.")


def prepare_data(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    X = X.apply(pd.to_numeric, errors="coerce")

    valid_rows = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    print(f"Data siap digunakan untuk modelling: {len(X)} baris.")

    print("\nDistribusi target:")
    print(y.value_counts())

    return X, y


def train_model(X_train, y_train):
    print("\nMelatih Decision Tree...")

    model = DecisionTreeClassifier(
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("Decision Tree berhasil dilatih.")

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
    )

    print("\n" + "=" * 60)
    print("HASIL EVALUASI MODEL")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy (%): {accuracy * 100:.2f}%")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nUrutan kelas:")
    print(CLASS_LABELS)

    save_results(
        accuracy,
        cm,
    )

    return accuracy, cm

def save_results(accuracy, cm):
    os.makedirs(
        os.path.join("include", "data"),
        exist_ok=True,
)
    results = []

    for i, actual_class in enumerate(CLASS_LABELS):
        for j, predicted_class in enumerate(CLASS_LABELS):
            results.append(
                {
                    "model": "Decision Tree",
                    "accuracy": accuracy,
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "count": int(cm[i, j]),
                }
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nHasil modelling berhasil disimpan: "
        f"{OUTPUT_FILE}"
    )

def main():
    print("=" * 60)
    print("MODELLING - DECISION TREE")
    print("=" * 60)

    df = load_data()

    validate_data(df)

    X, y = prepare_data(df)

    if len(X) < 10:
        raise ValueError(
            "Jumlah data terlalu sedikit untuk modelling."
        )

    if y.nunique() < 2:
        raise ValueError(
            "Target hanya memiliki satu kelas. "
            "Decision Tree membutuhkan minimal dua kelas."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"\nData training: {len(X_train)}")
    print(f"Data testing : {len(X_test)}")

    model = train_model(
        X_train,
        y_train,
    )

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    print("\nModelling selesai.")


if __name__ == "__main__":
    main()