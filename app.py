import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Kualitas Udara Denpasar",
    page_icon="🌤️",
    layout="wide",
)

st.title("🌤️ Analisis Kualitas Udara Denpasar")
st.caption("Dashboard interaktif berbasis OpenAQ, Open-Meteo, dan NASA POWER")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "processed"

OPENAQ_FILE = DATA_DIR / "denpasar_openaq_30days.csv"
METEO_FILE = DATA_DIR / "denpasar_openmeteo_30days.csv"
NASA_FILE = DATA_DIR / "denpasar_nasa_power.csv"

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    missing = [p.name for p in [OPENAQ_FILE, METEO_FILE, NASA_FILE] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "File berikut belum ditemukan di data/processed/: " + ", ".join(missing)
        )

    openaq = pd.read_csv(OPENAQ_FILE)
    meteo = pd.read_csv(METEO_FILE)
    nasa = pd.read_csv(NASA_FILE)
    return openaq, meteo, nasa

def clean_openaq(df):
    df = df.copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"], errors="coerce", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["location_id", "observed_at", "value"])
    df = df.drop_duplicates(
        subset=["location_id", "sensor_id", "parameter_name", "observed_at"]
    )
    return df.sort_values(["location_id", "observed_at"]).reset_index(drop=True)


def clean_meteo(df):
    df = df.copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"], errors="coerce", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["location_id", "parameter_name", "observed_at", "value"])
    df = df.drop_duplicates(subset=["location_id", "parameter_name", "observed_at"])
    return df.sort_values(["location_id", "observed_at"]).reset_index(drop=True)


def clean_nasa(df):
    df = df.copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"], errors="coerce", utc=True)
    df["parameter_name"] = df["parameter_name"].astype(str).str.upper()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["location_id", "parameter_name", "observed_at", "value"])
    df = df[df["parameter_name"] == "ALLSKY_SFC_SW_DWN"].copy()
    df = df.drop_duplicates(subset=["location_id", "parameter_name", "observed_at"])
    return df.sort_values(["location_id", "observed_at"]).reset_index(drop=True)


@st.cache_data
def prepare_analysis():
    raw_openaq, raw_meteo, raw_nasa = load_data()
    openaq = clean_openaq(raw_openaq)
    meteo = clean_meteo(raw_meteo)
    nasa = clean_nasa(raw_nasa)

    # ---------------- BQ1 ----------------
    pm25 = openaq[openaq["parameter_name"].astype(str).str.lower() == "pm25"].copy()
    pm25["observed_at_local"] = pm25["observed_at"].dt.tz_convert("Asia/Makassar")
    pm25["hour"] = pm25["observed_at_local"].dt.hour

    pm25_by_location = (
        pm25.groupby(["location_id", "location_name"])["value"]
        .agg(count="count", mean="mean", median="median", std="std", minimum="min", maximum="max")
        .reset_index()
    )

    pm25_by_hour = pm25.groupby(["location_name", "hour"])["value"].mean().reset_index()
    pm25_heatmap = pm25_by_hour.pivot(index="location_name", columns="hour", values="value")

    # ---------------- BQ2 ----------------
    pm25_hourly = (
        pm25.assign(observed_at_hour=pm25["observed_at"].dt.floor("h"))
        .groupby(
            ["location_id", "location_name", "latitude", "longitude", "observed_at_hour"],
            as_index=False,
        )["value"]
        .mean()
        .rename(columns={"observed_at_hour": "observed_at", "value": "pm25"})
    )

    meteo_wide = (
        meteo.pivot_table(
            index=["location_id", "observed_at"],
            columns="parameter_name",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    
    meteo_wide.columns.name = None
    bq2 = pd.merge(pm25_hourly, meteo_wide, on=["location_id", "observed_at"], how="inner")

    # ---------------- BQ3 ----------------
    nasa_summary = (
        nasa.groupby("location_id")["value"]
        .agg(
            solar_mean="mean",
            solar_median="median",
            solar_std="std",
            solar_min="min",
            solar_max="max",
        )
        .reset_index()
    )

    bq3 = pd.merge(bq2, nasa_summary, on="location_id", how="left")

    location_summary = (
        bq3.groupby(["location_id", "location_name"], as_index=False)
        .agg(
            pm25_mean=("pm25", "mean"),
            pm25_median=("pm25", "median"),
            pm25_std=("pm25", "std"),
            pm25_min=("pm25", "min"),
            pm25_max=("pm25", "max"),
            jumlah_observasi=("pm25", "count"),
            solar_mean=("solar_mean", "first"),
            solar_median=("solar_median", "first"),
            solar_std=("solar_std", "first"),
        )

        .sort_values("pm25_mean", ascending=False)
    )

    bq3["observed_at"] = pd.to_datetime(bq3["observed_at"], utc=True)
    bq3["observed_at_wita"] = bq3["observed_at"].dt.tz_convert("Asia/Makassar")
    bq3["hour"] = bq3["observed_at_wita"].dt.hour

    return openaq, meteo, nasa, pm25, pm25_by_location, pm25_heatmap, bq2, bq3, location_summary


@st.cache_data
def prepare_ml(bq3):
    """Menyiapkan data machine learning dengan logika yang selaras dengan BQ4 notebook."""
    df = bq3.copy()

    required_columns = [
        "location_id", "observed_at", "pm25",
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "wind_speed_10m", "wind_direction_10m",
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Kolom wajib BQ4 tidak ditemukan: {missing}")

    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True, errors="coerce")
    numeric_columns = [
        "pm25", "temperature_2m", "relative_humidity_2m",
        "precipitation", "wind_speed_10m", "wind_direction_10m",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["location_id", "observed_at", "pm25"]).copy()

    # Satu observasi per lokasi dan timestamp.
    if df.duplicated(["location_id", "observed_at"]).any():
        agg = {col: "median" for col in numeric_columns}
        df = df.groupby(
            ["location_id", "observed_at"], as_index=False
        ).agg(agg)

    df = df.sort_values(["location_id", "observed_at"]).reset_index(drop=True)

    local = df["observed_at"].dt.tz_convert("Asia/Makassar")
    df["hour"] = local.dt.hour
    df["day_of_week"] = local.dt.dayofweek
    df["month"] = local.dt.month

    # Target: rata-rata PM2.5 pada jendela 24 jam yang berakhir pada waktu saat ini.
    rolling_parts = []
    for _, group in df.groupby("location_id", sort=False):
        group = group.sort_values("observed_at").copy()
        group["pm25_24h"] = (
            group.set_index("observed_at")["pm25"]
            .rolling("24h", min_periods=24)
            .mean()
            .to_numpy()
        )
        rolling_parts.append(group)

    df = (
        pd.concat(rolling_parts, ignore_index=True)
        .sort_values(["location_id", "observed_at"])
        .reset_index(drop=True)
    )

    # Representasi sirkular arah angin.
    wind_rad = np.deg2rad(df["wind_direction_10m"])
    df["wind_direction_sin"] = np.sin(wind_rad)
    df["wind_direction_cos"] = np.cos(wind_rad)

    # Fitur cuaca 24 jam sebelumnya, tidak memasukkan waktu target.
    weather_columns = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "wind_speed_10m", "wind_direction_sin", "wind_direction_cos",
    ]
    feature_columns = [
        "temperature_2m_24h", "relative_humidity_2m_24h",
        "precipitation_24h", "wind_speed_10m_24h",
        "wind_direction_sin_24h", "wind_direction_cos_24h",
    ]
    for feature in feature_columns:
        df[feature] = np.nan

    for _, group in df.groupby("location_id", sort=False):
        group = group.sort_values("observed_at")
        indexed = group.set_index("observed_at")
        for source_col in weather_columns:
            rolling = indexed[source_col].rolling(
                "24h", closed="left", min_periods=24
            )
            values = rolling.sum() if source_col == "precipitation" else rolling.mean()
            target_col = (
                "precipitation_24h" if source_col == "precipitation"
                else f"{source_col}_24h"
            )
            df.loc[group.index, target_col] = values.to_numpy()

    def classify_pm25(value):
        if pd.isna(value):
            return np.nan
        if value <= 15.5:
            return "Baik"
        if value <= 55.4:
            return "Sedang"
        return "Tidak Sehat+"

    df["quality_category"] = df["pm25_24h"].apply(classify_pm25)
    features = feature_columns + ["hour", "day_of_week", "month"]
    target = "quality_category"

    model_data = (
        df[["location_id", "observed_at", "pm25", "pm25_24h", *features, target]]
        .dropna(subset=features + [target])
        .copy()
        .sort_values("observed_at")
        .reset_index(drop=True)
    )
    if model_data.empty or model_data["observed_at"].nunique() < 2:
        raise ValueError(
            "Data BQ4 tidak cukup setelah pembuatan fitur 24 jam. "
            "Periksa jumlah data dan kontinuitas timestamp per lokasi."
        )

    unique_times = (
        model_data["observed_at"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    split_position = int(len(unique_times) * 0.80)
    split_position = min(max(split_position, 1), len(unique_times) - 1)
    split_time = unique_times.iloc[split_position]

    train = model_data[model_data["observed_at"] < split_time].copy()
    test = model_data[model_data["observed_at"] >= split_time].copy()
    if train.empty or test.empty:
        raise ValueError("Train atau test kosong; periksa rentang waktu data.")

    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
    }

    results = []
    predictions = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        fitted_models[name] = model
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision Weighted": precision_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "Recall Weighted": recall_score(
                y_test, pred, average="weighted", zero_division=0
            ),
            "F1 Weighted": f1_score(y_test, pred, average="weighted", zero_division=0),
            "F1 Macro": f1_score(y_test, pred, average="macro", zero_division=0),
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values(["F1 Macro", "F1 Weighted"], ascending=False)
        .reset_index(drop=True)
    )

    rf = fitted_models["Random Forest"]
    feature_importance = (
        pd.DataFrame({"Feature": features, "Importance": rf.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    return (
        model_data, train, test, features, results_df,
        predictions, fitted_models, feature_importance,
    )


# ============================================================
# LOAD & PREPARE
# ============================================================

try:
    (
        openaq,
        meteo,
        nasa,
        pm25,
        pm25_by_location,
        pm25_heatmap,
        bq2,
        bq3,
        location_summary,
    ) = prepare_analysis()
    (
        model_data,
        train_data,
        test_data,
        features,
        results_df,
        predictions,
        fitted_models,
        feature_importance,
    ) = prepare_ml(bq3)

except Exception as e:
    st.error("Data belum dapat diproses.")
    st.exception(e)
    st.info(
        "Pastikan struktur folder adalah `data/processed/` dan berisi tiga CSV dari pipeline kamu."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Navigasi")
page = st.sidebar.radio(
    "Pilih halaman",
    [
        "Dashboard",
        "Analisis PM2.5",
        "Faktor Meteorologi",
        "Lingkungan NASA POWER",
        "Machine Learning",
        "Data",
    ],

)

st.sidebar.divider()
st.sidebar.caption("Sumber data")
st.sidebar.caption("• OpenAQ")
st.sidebar.caption("• Open-Meteo")
st.sidebar.caption("• NASA POWER")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":
    st.header("Dashboard")
    st.write(
        "Ringkasan kondisi PM2.5 dan data lingkungan pada lokasi pengamatan di Kota Denpasar."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lokasi", int(pm25["location_id"].nunique()))
    c2.metric("Data PM2.5", f"{len(pm25):,}")
    c3.metric("Rata-rata PM2.5", f"{pm25['value'].mean():.2f} µg/m³")
    highest = pm25_by_location.sort_values("mean", ascending=False).iloc[0]
    c4.metric("PM2.5 Tertinggi", f"{highest['mean']:.2f} µg/m³")
    if not pm25.empty:
        st.caption(
            f"Rentang pengamatan PM2.5 (WITA): "
            f"{pm25['observed_at_local'].min():%d %b %Y %H:%M} hingga "
            f"{pm25['observed_at_local'].max():%d %b %Y %H:%M}."
        )

    st.subheader("Rata-rata PM2.5 per Lokasi")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=pm25_by_location.sort_values("mean", ascending=False), x="mean", y="location_name", ax=ax)
    ax.set_xlabel("Rata-rata PM2.5 (µg/m³)")
    ax.set_ylabel("Lokasi")
    ax.set_title("Rata-rata Konsentrasi PM2.5 Berdasarkan Lokasi")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Konsentrasi PM2.5 terhadap Waktu")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.scatterplot(data=pm25, x="observed_at", y="value", hue="location_name", alpha=0.6, ax=ax)
    ax.set_xlabel("Waktu")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Konsentrasi PM2.5 terhadap Waktu")
    ax.legend(title="Lokasi", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# ANALISIS PM2.5
# ============================================================

elif page == "Analisis PM2.5":
    st.header("Analisis PM2.5")

    locations = sorted(pm25["location_name"].dropna().unique())
    selected = st.multiselect("Pilih lokasi", locations, default=locations)
    filtered = pm25[pm25["location_name"].isin(selected)].copy()

    if filtered.empty:
        st.warning("Tidak ada data untuk lokasi yang dipilih.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah observasi", f"{len(filtered):,}")
    c2.metric("Rata-rata", f"{filtered['value'].mean():.2f} µg/m³")
    c3.metric("Maksimum", f"{filtered['value'].max():.2f} µg/m³")

    tab1, tab2, tab3 = st.tabs(["Tren Waktu", "Distribusi", "Pola Jam"])
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.scatterplot(data=filtered, x="observed_at", y="value", hue="location_name", alpha=0.55, ax=ax)
        ax.set_xlabel("Waktu")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title("Konsentrasi PM2.5 terhadap Waktu")
        ax.legend(title="Lokasi", bbox_to_anchor=(1.02, 1), loc="upper left")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=filtered, x="location_name", y="value", ax=ax)
        ax.set_xlabel("Lokasi")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title("Distribusi PM2.5 Berdasarkan Lokasi")
        ax.tick_params(axis="x", rotation=20)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.dataframe(
            filtered.groupby("location_name")["value"]
            .agg(Jumlah="count", Rata_rata="mean", Median="median", Minimum="min", Maksimum="max")
            .reset_index(),
            use_container_width=True,
        )

    with tab3:
        hourly = (
            filtered.groupby(["location_name", "hour"])["value"]
            .mean()
            .reset_index()
        )
        heat = hourly.pivot(index="location_name", columns="hour", values="value")
        fig, ax = plt.subplots(figsize=(14, 4.8))
        sns.heatmap(heat, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
        ax.set_title("Rata-rata PM2.5 Berdasarkan Lokasi dan Jam")
        ax.set_xlabel("Jam")
        ax.set_ylabel("Lokasi")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ============================================================
# FAKTOR METEOROLOGI
# ============================================================

elif page == "Faktor Meteorologi":
    st.header("Faktor Meteorologi")
    st.write("Menghubungkan PM2.5 dengan temperatur, kelembapan, curah hujan, dan kecepatan angin.")

    variable_map = {
        "Temperatur": ("temperature_2m", "Temperatur (°C)"),
        "Kelembapan": ("relative_humidity_2m", "Relative Humidity (%)"),
        "Curah hujan": ("precipitation", "Precipitation"),
        "Kecepatan angin": ("wind_speed_10m", "Wind Speed"),
    }

    choice = st.selectbox("Variabel meteorologis", list(variable_map.keys()))
    xcol, xlabel = variable_map[choice]

    if xcol not in bq2.columns:
        st.warning(f"Kolom `{xcol}` tidak tersedia pada data Open-Meteo.")

    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=bq2, x=xcol, y="pm25", hue="location_name", alpha=0.6, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"Hubungan {choice} dan PM2.5")
        ax.legend(title="Lokasi", bbox_to_anchor=(1.02, 1), loc="upper left")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


    corr_columns = ["pm25", "temperature_2m", "relative_humidity_2m", "precipitation", "wind_speed_10m"]
    available_corr = [c for c in corr_columns if c in bq2.columns]
    corr = bq2[available_corr].corr()

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation PM2.5 dan Faktor Meteorologis")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ============================================================
# NASA POWER
# ============================================================

elif page == "Lingkungan NASA POWER":
    st.header("Karakteristik Lingkungan NASA POWER")
    st.write("NASA POWER digunakan pada notebook sebagai faktor lingkungan tambahan, yaitu radiasi matahari.")
    st.metric("Parameter", "ALLSKY_SFC_SW_DWN")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=location_summary, x="solar_mean", y="location_name", ax=ax)
        ax.set_xlabel("Rata-rata Solar Radiation")
        ax.set_ylabel("Lokasi")
        ax.set_title("Rata-rata Solar Radiation NASA POWER")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=location_summary, x="solar_mean", y="pm25_mean", hue="location_name", s=150, ax=ax)
        ax.set_xlabel("Rata-rata Solar Radiation")
        ax.set_ylabel("Rata-rata PM2.5 (µg/m³)")
        ax.set_title("Solar Radiation dan Rata-rata PM2.5")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Ringkasan per Lokasi")
    st.dataframe(location_summary, use_container_width=True)

    # Audit konsistensi waktu, mengikuti notebook BQ3.
    pm_start, pm_end = bq2["observed_at"].min(), bq2["observed_at"].max()
    nasa_start, nasa_end = nasa["observed_at"].min(), nasa["observed_at"].max()
    overlap = (
        pd.notna(pm_start) and pd.notna(pm_end)
        and pd.notna(nasa_start) and pd.notna(nasa_end)
        and max(pm_start, nasa_start) <= min(pm_end, nasa_end)
    )

    st.subheader("Audit kesesuaian data")
    st.write(f"Rentang PM2.5 + meteorologi (UTC): **{pm_start}** hingga **{pm_end}**")
    st.write(f"Rentang NASA POWER (UTC): **{nasa_start}** hingga **{nasa_end}**")
    if not overlap:
        st.warning(
            "Rentang waktu PM2.5 dan NASA POWER tidak bertumpang tindih. "
            "Jangan menafsirkan grafik gabungan sebagai bukti hubungan temporal."
        )
    else:
        st.info(
            "Rentang tanggal kedua dataset bertumpang tindih. Namun, ringkasan NASA POWER "
            "digabungkan berdasarkan lokasi, bukan dicocokkan per jam; hubungan sebab-akibat "
            "tidak dapat disimpulkan dari grafik ini."
        )

    if location_summary["solar_mean"].nunique() <= 1:
        st.warning(
            "Nilai rata-rata solar_mean sama pada seluruh lokasi. Grafik ini tidak dapat "
            "menunjukkan perbedaan radiasi matahari antar lokasi pada data yang tersedia."
        )


# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "Machine Learning":
    st.header("Machine Learning")
    st.write("Klasifikasi kategori kualitas udara berdasarkan fitur meteorologis dan lingkungan.")

    st.info(
        "Kategori target mengikuti notebook: PM2.5 ≤ 15.5 = Baik, 15.5–55.4 = Sedang, dan > 55.4 = Tidak Sehat+."
    )
    st.caption("Fitur machine learning diselaraskan dengan BQ4: menggunakan ringkasan cuaca 24 jam sebelumnya. Solar_mean tidak dipakai sebagai fitur temporal karena merupakan ringkasan per lokasi, bukan data radiasi per jam.")


    c1, c2, c3 = st.columns(3)
    c1.metric("Data ML", f"{len(model_data):,}")
    c2.metric("Train", f"{len(train_data):,}")
    c3.metric("Test", f"{len(test_data):,}")

    st.subheader("Distribusi Kategori")
    class_dist = (
        model_data["quality_category"].value_counts()
        .rename_axis("Kategori").reset_index(name="Jumlah")
    )
    class_dist["Proporsi (%)"] = (
        class_dist["Jumlah"] / class_dist["Jumlah"].sum() * 100
    ).round(2)
    st.dataframe(class_dist, use_container_width=True)

    st.subheader("Distribusi kelas pada train dan test")
    train_dist = train_data["quality_category"].value_counts().rename("Train")
    test_dist = test_data["quality_category"].value_counts().rename("Test")
    split_dist = pd.concat([train_dist, test_dist], axis=1).fillna(0).astype(int)
    st.dataframe(split_dist, use_container_width=True)
    st.caption(
        f"Batas pemisahan waktu (UTC): {test_data['observed_at'].min()}. "
        "Pemisahan mengikuti timestamp unik agar pengamatan pada waktu yang sama "
        "tidak tersebar ke train dan test sekaligus."
    )

    st.subheader("Perbandingan Model")
    display_results = results_df.copy()
    metric_cols = [c for c in display_results.columns if c != "Model"]
    display_results[metric_cols] = display_results[metric_cols].mul(100).round(2)
    st.dataframe(display_results, use_container_width=True)

    best_model_name = results_df.iloc[0]["Model"]
    st.success(f"Model teratas berdasarkan F1 Macro: **{best_model_name}**")
    st.caption(
        "Peringkat ini mengikuti F1 Macro, lalu F1 Weighted sebagai pemecah seri. "
        "Nilai metrik yang sama tidak otomatis berarti model memiliki perilaku prediksi yang sama."
    )
    model_options = results_df["Model"].tolist()
    model_choice = st.selectbox(
        "Model untuk evaluasi", model_options,
        index=model_options.index(best_model_name)
    )

    y_pred = predictions[model_choice]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(test_data['quality_category'], y_pred) * 100:.2f}%")
    m2.metric("Precision", f"{precision_score(test_data['quality_category'], y_pred, average='weighted', zero_division=0) * 100:.2f}%")
    m3.metric("Recall", f"{recall_score(test_data['quality_category'], y_pred, average='weighted', zero_division=0) * 100:.2f}%")
    m4.metric("F1 Macro", f"{f1_score(test_data['quality_category'], y_pred, average='macro', zero_division=0) * 100:.2f}%")

    tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])

    with tab1:
        labels = ["Baik", "Sedang", "Tidak Sehat+"]
        cm = confusion_matrix(test_data["quality_category"], y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        ax.set_title(f"Confusion Matrix - {model_choice}")
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


    with tab2:
        report = classification_report(
            test_data["quality_category"],
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3), use_container_width=True)

    with tab3:
        if model_choice == "Random Forest":
            fi = feature_importance.copy()
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(fi["Feature"], fi["Importance"])
            ax.invert_yaxis()
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.set_title("Feature Importance - Random Forest")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:
            st.info("Feature importance pada halaman ini mengikuti Random Forest pada notebook. Pilih Random Forest untuk melihatnya.")


    # Audit kelas pada test untuk membatasi klaim evaluasi.
    test_class_counts = test_data["quality_category"].value_counts()
    absent_test_classes = [
        label for label in ["Baik", "Sedang", "Tidak Sehat+"]
        if label not in test_class_counts.index
    ]
    if absent_test_classes:
        st.warning(
            "Kelas berikut tidak muncul sebagai label aktual pada data test: "
            + ", ".join(absent_test_classes)
            + ". Kemampuan model pada kelas tersebut belum dapat dinilai secara memadai "
              "dari data test ini."
        )

    # Peringatan interpretasi model
    baseline_row = results_df[results_df["Model"] == "Baseline"].iloc[0]
    selected_row = results_df[results_df["Model"] == model_choice].iloc[0]
    if model_choice != "Baseline" and selected_row["Accuracy"] <= baseline_row["Accuracy"] + 1e-12:

        st.warning(
            "Model yang dipilih belum mengungguli baseline berdasarkan accuracy. Hindari menyimpulkan bahwa model sudah memiliki performa prediksi yang baik."
        )


# ============================================================
# DATA
# ============================================================

else:
    st.header("Data")
    st.write("Data yang digunakan oleh dashboard setelah proses cleaning.")


    dataset = st.selectbox("Pilih dataset", ["OpenAQ", "Open-Meteo", "NASA POWER"])
    if dataset == "OpenAQ":
        st.dataframe(openaq, use_container_width=True, height=500)
        st.download_button(
            "Download OpenAQ cleaned",
            openaq.to_csv(index=False).encode("utf-8"),
            "denpasar_openaq_clean.csv",
            "text/csv",
        )

    elif dataset == "Open-Meteo":
        st.dataframe(meteo, use_container_width=True, height=500)
        st.download_button(
            "Download Open-Meteo cleaned",
            meteo.to_csv(index=False).encode("utf-8"),
            "denpasar_openmeteo_clean.csv",
            "text/csv",
        )

    else:
        st.dataframe(nasa, use_container_width=True, height=500)
        st.download_button(
            "Download NASA POWER cleaned",
            nasa.to_csv(index=False).encode("utf-8"),
            "denpasar_nasa_power_clean.csv",
            "text/csv",
        )


    st.subheader("Jumlah Data")
    summary = pd.DataFrame({
        "Dataset": ["OpenAQ", "Open-Meteo", "NASA POWER"],
        "Jumlah Baris": [len(openaq), len(meteo), len(nasa)],
    })

    st.dataframe(summary, use_container_width=True)
