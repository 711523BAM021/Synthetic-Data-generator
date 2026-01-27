from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import os
import time


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
CHART_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

# ================= CLEANING =================
def clean_data(df):
    df = df.drop_duplicates()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    for col in df.select_dtypes(include='number'):
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna("Unknown")

    return df

# ================= DETECT TYPE =================
def is_timeseries(df):
    return any("date" in c.lower() or "time" in c.lower() for c in df.columns)

# ================= TABULAR MODEL =================
def generate_tabular(df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    for col in df.columns:
        if df[col].nunique() == len(df):
            metadata.update_column(col, sdtype='id')

    model = GaussianCopulaSynthesizer(metadata)
    model.fit(df)
    return model.sample(len(df))

# ================= TIME SERIES MODEL =================
def generate_timeseries(df):
    time_col = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()][0]
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.sort_values(time_col)

    num_df = df.select_dtypes(include='number')
    if num_df.shape[1] == 0:
        return pd.DataFrame()

    min_vals = num_df.min()
    max_vals = num_df.max()
    norm_data = (num_df - min_vals) / (max_vals - min_vals + 1e-6)

    SEQ_LEN = min(24, len(norm_data) - 1)
    if SEQ_LEN < 2:
        return pd.DataFrame()

    sequences = np.array([
        norm_data.iloc[i:i+SEQ_LEN].values
        for i in range(len(norm_data) - SEQ_LEN)
    ])

    num_features = num_df.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(SEQ_LEN, num_features)),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(num_features)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(sequences, sequences[:, -1, :], epochs=3, verbose=0)

    synthetic_num = model.predict(sequences)
    synthetic_num = pd.DataFrame(synthetic_num, columns=num_df.columns)
    synthetic_num = synthetic_num * (max_vals - min_vals) + min_vals

    synthetic_num = synthetic_num.reset_index(drop=True)
    df = df.reset_index(drop=True)

    synthetic_df = df.copy()
    synthetic_df[num_df.columns] = synthetic_num

    return synthetic_df

# ================= REALISM SCORE =================
def realism_score(real_df, synth_df):
    real_num = real_df.select_dtypes(include='number')
    synth_num = synth_df.select_dtypes(include='number')

    common_cols = real_num.columns.intersection(synth_num.columns)
    if len(common_cols) == 0:
        return 50.0

    diff = abs(real_num[common_cols].mean() - synth_num[common_cols].mean()) / (abs(real_num[common_cols].mean()) + 1e-6)
    realism = 100 - np.mean(diff.values) * 100
    return round(max(min(realism, 100), 0), 2)

# ================= CHART =================
def create_chart(real_df, synth_df):
    num_cols = real_df.select_dtypes(include='number').columns
    if len(num_cols) == 0:
        return None

    col = num_cols[0]
    plt.figure()
    real_df[col].hist(alpha=0.5, label="Real")
    synth_df[col].hist(alpha=0.5, label="Synthetic")
    plt.legend()

    chart_path = os.path.join(CHART_FOLDER, "chart.png")
    plt.savefig(chart_path)
    plt.close()
    return "chart.png"

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about_models.html")

@app.route('/generate', methods=['POST'])
def generate():
    start_time = time.time()

    file = request.files['file']
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    df = pd.read_csv(path)
    df = clean_data(df)
    df.to_csv(os.path.join(OUTPUT_FOLDER, "cleaned_real.csv"), index=False)

    rows, cols = df.shape
    numeric_cols = len(df.select_dtypes(include='number').columns)
    cat_cols = len(df.select_dtypes(include='object').columns)

    limitations = []
    if rows < 30:
        limitations.append("Dataset is small; realism may be reduced.")
    if numeric_cols == 0:
        limitations.append("No numeric data detected.")

    if is_timeseries(df):
        synthetic_df = generate_timeseries(df)
        model_type = "LSTM Time-Series Model"
        dataset_type = "Time-Series Dataset"
        reason = "Date/Time column detected"
    else:
        synthetic_df = generate_tabular(df)
        model_type = "Gaussian Copula Model"
        dataset_type = "Tabular Dataset"
        reason = "Structured tabular data detected"

    synthetic_df.to_csv(os.path.join(OUTPUT_FOLDER, "synthetic_data.csv"), index=False)

    realism = realism_score(df, synthetic_df)
    chart = create_chart(df, synthetic_df)
    synth_preview = synthetic_df.head().to_html(classes="data")

    processing_time = round(time.time() - start_time, 2)

    return render_template("result.html",
        realism=realism, chart=chart, synth_preview=synth_preview,
        model_type=model_type, dataset_type=dataset_type,
        reason=reason, numeric_cols=numeric_cols, cat_cols=cat_cols,
        limitations=limitations, time_taken=processing_time)

@app.route('/download_real')
def download_real():
    return send_file("outputs/cleaned_real.csv", as_attachment=True, download_name="cleaned_real_data.csv")

@app.route('/download_synth')
def download_synth():
    return send_file("outputs/synthetic_data.csv", as_attachment=True, download_name="synthetic_data.csv")

if __name__ == "__main__":
    app.run(debug=True)
