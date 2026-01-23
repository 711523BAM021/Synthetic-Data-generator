from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
os.makedirs("data", exist_ok=True)

def generate_dataset(spec):
    size_map = {"small":100,"medium":1000,"large":10000}
    rows = size_map[spec["dataset_size"]]
    data = {}

    for c in spec["columns"]:
        if c["type"] == "numerical":
            data[c["name"]] = np.random.randint(10,100,rows)
        else:
            data[c["name"]] = np.random.choice(["A","B","C"],rows)

    if spec["dataset_type"] != "Clustering" and spec["target"]:
        if spec["dataset_type"] == "Classification":
            data[spec["target"]] = np.random.choice([0,1],rows)
        else:
            data[spec["target"]] = np.random.randint(1000,5000,rows)

    df = pd.DataFrame(data)
    df.fillna(method="ffill", inplace=True)
    return df

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    spec = request.json
    df = generate_dataset(spec)
    df.to_csv("data/generated_dataset.csv", index=False)

    return jsonify({
        "rows": len(df),
        "columns": len(df.columns),
        "dataset_type": spec["dataset_type"],
        "preview": df.head(10).to_dict(orient="records")
    })

@app.route("/download")
def download():
    name = request.args.get("filename","synthetic_dataset")
    fmt = request.args.get("format","csv")
    base = "data/generated_dataset"

    if fmt=="csv":
        return send_file(f"{base}.csv", as_attachment=True, download_name=f"{name}.csv")
    if fmt=="excel":
        df = pd.read_csv(f"{base}.csv")
        df.to_excel(f"{base}.xlsx", index=False)
        return send_file(f"{base}.xlsx", as_attachment=True, download_name=f"{name}.xlsx")
    if fmt=="json":
        df = pd.read_csv(f"{base}.csv")
        df.to_json(f"{base}.json", orient="records")
        return send_file(f"{base}.json", as_attachment=True, download_name=f"{name}.json")

@app.route("/download-metadata")
def metadata():
    name = request.args.get("filename","synthetic_dataset")
    df = pd.read_csv("data/generated_dataset.csv")
    meta = f"""Dataset Name: {name}
Rows: {len(df)}
Columns: {len(df.columns)}
ML Ready: Yes
Generated On: {datetime.now()}"""
    path="data/metadata.txt"
    open(path,"w").write(meta)
    return send_file(path, as_attachment=True, download_name=f"{name}_metadata.txt")

if __name__ == "__main__":
    app.run(debug=True)
