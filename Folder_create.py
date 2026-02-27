import os

base_path = r"E:\Personal\GitHub\Python Code Repo\nvidia-stock-forecasting-ml"

folders = [
    "data/raw",
    "data/processed",
    "src",
    "src/models",
    "scripts"
]

files = {
    "src/__init__.py": "",
    "src/config.py": "",
    "src/data_pipeline.py": "",
    "src/features.py": "",
    "src/targets.py": "",
    "src/train.py": "",
    "src/evaluate.py": "",
    "src/plots.py": "",
    "src/models/__init__.py": "",
    "src/models/linear_models.py": "",
    "src/models/tree_models.py": "",
    "src/models/lstm_model.py": "",
    "src/models/arima_models.py": "",
    "scripts/run_download_data.py": "",
    "scripts/run_feature_build.py": "",
    "scripts/run_train_all.py": "",
    "scripts/run_evaluate_all.py": "",
    "requirements.txt": "",
    "README.md": "# NVIDIA Stock Forecasting ML Project\n"
}

# Create folders
for folder in folders:
    path = os.path.join(base_path, folder)
    os.makedirs(path, exist_ok=True)

# Create files
for file_path, content in files.items():
    full_path = os.path.join(base_path, file_path)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content)

print("Repository structure created successfully!")