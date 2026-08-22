# Cat vs Dog Classification

A simple Streamlit web app that classifies an uploaded image as **cat** or **dog** using a locally trained Keras CNN model.

## Overview

The app:

1. Accepts a JPG, JPEG, or PNG image.
2. Converts it to the model's expected image format.
3. Resizes the image to `256 x 256` pixels.
4. Adds a batch dimension with shape `(1, 256, 256, 3)`.
5. Returns the predicted class and confidence.

The prediction runs locally. No external image API or API key is required.

## Project Files

- `app.py` - Streamlit application.
- `model.pkl` - Saved trained Keras model.
- `Cat_v_Dog_classification.ipynb` - Notebook used for model training and testing.
- `requirements.txt` - Python dependencies.
- `test_cat.jpg`, `dog1.jpg` - Reference images used by the interface.

## Run Locally

Use Python 3.12 and create or activate a virtual environment:

```powershell
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## GitHub Upload

From the project folder:

```powershell
git init
git add app.py model.pkl requirements.txt README.md .gitignore test_cat.jpg dog1.jpg Cat_v_Dog_classification.ipynb
git commit -m "Create cat and dog classification app"
git branch -M main
git remote add origin https://github.com/chintanthesiya/Cat-vs-Dog_Classification.git
git push -u origin main
```

Repository: https://github.com/chintanthesiya/Cat-vs-Dog_Classification

## Deploy on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository.
2. Select **Create app**.
3. Choose repository `chintanthesiya/Cat-vs-Dog_Classification`.
4. Select branch `main`.
5. Set the main file path to `app.py`.
6. Select **Deploy**.

Streamlit Cloud installs the packages from `requirements.txt` and loads `model.pkl` from the repository root.

## Notes

- The model file must remain beside `app.py`.
- Do not commit `.env` or private API keys.
- The included dataset is ignored by Git to keep the repository manageable.
