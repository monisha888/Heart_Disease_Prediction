# ❤️ Heart Disease Prediction

🔗 **Live Demo:** [Add your deployed app link here](#)

## 📌 About

This project builds a complete supervised ML pipeline on the **UCI Heart Disease** dataset and exposes it through a simple, clean **Flask** web app where a user enters clinical values and gets an instant prediction.

**Pipeline highlights:**

- 🧹 Outlier handling via **Yeo-Johnson transformation** + IQR trimming
- 🎯 **Feature selection** — constant, quasi-constant, and hypothesis-testing based filtering
- ⚖️ Class balancing with **SMOTE**
- 📏 Feature scaling with **StandardScaler**
- 🤖 Training & comparison of **8 classification algorithms**
- 💾 Best model (**Gaussian Naive Bayes**) persisted for real-time inference
- 🌐 Deployed as a lightweight **Flask + gunicorn** web service

---

| Field | Description |
|---|---|
| `age` | Patient's age |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `thalach` | Max heart rate achieved |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `thal` | Thalassemia indicator |

---

## 🗂️ Project Structure

```
Heart-Disease-Prediction/
│
├── app.py                       # Flask app — loads model.pkl & scaled.pkl, serves predictions
├── main.py                      # Orchestrates the full training pipeline (HEART class)
├── variable_transformation.py   # Yeo-Johnson transform + IQR-based outlier trimming
├── feature_selection.py         # Constant / quasi-constant / hypothesis-test feature filtering
├── all_models.py                # Trains & evaluates 8 ML algorithms
├── log.py                       # Centralized logger factory
├── heart.csv                    # Training dataset
├── model.pkl                    # Serialized trained model (Gaussian Naive Bayes)
├── scaled.pkl                   # Serialized StandardScaler used at inference
├── requirements.txt             # Python dependencies
├── Procfile                     # Deployment process definition (gunicorn)
├── templates/
│   └── index.html               # Web UI (form + result)
└── logs/                        # Auto-generated log files (create empty folder if missing)
```

---

## 🧠 ML Pipeline

The `HEART` class in `main.py` runs everything end to end:

| Step | Method | What it does |
|---|---|---|
| 1 | `__init__` | Loads `heart.csv`, splits 80/20 train/test |
| 2 | `variable_transformation_outliers()` | Yeo-Johnson transform + IQR trimming on every feature |
| 3 | `best_col()` | Drops constant, quasi-constant & statistically insignificant columns |
| 4 | `data_balancing()` | Balances classes with **SMOTE**, scales with **StandardScaler** |
| 5 | `all_model()` | Trains & logs metrics for KNN, Naive Bayes, Logistic Regression, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost |
| 6 | `best_model()` | Fits final **Gaussian Naive Bayes**, exports `model.pkl` + `scaled.pkl` |

Run the training pipeline:

```bash
python main.py
```

> ⚠️ `log.py` currently points to a hardcoded Windows path. Change it to a relative path (e.g. `logs/{script_name}.log`) before running elsewhere or deploying.

---


## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/heart-disease-prediction.git
cd heart-disease-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python app.py
```

Then open **http://127.0.0.1:5000** and fill in the form to get a prediction.

---

## 🚀 Deployment

This repo ships with a **Procfile** (`web: gunicorn app:app`), so it's ready for any Procfile-based host.

### Render
1. Push to GitHub → [render.com](https://render.com) → **New → Web Service** → connect repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `gunicorn app:app`
4. Deploy, then paste your live URL into the [Live Demo](#-live-demo) section above ⬆️


## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn, XGBoost, imbalanced-learn |
| Web Framework | Flask |
| Server | gunicorn |
| Deployment | Render / Heroku / Docker (HF Spaces) |

---

## 🔮 Future Improvements

- [ ] Add hyperparameter tuning (GridSearchCV) for the final model
- [ ] Add an AUC-ROC comparison chart across all 8 models
- [ ] Containerize with Docker for consistent local + prod environments
- [ ] Add unit tests for the pipeline modules
- [ ] Replace hardcoded log paths with configurable, relative paths

---


**Built by [Vihara Tech](#)** — *Learn · Intern · Get Placed*

</div>
