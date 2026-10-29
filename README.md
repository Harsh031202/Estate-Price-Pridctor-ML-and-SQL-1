<a id="readme-top"></a>

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,50:0f766e,100:14b8a6&height=200&section=header&text=Dragon%20Real%20Estates&fontSize=54&fontColor=ffffff&fontAlignY=38&desc=House%20Price%20Prediction%20with%20Machine%20Learning&descSize=20&descAlignY=62" alt="Dragon Real Estates banner" width="100%" />

### 🏠 Enter 13 property details. Get an estimated house price.

An end-to-end machine learning project that trains, compares, and ships a house price predictor, from raw data to a saved model you can run from the command line.

<br />

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Best Model](https://img.shields.io/badge/Best_Model-Random_Forest-14b8a6?style=for-the-badge)](#-model-results)
[![Stars](https://img.shields.io/github/stars/Harsh031202/Estate-Price-Pridctor-ML-and-SQL-1?style=for-the-badge&logo=github&color=eab308)](https://github.com/Harsh031202/Estate-Price-Pridctor-ML-and-SQL-1/stargazers)

[Overview](#-overview) · [Dataset](#-dataset) · [How It Works](#-how-it-works) · [Results](#-model-results) · [Quick Start](#-quick-start) · [Structure](#-project-structure) · [Roadmap](#-roadmap)

</div>

<br />

---

## ✨ Overview

**Dragon Real Estates** is a regression project that predicts the median value of a home from 13 neighborhood and property attributes. The notebook walks through the full workflow: exploring the data, preparing it, training several models, comparing them with cross-validation, and saving the final model together with its preprocessing pipeline.

Once saved, the model can be used by anyone, with no retraining needed. Run one script, answer 13 prompts, and get a price estimate.

> [!NOTE]
> Prices in this dataset are expressed in **thousands of US dollars**, so a prediction of `24` means roughly **$24,000**.

<br />

## 🚀 Highlights

<table>
<tr>
<td width="50%" valign="top">

### 📓 Full ML Workflow
Data exploration, preprocessing, model training, evaluation, and export, all documented step by step in a Jupyter notebook.

### 🤖 Three Models Compared
**Linear Regression**, **Decision Tree**, and **Random Forest** are evaluated side by side on the same data.

</td>
<td width="50%" valign="top">

### 🧪 Cross-Validated Scoring
Each model is scored with cross-validation, so the comparison is more reliable than a single train/test split.

### 📦 Ready-to-Use Model
The trained model and its preprocessing pipeline are saved together in `Dragon.joblib`, so predictions apply the exact same transformations as training.

### ⌨️ Instant Predictions
Run `main.py`, enter the features, and get a price right in your terminal.

</td>
</tr>
</table>

<br />

## 📊 Dataset

The project uses the classic **Boston Housing** dataset (`housing.data`, with its documentation in `housing.names`). A CSV version is included as `data.csv`. Every row describes one neighborhood using the 13 features below, and the target is the median home value (`MEDV`).

| Feature | Description |
|---------|-------------|
| `CRIM` | Per-capita crime rate by town |
| `ZN` | Share of residential land zoned for lots over 25,000 sq. ft. |
| `INDUS` | Share of non-retail business acres per town |
| `CHAS` | Borders the Charles River (`1`) or not (`0`) |
| `NOX` | Nitric oxide concentration |
| `RM` | Average number of rooms per dwelling |
| `AGE` | Share of owner-occupied units built before 1940 |
| `DIS` | Weighted distance to five employment centres |
| `RAD` | Index of accessibility to radial highways |
| `TAX` | Property-tax rate per $10,000 |
| `PTRATIO` | Pupil-teacher ratio by town |
| `B` | Demographic index (`1000(Bk - 0.63)²`) |
| `LSTAT` | Percentage of lower-status population |
| **`MEDV`** | **Target: median home value, in $1,000s** |

> [!IMPORTANT]
> This dataset is widely used for learning, but the `B` feature encodes a racial-demographics term, and scikit-learn removed its built-in loader over ethical concerns. It is fine for a learning project like this one. It is not appropriate for real-world pricing or lending decisions.

<br />

## 🔬 How It Works

```mermaid
flowchart LR
    A[(📁 housing.data<br/>data.csv)] --> B[🔍 Explore<br/>the data]
    B --> C[🧹 Preprocessing<br/>pipeline]
    C --> D{{🤖 Train and compare}}
    D --> E[Linear<br/>Regression]
    D --> F[Decision<br/>Tree]
    D --> G[Random<br/>Forest]
    E --> H[📏 Cross-validation]
    F --> H
    G --> H
    H --> I[🏆 Pick the best model]
    I --> J[(💾 Dragon.joblib<br/>model + pipeline)]
    J --> K[⌨️ main.py<br/>Model Usage.ipynb]
    K --> L([🏠 Predicted price])

    style A fill:#0f172a,stroke:#14b8a6,color:#fff
    style J fill:#0f766e,stroke:#14b8a6,color:#fff
    style I fill:#14b8a6,stroke:#0f766e,color:#fff
    style L fill:#0f172a,stroke:#14b8a6,color:#fff
```

<br />

## 🏆 Model Results

Three models were compared using cross-validation. **Lower is better.**

| Model | Mean score | Std. deviation | |
|-------|:----------:|:--------------:|:-:|
| Linear Regression | 4.22 | 0.75 | |
| Decision Tree | 4.19 | 0.85 | |
| **Random Forest** | **3.49** | **0.76** | 🏆 |

```mermaid
xychart-beta
    title "Cross-validation score by model (lower is better)"
    x-axis ["Linear Regression", "Decision Tree", "Random Forest"]
    y-axis "Score" 0 --> 5
    bar [4.22, 4.19, 3.49]
```

The **Random Forest** gives the lowest error by a clear margin. The raw numbers are in [`Outputs from different models`](Outputs%20from%20different%20models).

<br />

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Harsh031202/Estate-Price-Pridctor-ML-and-SQL-1.git
cd Estate-Price-Pridctor-ML-and-SQL-1
```

### 2. Install the dependencies

```bash
pip install numpy pandas scikit-learn joblib matplotlib jupyter
```

> [!TIP]
> `Dragon.joblib` was saved with scikit-learn, so load it with a compatible version. If you see a version warning or error, install the version the model was trained with.

### 3. Predict a price

```bash
python main.py
```

The script asks for the 13 features, in this order:

`CRIM` → `ZN` → `INDUS` → `CHAS` → `NOX` → `RM` → `AGE` → `DIS` → `RAD` → `TAX` → `PTRATIO` → `B` → `LSTAT`

It then applies the saved pipeline and prints the expected price.

<details>
<summary><b>📝 Example input (first record of the dataset)</b></summary>

<br />

| Feature | Value | | Feature | Value |
|---------|------:|-|---------|------:|
| `CRIM` | 0.00632 | | `DIS` | 4.09 |
| `ZN` | 18 | | `RAD` | 1 |
| `INDUS` | 2.31 | | `TAX` | 296 |
| `CHAS` | 0 | | `PTRATIO` | 15.3 |
| `NOX` | 0.538 | | `B` | 396.9 |
| `RM` | 6.575 | | `LSTAT` | 4.98 |
| `AGE` | 65.2 | | | |

The actual median value for this record in the dataset is **24.0** ($24,000).

</details>

### 4. Explore the notebooks

```bash
jupyter notebook
```

| Notebook | What it covers |
|----------|----------------|
| [`Dragon Real Estates.ipynb`](Dragon%20Real%20Estates.ipynb) | The full workflow: exploration, preprocessing, training, and model comparison |
| [`Model Usage.ipynb`](Model%20Usage.ipynb) | Loading the saved model and making predictions |

<br />

## 📁 Project Structure

```text
Estate-Price-Pridctor-ML-and-SQL-1/
├── Dragon Real Estates.ipynb    # End-to-end ML workflow
├── Model Usage.ipynb            # Using the saved model
├── main.py                      # Command-line price predictor
├── Dragon.joblib                # Saved model + preprocessing pipeline
├── data.csv                     # Dataset (CSV)
├── housing.data                 # Original dataset
├── housing.names                # Dataset documentation
└── Outputs from different models   # Cross-validation results
```

<br />

## 🛠️ Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-2E7D32?style=for-the-badge)

</div>

<br />

## 🗺️ Roadmap

- [x] Train and compare Linear Regression, Decision Tree, and Random Forest
- [x] Save the model and pipeline together with Joblib
- [x] Command-line predictor
- [ ] Add a SQL layer to store the dataset and prediction history
- [ ] Hyperparameter tuning for the Random Forest
- [ ] Simple web interface for predictions
- [ ] `requirements.txt` for one-command setup

<br />

## 🤝 Contributing

Suggestions and improvements are welcome. Found a bug or have an idea? [Open an issue](https://github.com/Harsh031202/Estate-Price-Pridctor-ML-and-SQL-1/issues), or fork the repo and send a pull request.

<br />

## 👤 Author

**DarK** · [@Harsh031202](https://github.com/Harsh031202)

If this project helped you learn something, consider giving it a ⭐.

<br />

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:14b8a6,50:0f766e,100:0f172a&height=100&section=footer" alt="footer" width="100%" />

<a href="#readme-top">↑ Back to top</a>

</div>
