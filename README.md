# 📊 Interactive EDA Classroom Dashboard

An interactive, single-file **Exploratory Data Analysis (EDA)** web application built with **Python, Flask, Pandas, NumPy, Matplotlib, and Seaborn**, designed to teach data visualization concepts using the [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) dataset from Kaggle.

The dashboard lets users explore a dataset visually — through histograms, bar charts, scatter plots, box plots, heatmaps, and pair plots — while also comparing how the same chart is built in raw **Matplotlib** vs. **Seaborn**.

---

## ✨ Features

- **Overview Panel** — dataset KPIs (student count, column count, average scores, missing values, duplicates), data types, and a live preview table.
- **Univariate Analysis** — histograms with mean/median reference lines, box plots with five-number summaries, count plots, and pie charts.
- **Bivariate Analysis** — bar charts (grouped averages), line charts (trend across categories), scatter plots with Pearson correlation and trendlines, and grouped box plots.
- **Multivariate Analysis** — correlation heatmaps and pair plots for exploring relationships across all numeric features at once.
- **Matplotlib vs. Seaborn Lab** — side-by-side, beginner-friendly code snippets showing how to build each chart type in both libraries.
- **Built-in Teaching Notes** — every chart ships with an explanation of its purpose, when to use it, what to observe, and its limitations.
- Charts are rendered server-side and streamed to the browser as base64-encoded images — no client-side charting library required.

## 🛠️ Tech Stack

| Layer            | Tools                                  |
|------------------|-----------------------------------------|
| Backend          | Python, Flask                          |
| Data Processing  | Pandas, NumPy                          |
| Visualization    | Matplotlib, Seaborn                    |
| Frontend         | Vanilla HTML, CSS, JavaScript          |
| Dataset          | Kaggle — Students Performance in Exams |

## 📂 Project Structure

```
.
├── eda_dashboard.py         # Single-file Flask app (backend + frontend + chart logic)
├── StudentsPerformance.csv  # Dataset (place in the same folder as the script)
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Install dependencies
pip install flask pandas numpy matplotlib seaborn
```

### Add the Dataset
Download `StudentsPerformance.csv` from [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) and place it in the same directory as `eda_dashboard.py`.

### Run the App

```bash
python eda_dashboard.py
```

Then open your browser to:

```
http://127.0.0.1:5000
```

## 🧩 API Endpoints

| Endpoint               | Description                                              |
|-------------------------|------------------------------------------------------------|
| `GET /`                 | Renders the dashboard UI                                  |
| `GET /api/summary`      | Returns dataset KPIs, column types, and a preview          |
| `GET /api/chart`        | Generates a chart (histogram, bar, line, box, scatter, heatmap, count, pie, pairplot) as a base64 image, plus its teaching notes and code samples |
| `GET /api/code_comparison` | Returns Matplotlib vs. Seaborn code for a given chart type |

## 🎓 Motivation

This project was built as a hands-on learning exercise combining several tools:
- **Kaggle** for sourcing a real-world dataset
- **Antigravity** as part of the development workflow
- **Command Prompt** for running and testing the Flask server locally

The goal was to go beyond static notebooks and build an interactive tool that not only visualizes data but also *teaches* the reasoning behind each chart choice — making it useful both as a personal EDA tool and as a classroom teaching aid.

## 📸 Screenshots

*(Add screenshots or a GIF of the dashboard here once deployed.)*

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙌 Acknowledgments

- Dataset: [Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams) by Royce Kimmons (via Kaggle)
- Built with Flask, Pandas, NumPy, Matplotlib, and Seaborn
-
