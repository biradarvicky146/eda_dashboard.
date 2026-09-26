import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, jsonify, request, render_template_string

# Configurable path for dataset
CSV_PATH = os.environ.get("EDA_CSV_PATH", "StudentsPerformance.csv")

# Set consistent seaborn aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8


def load_dataset(filepath=None):
    """Loads the CSV dataset from disk using Pandas.
    Returns cleaned DataFrame or None if not found.
    """
    if filepath is None:
        filepath = CSV_PATH
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        return clean_dataset(df)
    except Exception as e:
        print(f"Error reading dataset from {filepath}: {e}")
        return None


def clean_dataset(df):
    """Cleans dataset: strips whitespace from column headers and string values,
    and coerces score columns to numeric values.
    """
    if df is None or df.empty:
        return df
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    for col in cleaned.select_dtypes(include='object').columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
    for col in ['math score', 'reading score', 'writing score']:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')
    return cleaned


def get_dataset_summary(df):
    """Computes summary statistics, null counts, duplicate counts, and preview rows."""
    if df is None or df.empty:
        return {"error": "Dataset is not available or could not be loaded."}

    num_cols = get_numeric_columns(df)
    cat_cols = get_categorical_columns(df)

    math_avg = round(float(df['math score'].mean()), 2) if 'math score' in df.columns else None
    reading_avg = round(float(df['reading score'].mean()), 2) if 'reading score' in df.columns else None
    writing_avg = round(float(df['writing score'].mean()), 2) if 'writing score' in df.columns else None

    missing_counts = {col: int(df[col].isnull().sum()) for col in df.columns}
    total_missing = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())

    preview_records = df.head(5).to_dict(orient='records')
    describe_dict = df[num_cols].describe().round(2).to_dict() if num_cols else {}

    return {
        "total_students": int(len(df)),
        "total_columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "avg_math_score": math_avg,
        "avg_reading_score": reading_avg,
        "avg_writing_score": writing_avg,
        "total_missing": total_missing,
        "missing_by_column": missing_counts,
        "duplicate_count": duplicate_count,
        "preview": preview_records,
        "numeric_summary": describe_dict
    }


def get_numeric_columns(df):
    """Returns a list of numeric column names from the DataFrame."""
    if df is None or df.empty:
        return []
    return list(df.select_dtypes(include=[np.number]).columns)


def get_categorical_columns(df):
    """Returns a list of categorical / object column names from the DataFrame."""
    if df is None or df.empty:
        return []
    return list(df.select_dtypes(include=['object', 'category']).columns)


def fig_to_base64(fig):
    """Encodes a Matplotlib figure into a base64 PNG data string in-memory."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def create_histogram(df, column, library='seaborn'):
    """Generates a histogram with frequency distribution.
    Supports both Seaborn and Matplotlib.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in dataset.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Histogram requires a numeric column. '{column}' is categorical.")

    fig, ax = plt.subplots(figsize=(8, 5))
    clean_series = df[column].dropna()

    if library == 'seaborn':
        sns.histplot(data=df, x=column, kde=True, bins=20, color='#2563eb', edgecolor='white', ax=ax)
        ax.set_title(f"Distribution of {column.title()} with KDE (Seaborn)", fontsize=13, fontweight='bold', pad=12)
    else:
        ax.hist(clean_series, bins=20, color='#2563eb', edgecolor='black', alpha=0.75, rwidth=0.9)
        ax.set_title(f"Distribution of {column.title()} (Matplotlib)", fontsize=13, fontweight='bold', pad=12)

    ax.set_xlabel(column.title(), fontsize=11, fontweight='semibold')
    ax.set_ylabel("Frequency (Student Count)", fontsize=11, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig_to_base64(fig)


def create_bar_chart(df, x_column, y_column=None, library='seaborn'):
    """Generates a bar chart. If y_column is provided, plots mean of y by x.
    Otherwise plots frequency counts of x.
    """
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' does not exist.")

    fig, ax = plt.subplots(figsize=(8, 5))

    if y_column and y_column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[y_column]):
            raise ValueError(f"Y-axis column '{y_column}' must be numeric.")

        if library == 'seaborn':
            sns.barplot(data=df, x=x_column, y=y_column, errorbar=None, palette="Blues_d", hue=x_column, legend=False, ax=ax)
            ax.set_title(f"Average {y_column.title()} by {x_column.title()} (Seaborn)", fontsize=13, fontweight='bold', pad=12)
        else:
            grouped = df.groupby(x_column)[y_column].mean().reset_index()
            bars = ax.bar(grouped[x_column].astype(str), grouped[y_column], color='#3b82f6', edgecolor='#1d4ed8')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}", ha='center', va='bottom', fontsize=9)
            ax.set_title(f"Average {y_column.title()} by {x_column.title()} (Matplotlib)", fontsize=13, fontweight='bold', pad=12)

        ax.set_ylabel(f"Mean {y_column.title()}", fontsize=11, fontweight='semibold')
    else:
        counts = df[x_column].value_counts()
        if library == 'seaborn':
            sns.countplot(data=df, x=x_column, order=counts.index, palette="mako", hue=x_column, legend=False, ax=ax)
            ax.set_title(f"Student Counts by {x_column.title()} (Seaborn)", fontsize=13, fontweight='bold', pad=12)
        else:
            bars = ax.bar(counts.index.astype(str), counts.values, color='#059669', edgecolor='#065f46')
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=9)
            ax.set_title(f"Student Counts by {x_column.title()} (Matplotlib)", fontsize=13, fontweight='bold', pad=12)

        ax.set_ylabel("Student Count", fontsize=11, fontweight='semibold')

    ax.set_xlabel(x_column.title(), fontsize=11, fontweight='semibold')
    if len(df[x_column].dropna().unique()) > 3:
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    fig.tight_layout()
    return fig_to_base64(fig)


def create_line_chart(df, x_column, y_column, library='seaborn'):
    """Generates a line chart for ordered or grouped comparison."""
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError("X and Y columns must exist in dataset.")
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        raise ValueError(f"Y-axis column '{y_column}' must be numeric.")

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = df.groupby(x_column)[y_column].mean().reset_index()

    if library == 'seaborn':
        sns.lineplot(data=grouped, x=x_column, y=y_column, marker='o', markersize=8, color='#0284c7', linewidth=2.5, ax=ax)
        ax.set_title(f"Trend: Mean {y_column.title()} across {x_column.title()} (Seaborn)", fontsize=13, fontweight='bold', pad=12)
    else:
        ax.plot(grouped[x_column].astype(str), grouped[y_column], marker='o', markersize=8, color='#0284c7', linewidth=2.5)
        for i, row in grouped.iterrows():
            ax.text(i, row[y_column] + 0.8, f"{row[y_column]:.1f}", ha='center', fontsize=9)
        ax.set_title(f"Trend: Mean {y_column.title()} across {x_column.title()} (Matplotlib)", fontsize=13, fontweight='bold', pad=12)

    ax.set_xlabel(x_column.title(), fontsize=11, fontweight='semibold')
    ax.set_ylabel(f"Mean {y_column.title()}", fontsize=11, fontweight='semibold')
    if len(grouped) > 3:
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    return fig_to_base64(fig)


def create_box_plot(df, y_column, x_column=None, library='seaborn'):
    """Generates a box plot to display 5-number summary (Min, Q1, Median, Q3, Max, Outliers).
    Can be univariate (y only) or bivariate (y grouped by category x).
    """
    if y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found.")
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        raise ValueError(f"Box plot requires numeric Y variable '{y_column}'.")

    fig, ax = plt.subplots(figsize=(8, 5))

    if x_column and x_column in df.columns:
        if library == 'seaborn':
            sns.boxplot(data=df, x=x_column, y=y_column, palette="Set3", hue=x_column, legend=False, ax=ax)
            ax.set_title(f"{y_column.title()} Distribution by {x_column.title()} (Seaborn)", fontsize=13, fontweight='bold', pad=12)
        else:
            categories = [str(c) for c in df[x_column].dropna().unique()]
            data_groups = [df[df[x_column] == cat][y_column].dropna().values for cat in categories]
            bp = ax.boxplot(data_groups, tick_labels=categories, patch_artist=True, medianprops=dict(color='red', linewidth=1.5))
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            ax.set_title(f"{y_column.title()} Distribution by {x_column.title()} (Matplotlib)", fontsize=13, fontweight='bold', pad=12)

        ax.set_xlabel(x_column.title(), fontsize=11, fontweight='semibold')
        if len(df[x_column].dropna().unique()) > 3:
            plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    else:
        if library == 'seaborn':
            sns.boxplot(data=df, y=y_column, color='#93c5fd', ax=ax)
            ax.set_title(f"Univariate Box Plot of {y_column.title()} (Seaborn)", fontsize=13, fontweight='bold', pad=12)
        else:
            bp = ax.boxplot(df[y_column].dropna().values, patch_artist=True, medianprops=dict(color='red', linewidth=1.5))
            bp['boxes'][0].set_facecolor('#93c5fd')
            ax.set_xticks([1])
            ax.set_xticklabels([y_column.title()])
            ax.set_title(f"Univariate Box Plot of {y_column.title()} (Matplotlib)", fontsize=13, fontweight='bold', pad=12)

    ax.set_ylabel(y_column.title(), fontsize=11, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    fig.tight_layout()
    return fig_to_base64(fig)


def create_scatter_plot(df, x_column, y_column, hue_column=None, library='seaborn'):
    """Generates a scatter plot to examine bivariate numerical correlation."""
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError("X and Y columns must exist in dataset.")
    if not pd.api.types.is_numeric_dtype(df[x_column]) or not pd.api.types.is_numeric_dtype(df[y_column]):
        raise ValueError("Scatter plot requires both X and Y columns to be numeric.")

    fig, ax = plt.subplots(figsize=(8, 5))
    has_hue = hue_column and hue_column in df.columns

    if library == 'seaborn':
        sns.scatterplot(data=df, x=x_column, y=y_column, hue=hue_column if has_hue else None,
                        palette="tab10" if has_hue else None, alpha=0.75, s=50, ax=ax)
        ax.set_title(f"{y_column.title()} vs {x_column.title()} (Seaborn)", fontsize=13, fontweight='bold', pad=12)
    else:
        if has_hue:
            categories = df[hue_column].dropna().unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
            for cat, color in zip(categories, colors):
                subset = df[df[hue_column] == cat]
                ax.scatter(subset[x_column], subset[y_column], label=str(cat), alpha=0.75, s=45, color=color)
            ax.legend(title=hue_column.title(), frameon=True)
        else:
            ax.scatter(df[x_column], df[y_column], color='#2563eb', alpha=0.7, s=45, edgecolors='none')
        ax.set_title(f"{y_column.title()} vs {x_column.title()} (Matplotlib)", fontsize=13, fontweight='bold', pad=12)

    ax.set_xlabel(x_column.title(), fontsize=11, fontweight='semibold')
    ax.set_ylabel(y_column.title(), fontsize=11, fontweight='semibold')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    return fig_to_base64(fig)


def create_heatmap(df, library='seaborn'):
    """Generates a Pearson correlation heatmap across numeric columns."""
    num_cols = get_numeric_columns(df)
    if len(num_cols) < 2:
        raise ValueError("Heatmap requires at least 2 numeric columns.")

    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))

    if library == 'seaborn':
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1,
                    square=True, linewidths=1.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title("Correlation Heatmap (Seaborn)", fontsize=13, fontweight='bold', pad=12)
    else:
        im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Pearson Correlation", rotation=-90, va="bottom")
        ax.set_xticks(range(len(num_cols)))
        ax.set_yticks(range(len(num_cols)))
        ax.set_xticklabels([c.title() for c in num_cols], rotation=25, ha='right')
        ax.set_yticklabels([c.title() for c in num_cols])
        for i in range(len(num_cols)):
            for j in range(len(num_cols)):
                val = corr.iloc[i, j]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight='bold', fontsize=10)
        ax.set_title("Correlation Heatmap (Matplotlib)", fontsize=13, fontweight='bold', pad=12)

    fig.tight_layout()
    return fig_to_base64(fig)


def create_count_plot(df, column, library='seaborn'):
    """Generates a count plot for a categorical column."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    counts = df[column].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))

    if library == 'seaborn':
        sns.countplot(data=df, x=column, order=counts.index, palette="viridis", hue=column, legend=False, ax=ax)
        ax.set_title(f"Frequency of {column.title()} (Seaborn)", fontsize=13, fontweight='bold', pad=12)
    else:
        bars = ax.bar(counts.index.astype(str), counts.values, color='#4f46e5', edgecolor='#312e81')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5, f"{int(bar.get_height())}", ha='center', va='bottom', fontsize=9)
        ax.set_title(f"Frequency of {column.title()} (Matplotlib)", fontsize=13, fontweight='bold', pad=12)

    ax.set_xlabel(column.title(), fontsize=11, fontweight='semibold')
    ax.set_ylabel("Student Count", fontsize=11, fontweight='semibold')
    if len(counts) > 3:
        plt.setp(ax.get_xticklabels(), rotation=25, ha='right')
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    fig.tight_layout()
    return fig_to_base64(fig)


def create_pie_chart(df, column):
    """Generates a pie chart displaying proportions of categorical groups."""
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    counts = df[column].value_counts()
    if len(counts) > 10:
        raise ValueError("Pie charts are only suitable for categories with 10 or fewer distinct groups.")

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(counts)))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=[str(l).title() for l in counts.index],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        wedgeprops=dict(edgecolor='white', linewidth=2),
        textprops=dict(fontsize=10)
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    ax.set_title(f"Proportion Breakdown: {column.title()}", fontsize=13, fontweight='bold', pad=14)
    ax.axis('equal')
    fig.tight_layout()
    return fig_to_base64(fig)


def create_pairplot(df, columns=None, hue_column=None):
    """Generates a pair plot matrix across numeric columns using Seaborn."""
    if columns is None:
        columns = get_numeric_columns(df)
    if len(columns) < 2:
        raise ValueError("Pair plot requires at least 2 numeric columns.")

    has_hue = hue_column and hue_column in df.columns
    g = sns.pairplot(
        df,
        vars=columns,
        hue=hue_column if has_hue else None,
        palette="tab10" if has_hue else None,
        diag_kind="kde",
        corner=False,
        height=2.2,
        aspect=1.1
    )
    g.fig.subplots_adjust(top=0.92)
    hue_suffix = f" (Grouped by {hue_column.title()})" if has_hue else ""
    g.fig.suptitle(f"Pairwise Feature Distributions{hue_suffix}", fontsize=13, fontweight='bold')
    b64 = fig_to_base64(g.fig)
    return b64


def get_chart_explanation(chart_type):
    """Returns educational explanations, usage criteria, and interpretation guidelines for each chart."""
    explanations = {
        "histogram": {
            "title": "Histogram",
            "what_is_it": "A graph that groups continuous numerical values into consecutive intervals (bins) and shows the count of observations falling into each bin.",
            "purpose": "Uncover the distribution shape, central tendency, variance, skewness, and detect potential outliers or multi-modality.",
            "when_to_use": "Whenever you are examining a single quantitative continuous variable (Univariate numerical analysis).",
            "what_to_observe": "Check whether the distribution is bell-shaped (normal), right-skewed (tail to the right), left-skewed, or multimodal. Look for peaks and isolated extreme bars.",
            "example_question": "Do student exam scores follow a normal bell curve, or are scores skewed towards passing grades?",
            "data_type": "Continuous / Quantitative Numerical",
            "limitations": "Bin count selection alters appearance; does not display individual values or category breakdowns."
        },
        "bar_chart": {
            "title": "Bar Chart",
            "what_is_it": "A plot using rectangular bars where the length or height represents an aggregated statistical summary (mean, sum) for distinct categories.",
            "purpose": "Compare quantitative averages or totals across distinct categorical groups.",
            "when_to_use": "Bivariate analysis comparing a numeric value across categorical classes (e.g., average math score by parental education).",
            "what_to_observe": "Compare bar heights to spot the highest-performing and lowest-performing cohorts and evaluate the magnitude of differences between categories.",
            "example_question": "Do students receiving standard lunch achieve significantly higher writing scores compared to those with free/reduced lunch?",
            "data_type": "Categorical (X) + Numerical Summary Metric (Y)",
            "limitations": "Hides underlying variance, data spread, and outliers within each group (unlike a box plot)."
        },
        "line_chart": {
            "title": "Line Chart",
            "what_is_it": "A plot connecting successive data points with line segments to display a continuous progression or ordered trend.",
            "purpose": "Identify trends, trajectories, inflection points, and ordinal patterns across ordered levels.",
            "when_to_use": "Comparing mean scores across naturally ordered levels (like education progression) or sequential variables.",
            "what_to_observe": "Look at slope direction (positive/negative), slope steepness, plateaus, and whether progression is linear or diminishing.",
            "example_question": "Does student test performance steadily climb as parental level of education advances from high school to master's degree?",
            "data_type": "Ordered Categorical / Ordinal or Continuous Sequence",
            "limitations": "Should not be used for unordered nominal categories where line connections imply a false continuous transition."
        },
        "box_plot": {
            "title": "Box Plot (Box-and-Whisker)",
            "what_is_it": "A standardized visual summary of the five-number statistical summary: Minimum, First Quartile (Q1), Median (Q2), Third Quartile (Q3), and Maximum, with outliers shown as individual dots.",
            "purpose": "Visualize spread, central tendency, interquartile range (IQR), skewness, and isolate statistical anomalies.",
            "when_to_use": "Comparing numerical distributions across multiple categories side-by-side or screening for outliers.",
            "what_to_observe": "Look at median position inside the box, the height of the box (IQR spread), whisker span, and distinct dots beyond whiskers representing extreme outliers.",
            "example_question": "Are there severe low-score outliers in math, and does score variability differ between test preparation completers vs non-completers?",
            "data_type": "Continuous Numerical (Y) with optional Categorical Grouping (X)",
            "limitations": "Can obscure bimodal or multimodal distribution shapes (a violin plot or KDE curve is better for detecting multiple peaks)."
        },
        "scatter_plot": {
            "title": "Scatter Plot",
            "what_is_it": "A Cartesian coordinate chart where individual data records are plotted as points at coordinates (X, Y).",
            "purpose": "Evaluate the strength, direction, linearity, and clustering of relationships between two quantitative variables.",
            "when_to_use": "Bivariate numerical analysis to see if changes in one continuous variable relate to changes in another.",
            "what_to_observe": "Check for linear alignment (positive/negative slope), spread of points (strong vs weak correlation), non-linear curves, and isolated outlier points.",
            "example_question": "How strongly does a student's reading comprehension score correlate with their writing composition exam score?",
            "data_type": "Two Continuous Numerical Variables (optional Categorical hue)",
            "limitations": "Points can overlap heavily (overplotting) when dealing with large datasets; does not prove causal relationships."
        },
        "heatmap": {
            "title": "Correlation Heatmap",
            "what_is_it": "A color-coded 2D matrix displaying Pearson correlation coefficients (-1.00 to +1.00) between every pair of numeric variables.",
            "purpose": "Quickly identify multicollinearity, strong positive/negative associations, and independent variables in a single view.",
            "when_to_use": "Multivariate screening across all quantitative features during initial EDA.",
            "what_to_observe": "Warm/dark colors near +1.0 indicate strong positive associations; values near -1.0 indicate inverse relationships; values near 0 indicate lack of linear association.",
            "example_question": "Which pair of subjects shares the highest correlation: reading and writing, or math and reading?",
            "data_type": "Matrix of Continuous Numerical Variables",
            "limitations": "Captures only linear relationships; non-linear associations may appear as zero; correlation does not imply causation!"
        },
        "count_plot": {
            "title": "Count Plot",
            "what_is_it": "A specialized categorical frequency bar chart showing the absolute number of observations in each discrete class.",
            "purpose": "Assess sample balance, detect class imbalances, and determine the distribution of categories in the dataset.",
            "when_to_use": "Univariate categorical exploration (e.g., verifying student gender balance or parental education distribution).",
            "what_to_observe": "Compare bar counts to detect heavily dominated or sparse categories that might bias downstream analyses.",
            "example_question": "How many students in the dataset completed the test preparation course versus those who took none?",
            "data_type": "Discrete Nominal or Ordinal Categorical",
            "limitations": "Shows only sample counts; does not reveal score metrics or internal category distributions."
        },
        "pie_chart": {
            "title": "Pie Chart",
            "what_is_it": "A circular statistical graphic divided into slices illustrating proportional percentages of a whole (100%).",
            "purpose": "Show part-to-whole relative compositions for categorical variables with very few categories.",
            "when_to_use": "Presenting simple high-level composition ratios (2 to 5 categories maximum).",
            "what_to_observe": "Look for the dominant slice and the relative percentage share of each category.",
            "example_question": "What percentage of the student cohort receives subsidized free/reduced lunch?",
            "data_type": "Categorical with few distinct classes (summing to 100%)",
            "limitations": "Human perception struggles to accurately judge angles and area sizes; completely ineffective for >5 categories."
        },
        "pairplot": {
            "title": "Pair Plot (Scatter Matrix)",
            "what_is_it": "A multi-panel grid displaying pairwise scatter plots for all combinations of numerical features, with univariate distributions along the diagonal.",
            "purpose": "Conduct comprehensive multivariate screening to uncover relationships, distributions, and class separations simultaneously.",
            "when_to_use": "Early-stage exploratory analysis to survey all numeric variables at once, optionally stratified by a category.",
            "what_to_observe": "Look at diagonal curves for univariate shapes; look at off-diagonal scatter plots for correlations and clustering between groups.",
            "example_question": "Do male and female students form separate clusters when examining math, reading, and writing scores together?",
            "data_type": "Multiple Continuous Numerical Features + Optional Categorical Hue",
            "limitations": "Computationally intensive for large datasets; visual clarity deteriorates when column count exceeds 5."
        }
    }
    return explanations.get(chart_type, {
        "title": chart_type.replace('_', ' ').title(),
        "what_is_it": "A statistical data visualization.",
        "purpose": "Explore patterns in data.",
        "when_to_use": "EDA workflows.",
        "what_to_observe": "Shapes, trends, and anomalies.",
        "example_question": "What pattern exists in the data?",
        "data_type": "Tabular Data",
        "limitations": "Context dependent."
    })


def generate_matplotlib_code(chart_type, x_column=None, y_column=None, hue_column=None):
    """Generates beginner-friendly executable Python code using pure Matplotlib."""
    x = x_column or "math score"
    y = y_column or "reading score"
    hue = hue_column or "gender"

    if chart_type == "histogram":
        return f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Create figure
plt.figure(figsize=(8, 5))

# Plot histogram with 20 bins
plt.hist(df['{x}'].dropna(), bins=20, color='#2563eb', edgecolor='black', alpha=0.75, rwidth=0.9)

# Add titles and axis labels
plt.title("Distribution of {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("Frequency (Student Count)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# Render plot
plt.tight_layout()
plt.show()"""

    elif chart_type == "bar_chart":
        return f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Compute group means using pandas
grouped = df.groupby('{x}')['{y}'].mean().reset_index()

# Create bar plot
plt.figure(figsize=(8, 5))
bars = plt.bar(grouped['{x}'].astype(str), grouped['{y}'], color='#3b82f6', edgecolor='#1d4ed8')

# Annotate values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f"{{height:.1f}}", ha='center', va='bottom', fontsize=9)

plt.title("Average {y.title()} by {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("Mean {y.title()}", fontsize=11)
plt.xticks(rotation=25, ha='right')
plt.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.tight_layout()
plt.show()"""

    elif chart_type == "line_chart":
        return f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Calculate group means
grouped = df.groupby('{x}')['{y}'].mean().reset_index()

plt.figure(figsize=(8, 5))
plt.plot(grouped['{x}'].astype(str), grouped['{y}'], marker='o', markersize=8, color='#0284c7', linewidth=2.5)

# Annotate data points
for i, row in grouped.iterrows():
    plt.text(i, row['{y}'] + 0.8, f"{{row['{y}']:.1f}}", ha='center', fontsize=9)

plt.title("Mean {y.title()} across {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("Mean {y.title()}", fontsize=11)
plt.xticks(rotation=25, ha='right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()"""

    elif chart_type == "box_plot":
        return f"""import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Separate groups
categories = df['{x}'].dropna().unique()
data_groups = [df[df['{x}'] == cat]['{y}'].dropna().values for cat in categories]

plt.figure(figsize=(8, 5))
bp = plt.boxplot(data_groups, tick_labels=categories, patch_artist=True, medianprops=dict(color='red', linewidth=1.5))

# Color individual boxes
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.title("{y.title()} Distribution by {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("{y.title()}", fontsize=11)
plt.xticks(rotation=25, ha='right')
plt.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.tight_layout()
plt.show()"""

    elif chart_type == "scatter_plot":
        return f"""import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

plt.figure(figsize=(8, 5))

# Plot category subsets with distinct colors
categories = df['{hue}'].dropna().unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

for cat, color in zip(categories, colors):
    subset = df[df['{hue}'] == cat]
    plt.scatter(subset['{x}'], subset['{y}'], label=str(cat), alpha=0.75, s=45, color=color)

plt.title("{y.title()} vs {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("{y.title()}", fontsize=11)
plt.legend(title='{hue.title()}', frameon=True)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()"""

    elif chart_type == "heatmap":
        return """import matplotlib.pyplot as plt
import pandas as pd

# Load dataset and compute correlation matrix
df = pd.read_csv("StudentsPerformance.csv")
num_cols = df.select_dtypes(include='number').columns
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Set axis tick labels
ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=25, ha='right')
ax.set_yticklabels(num_cols)

# Annotate correlation numbers
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        val = corr.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.6 else "black", fontweight='bold')

plt.title("Correlation Heatmap", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""

    elif chart_type == "count_plot":
        return f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")
counts = df['{x}'].value_counts()

plt.figure(figsize=(8, 5))
bars = plt.bar(counts.index.astype(str), counts.values, color='#4f46e5', edgecolor='#312e81')

for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, f"{{int(bar.get_height())}}", ha='center', va='bottom', fontsize=9)

plt.title("Frequency Count of {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("Student Count", fontsize=11)
plt.xticks(rotation=25, ha='right')
plt.grid(True, linestyle='--', alpha=0.4, axis='y')
plt.tight_layout()
plt.show()"""

    elif chart_type == "pie_chart":
        return f"""import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")
counts = df['{x}'].value_counts()

plt.figure(figsize=(7, 6))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
plt.title("Proportion Breakdown of {x.title()}", fontsize=14, fontweight='bold')
plt.axis('equal')
plt.tight_layout()
plt.show()"""

    elif chart_type == "pairplot":
        return """import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")
numeric_cols = ['math score', 'reading score', 'writing score']

# Pairwise scatter matrix in Pandas / Matplotlib
axes = scatter_matrix(df[numeric_cols], alpha=0.7, figsize=(8, 8), diagonal='kde', color='#2563eb')
plt.suptitle("Pairwise Scatter Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()"""

    return "# Code not available for this chart type."


def generate_seaborn_code(chart_type, x_column=None, y_column=None, hue_column=None):
    """Generates beginner-friendly executable Python code using Seaborn."""
    x = x_column or "math score"
    y = y_column or "reading score"
    hue = hue_column or "gender"

    if chart_type == "histogram":
        return f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

plt.figure(figsize=(8, 5))
# One-line histogram with Kernel Density Estimation (KDE)
sns.histplot(data=df, x='{x}', kde=True, bins=20, color='#2563eb')

plt.title("Distribution of {x.title()} with KDE", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("Frequency", fontsize=11)
plt.tight_layout()
plt.show()"""

    elif chart_type == "bar_chart":
        return f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

plt.figure(figsize=(8, 5))
# Seaborn automatically computes the mean of '{y}' for each category in '{x}'
sns.barplot(data=df, x='{x}', y='{y}', errorbar=None, palette="Blues_d", hue='{x}', legend=False)

plt.title("Average {y.title()} by {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("Mean {y.title()}", fontsize=11)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif chart_type == "line_chart":
        return f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")
grouped = df.groupby('{x}')['{y}'].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.lineplot(data=grouped, x='{x}', y='{y}', marker='o', markersize=8, color='#0284c7', linewidth=2.5)

plt.title("Mean {y.title()} across {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("Mean {y.title()}", fontsize=11)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif chart_type == "box_plot":
        return f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

plt.figure(figsize=(8, 5))
# Generates side-by-side boxplots with automatic 5-number summary and outliers
sns.boxplot(data=df, x='{x}', y='{y}', palette="Set3", hue='{x}', legend=False)

plt.title("{y.title()} Distribution by {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("{y.title()}", fontsize=11)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif chart_type == "scatter_plot":
        return f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

plt.figure(figsize=(8, 5))
# Scatter plot with automated legend and category palette mapping
sns.scatterplot(data=df, x='{x}', y='{y}', hue='{hue}', palette="tab10", alpha=0.75, s=50)

plt.title("{y.title()} vs {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("{y.title()}", fontsize=11)
plt.tight_layout()
plt.show()"""

    elif chart_type == "heatmap":
        return """import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset and compute correlation
df = pd.read_csv("StudentsPerformance.csv")
num_cols = df.select_dtypes(include='number').columns
corr = df[num_cols].corr()

plt.figure(figsize=(7, 5.5))
# Elegant heatmap with automated colorbar and matrix text annotations
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, square=True, linewidths=1.5)

plt.title("Correlation Heatmap", fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.show()"""

    elif chart_type == "count_plot":
        return f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

plt.figure(figsize=(8, 5))
# Automatically computes frequency counts for discrete categories
sns.countplot(data=df, x='{x}', order=df['{x}'].value_counts().index, palette="viridis", hue='{x}', legend=False)

plt.title("Frequency of {x.title()}", fontsize=14, fontweight='bold')
plt.xlabel("{x.title()}", fontsize=11)
plt.ylabel("Student Count", fontsize=11)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.show()"""

    elif chart_type == "pie_chart":
        return f"""# Note: Seaborn intentionally does not include a pie chart function
# because statistical best practice favors bar/count charts over pie slices.
# For proportions, use Matplotlib's plt.pie() or Seaborn's sns.countplot().

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")
counts = df['{x}'].value_counts()

plt.figure(figsize=(7, 6))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
plt.title("Proportion of {x.title()}", fontsize=14, fontweight='bold')
plt.axis('equal')
plt.show()"""

    elif chart_type == "pairplot":
        return f"""import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")
numeric_cols = ['math score', 'reading score', 'writing score']

# Pairwise scatter matrix with KDE diagonals and hue stratification in one call
g = sns.pairplot(df, vars=numeric_cols, hue='{hue}', palette="Set1", diag_kind="kde")
g.fig.subplots_adjust(top=0.92)
g.fig.suptitle("Pairwise Distributions Grouped by {hue.title()}", fontsize=14, fontweight='bold')
plt.show()"""

    return "# Code not available for this chart type."


def create_dashboard_html():
    """Returns the complete single-file HTML, CSS, and Vanilla JavaScript template string."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive EDA Dashboard & Python Data Science Guide</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --primary-light: #eff6ff;
            --dark: #0f172a;
            --slate: #334155;
            --muted: #64748b;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --border: #e2e8f0;
            --border-hover: #cbd5e1;
            --success: #059669;
            --warning: #d97706;
            --danger: #dc2626;
            --code-bg: #1e293b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--slate);
            line-height: 1.5;
        }

        header {
            background-color: var(--dark);
            color: #ffffff;
            padding: 24px 32px;
            border-bottom: 3px solid var(--primary);
        }

        .header-content {
            max-width: 1300px;
            margin: 0 auto;
        }

        .header-title {
            font-size: 26px;
            font-weight: 700;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-subtitle {
            color: #94a3b8;
            font-size: 14px;
            margin-top: 6px;
        }

        .badge {
            display: inline-block;
            padding: 3px 10px;
            font-size: 12px;
            font-weight: 600;
            border-radius: 9999px;
            background-color: rgba(37, 99, 235, 0.2);
            color: #93c5fd;
            border: 1px solid rgba(147, 197, 253, 0.3);
        }

        nav.tabs-nav {
            background-color: #ffffff;
            border-bottom: 1px solid var(--border);
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }

        .nav-container {
            max-width: 1300px;
            margin: 0 auto;
            display: flex;
            gap: 4px;
            overflow-x: auto;
            padding: 0 16px;
        }

        .tab-btn {
            background: none;
            border: none;
            padding: 16px 18px;
            font-size: 14px;
            font-weight: 600;
            color: var(--muted);
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.15s ease-in-out;
            white-space: nowrap;
        }

        .tab-btn:hover {
            color: var(--primary);
            background-color: var(--primary-light);
        }

        .tab-btn.active {
            color: var(--primary);
            border-bottom-color: var(--primary);
            background-color: #ffffff;
        }

        main {
            max-width: 1300px;
            margin: 24px auto;
            padding: 0 20px 48px;
        }

        .alert-error {
            background-color: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 24px;
            font-size: 15px;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .tab-pane {
            display: none;
        }

        .tab-pane.active {
            display: block;
        }

        .section-header {
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 22px;
            font-weight: 700;
            color: var(--dark);
        }

        .section-desc {
            font-size: 14px;
            color: var(--muted);
            margin-top: 4px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            transition: transform 0.1s ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: var(--border-hover);
        }

        .metric-label {
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--muted);
            margin-bottom: 6px;
        }

        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: var(--dark);
        }

        .metric-sub {
            font-size: 12px;
            color: var(--muted);
            margin-top: 4px;
        }

        .card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        }

        .card-title {
            font-size: 16px;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .control-panel {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: flex-end;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 180px;
            flex: 1;
        }

        .control-label {
            font-size: 13px;
            font-weight: 600;
            color: var(--slate);
        }

        select, input {
            padding: 9px 12px;
            font-size: 14px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background-color: #ffffff;
            color: var(--slate);
            outline: none;
            transition: border-color 0.15s;
        }

        select:focus, input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(37,99,235,0.1);
        }

        .btn-group {
            display: flex;
            border: 1px solid var(--border);
            border-radius: 6px;
            overflow: hidden;
        }

        .btn-toggle {
            padding: 8px 14px;
            font-size: 13px;
            font-weight: 600;
            background-color: #f1f5f9;
            color: var(--slate);
            border: none;
            cursor: pointer;
            transition: all 0.15s;
        }

        .btn-toggle:hover {
            background-color: #e2e8f0;
        }

        .btn-toggle.active {
            background-color: var(--primary);
            color: #ffffff;
        }

        .btn-action {
            background-color: var(--primary);
            color: #ffffff;
            font-size: 14px;
            font-weight: 600;
            padding: 9px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.15s;
        }

        .btn-action:hover {
            background-color: var(--primary-hover);
        }

        .chart-display-grid {
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 24px;
        }

        @media (max-width: 950px) {
            .chart-display-grid {
                grid-template-columns: 1fr;
            }
        }

        .chart-frame {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 440px;
            position: relative;
        }

        .chart-frame img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
        }

        .chart-loader {
            display: none;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 14px;
            font-weight: 600;
            color: var(--muted);
            background: rgba(255,255,255,0.9);
            padding: 12px 24px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }

        .explanation-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .info-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 6px;
            background: var(--primary-light);
            color: var(--primary);
        }

        .info-section {
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
        }

        .info-section:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }

        .info-heading {
            font-size: 13px;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .info-text {
            font-size: 13.5px;
            color: var(--slate);
            line-height: 1.5;
        }

        .code-container {
            margin-top: 20px;
            background: var(--code-bg);
            border-radius: 10px;
            overflow: hidden;
        }

        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #0f172a;
            padding: 10px 16px;
            border-bottom: 1px solid #334155;
        }

        .code-tabs {
            display: flex;
            gap: 8px;
        }

        .code-tab-btn {
            background: none;
            border: none;
            color: #94a3b8;
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
        }

        .code-tab-btn.active {
            background: #334155;
            color: #ffffff;
        }

        .btn-copy {
            background: #334155;
            border: none;
            color: #cbd5e1;
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.15s;
        }

        .btn-copy:hover {
            background: #475569;
            color: #ffffff;
        }

        pre code {
            display: block;
            padding: 16px;
            color: #f1f5f9;
            font-family: Consolas, Menlo, Monaco, "Courier New", monospace;
            font-size: 13px;
            line-height: 1.5;
            overflow-x: auto;
            max-height: 380px;
        }

        .data-table-container {
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
        }

        table.data-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
            background: #ffffff;
        }

        table.data-table th {
            background: #f1f5f9;
            color: var(--dark);
            font-weight: 600;
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            white-space: nowrap;
        }

        table.data-table td {
            padding: 10px 14px;
            border-bottom: 1px solid #f1f5f9;
            color: var(--slate);
            white-space: nowrap;
        }

        table.data-table tr:hover td {
            background-color: #f8fafc;
        }

        .comparison-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 24px;
        }

        @media (max-width: 850px) {
            .comparison-grid {
                grid-template-columns: 1fr;
            }
        }

        .cheat-sheet-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13.5px;
        }

        .cheat-sheet-table th {
            background: #f8fafc;
            color: var(--dark);
            font-weight: 700;
            padding: 12px 14px;
            border: 1px solid var(--border);
            text-align: left;
        }

        .cheat-sheet-table td {
            padding: 12px 14px;
            border: 1px solid var(--border);
            vertical-align: top;
            line-height: 1.5;
        }

        .cheat-sheet-table tr:nth-child(even) {
            background-color: #fafbfc;
        }

        .graph-badge {
            font-weight: 700;
            color: var(--primary);
            white-space: nowrap;
        }

        .tag {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            background: #e2e8f0;
            color: #475569;
            margin-right: 4px;
        }

        .tag-num {
            background: #dbeafe;
            color: #1e40af;
        }

        .tag-cat {
            background: #dcfce7;
            color: #166534;
        }

        .edu-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 16px;
        }

        .edu-card h4 {
            font-size: 15px;
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 6px;
        }

        .edu-card p {
            font-size: 13.5px;
            color: var(--slate);
        }
    </style>
</head>
<body>

    <header>
        <div class="header-content">
            <div class="header-title">
                <span>Interactive EDA Classroom Dashboard</span>
                <span class="badge">Python Data Science</span>
            </div>
            <div class="header-subtitle">
                A hands-on exploratory analysis teaching tool featuring Pandas, Matplotlib, Seaborn, and statistical interpretation.
            </div>
        </div>
    </header>

    <nav class="tabs-nav">
        <div class="nav-container">
            <button class="tab-btn active" onclick="switchTab('overview')">1. Overview</button>
            <button class="tab-btn" onclick="switchTab('concepts')">2. What is EDA?</button>
            <button class="tab-btn" onclick="switchTab('univariate')">3. Univariate Analysis</button>
            <button class="tab-btn" onclick="switchTab('bivariate')">4. Bivariate Analysis</button>
            <button class="tab-btn" onclick="switchTab('multivariate')">5. Multivariate Analysis</button>
            <button class="tab-btn" onclick="switchTab('mpl_vs_sns')">6. Matplotlib vs Seaborn</button>
            <button class="tab-btn" onclick="switchTab('comparison')">7. Graph Comparison Table</button>
        </div>
    </nav>

    <main>
        <!-- Missing File Error Alert -->
        <div id="missing-file-alert" class="alert-error" style="display: none;">
            <span>⚠️</span>
            <div>
                <strong>Dataset File Missing:</strong> Could not find <code>StudentsPerformance.csv</code> in the current folder.
                Please place the CSV file in the same directory as <code>eda_dashboard.py</code> and refresh this page.
            </div>
        </div>

        <!-- 1. OVERVIEW TAB -->
        <section id="pane-overview" class="tab-pane active">
            <div class="section-header">
                <h2 class="section-title">Dataset Overview & Summary Statistics</h2>
                <p class="section-desc">Key metadata, average exam marks, null checks, and dataset preview from StudentsPerformance.csv.</p>
            </div>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Students</div>
                    <div class="metric-value" id="card-students">--</div>
                    <div class="metric-sub">Total records in dataset</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Columns</div>
                    <div class="metric-value" id="card-columns">--</div>
                    <div class="metric-sub">Features analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Math Score</div>
                    <div class="metric-value" id="card-math" style="color: #2563eb;">--</div>
                    <div class="metric-sub">Out of 100 points</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Reading Score</div>
                    <div class="metric-value" id="card-reading" style="color: #059669;">--</div>
                    <div class="metric-sub">Out of 100 points</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Avg Writing Score</div>
                    <div class="metric-value" id="card-writing" style="color: #7c3aed;">--</div>
                    <div class="metric-sub">Out of 100 points</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Missing Values</div>
                    <div class="metric-value" id="card-missing">--</div>
                    <div class="metric-sub">Null or empty entries</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Duplicate Rows</div>
                    <div class="metric-value" id="card-duplicates">--</div>
                    <div class="metric-sub">Exact duplicate records</div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">
                    <span>First 5 Rows (Dataset Preview)</span>
                    <span style="font-size: 12px; font-weight: normal; color: var(--muted);">df.head(5)</span>
                </div>
                <div class="data-table-container">
                    <table class="data-table" id="preview-table">
                        <thead><tr id="preview-head"></tr></thead>
                        <tbody id="preview-body"></tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <div class="card-title">
                    <span>Feature Classifications & Types</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div>
                        <h4 style="font-size: 14px; margin-bottom: 8px;">Numerical Features (Continuous / Discrete):</h4>
                        <div id="num-cols-list" style="display: flex; flex-wrap: wrap; gap: 8px;"></div>
                    </div>
                    <div>
                        <h4 style="font-size: 14px; margin-bottom: 8px;">Categorical Features (Nominal / Ordinal):</h4>
                        <div id="cat-cols-list" style="display: flex; flex-wrap: wrap; gap: 8px;"></div>
                    </div>
                </div>
            </div>
        </section>

        <!-- 2. WHAT IS EDA? TAB -->
        <section id="pane-concepts" class="tab-pane">
            <div class="section-header">
                <h2 class="section-title">Core Concepts: What is Exploratory Data Analysis?</h2>
                <p class="section-desc">The teacher's primer on why EDA is the foundational starting point for every data science project.</p>
            </div>

            <div class="card">
                <h3 style="font-size: 18px; color: var(--dark); margin-bottom: 12px;">What is Exploratory Data Analysis (EDA)?</h3>
                <p style="font-size: 14.5px; line-height: 1.6; color: var(--slate); margin-bottom: 16px;">
                    Exploratory Data Analysis (EDA), pioneered by mathematician and statistician <strong>John Tukey in 1977</strong>,
                    is the systematic process of analyzing, summarizing, and visualizing datasets before applying machine learning algorithms or testing formal hypotheses.
                    Rather than imposing rigid assumptions onto raw numbers, EDA allows the data to "speak for itself" through summary metrics and visual representations.
                </p>

                <div class="edu-card">
                    <h4>Why is EDA Critical Before Machine Learning?</h4>
                    <p>
                        Training algorithms on dirty, skewed, or uninspected data leads to the classic <em>"Garbage In, Garbage Out"</em> pitfall.
                        Through EDA, you detect data collection errors, handle missing values, isolate severe outliers, verify feature correlations, and assess class imbalances.
                    </p>
                </div>
            </div>

            <div class="card">
                <h3 style="font-size: 18px; color: var(--dark); margin-bottom: 16px;">The 3 Core Levels of EDA</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;">
                    <div class="edu-card" style="border-left-color: #2563eb;">
                        <h4>1. Univariate Analysis</h4>
                        <p style="margin-bottom: 8px;"><strong>Examines 1 variable at a time in isolation.</strong></p>
                        <p><strong>Goal:</strong> Find the variable's center (mean, median), spread (IQR, standard deviation), distribution shape (skewness, bell curve), and extreme values.</p>
                        <p style="margin-top: 8px; font-size: 12px; color: var(--muted);"><strong>Key Charts:</strong> Histogram, Box Plot, Count Plot, Pie Chart.</p>
                    </div>

                    <div class="edu-card" style="border-left-color: #059669;">
                        <h4>2. Bivariate Analysis</h4>
                        <p style="margin-bottom: 8px;"><strong>Examines relationships between 2 variables.</strong></p>
                        <p><strong>Goal:</strong> Determine whether changes in one variable correspond with changes in another. Compares groups or evaluates statistical correlation.</p>
                        <p style="margin-top: 8px; font-size: 12px; color: var(--muted);"><strong>Key Charts:</strong> Scatter Plot, Grouped Bar Chart, Box Plot by Category, Line Chart.</p>
                    </div>

                    <div class="edu-card" style="border-left-color: #7c3aed;">
                        <h4>3. Multivariate Analysis</h4>
                        <p style="margin-bottom: 8px;"><strong>Examines 3 or more variables simultaneously.</strong></p>
                        <p><strong>Goal:</strong> Uncover complex interactions, multicollinearity, and multidimensional clustering without getting overwhelmed.</p>
                        <p style="margin-top: 8px; font-size: 12px; color: var(--muted);"><strong>Key Charts:</strong> Correlation Heatmap, Pair Plot (Scatter Matrix), Grouped Clustered Plots.</p>
                    </div>
                </div>
            </div>

            <div class="card">
                <h3 style="font-size: 18px; color: var(--dark); margin-bottom: 12px;">Python Tooling: Matplotlib vs Seaborn</h3>
                <p style="font-size: 14px; line-height: 1.6; color: var(--slate);">
                    Python offers two complementary charting libraries for data science:
                </p>
                <ul style="margin: 12px 0 12px 24px; font-size: 14px; color: var(--slate); line-height: 1.8;">
                    <li><strong>Matplotlib:</strong> The foundational, low-level plotting engine. It gives you microscopic control over every coordinate, spine, tick, and legend, but requires more lines of setup.</li>
                    <li><strong>Seaborn:</strong> A high-level statistical library built on top of Matplotlib. It natively connects to Pandas DataFrames, performs automatic statistical aggregations (like computing means and error bars), and provides beautiful modern color themes in a single line of code.</li>
                </ul>
            </div>
        </section>

        <!-- 3. UNIVARIATE ANALYSIS TAB -->
        <section id="pane-univariate" class="tab-pane">
            <div class="section-header">
                <h2 class="section-title">Univariate Analysis (Single Variable Exploration)</h2>
                <p class="section-desc">Inspect the statistical distribution, central tendency, frequency counts, or proportions of one feature.</p>
            </div>

            <div class="control-panel">
                <div class="control-group">
                    <label class="control-label" for="uni-col-select">Select Feature / Column:</label>
                    <select id="uni-col-select" onchange="onUnivariateColumnChange()"></select>
                </div>

                <div class="control-group">
                    <label class="control-label">Choose Graph Type:</label>
                    <div class="btn-group">
                        <button class="btn-toggle active" id="btn-uni-hist" onclick="setUnivariateChart('histogram')">Histogram</button>
                        <button class="btn-toggle" id="btn-uni-box" onclick="setUnivariateChart('box_plot')">Box Plot</button>
                        <button class="btn-toggle" id="btn-uni-count" onclick="setUnivariateChart('count_plot')">Count Plot</button>
                        <button class="btn-toggle" id="btn-uni-pie" onclick="setUnivariateChart('pie_chart')">Pie Chart</button>
                    </div>
                </div>

                <div class="control-group" style="max-width: 200px;">
                    <label class="control-label">Plotting Engine:</label>
                    <div class="btn-group">
                        <button class="btn-toggle active" id="btn-uni-sns" onclick="setUnivariateLibrary('seaborn')">Seaborn</button>
                        <button class="btn-toggle" id="btn-uni-mpl" onclick="setUnivariateLibrary('matplotlib')">Matplotlib</button>
                    </div>
                </div>

                <button class="btn-action" onclick="fetchUnivariateChart()">Update Chart</button>
            </div>

            <div class="chart-display-grid">
                <div class="chart-frame">
                    <div id="uni-loader" class="chart-loader">Rendering Chart...</div>
                    <img id="uni-chart-img" src="" alt="Univariate Chart Display">
                </div>

                <div class="explanation-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="info-pill" id="uni-pill-name">Chart</span>
                        <span class="tag" id="uni-pill-type">Numerical</span>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">📘 What is this graph?</div>
                        <div class="info-text" id="uni-exp-what">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">🎯 Primary Purpose</div>
                        <div class="info-text" id="uni-exp-purpose">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">⏱️ When to use it?</div>
                        <div class="info-text" id="uni-exp-when">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">🔍 What should students observe?</div>
                        <div class="info-text" id="uni-exp-observe">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">💡 Example Classroom Question</div>
                        <div class="info-text" id="uni-exp-question" style="font-style: italic; color: #1e40af;">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">⚠️ Limitations & Cautions</div>
                        <div class="info-text" id="uni-exp-limitations" style="color: #991b1b;">--</div>
                    </div>
                </div>
            </div>

            <div class="code-container">
                <div class="code-header">
                    <div class="code-tabs">
                        <button class="code-tab-btn active" id="btn-uni-code-sns" onclick="switchUniCodeTab('seaborn')">Seaborn Code</button>
                        <button class="code-tab-btn" id="btn-uni-code-mpl" onclick="switchUniCodeTab('matplotlib')">Matplotlib Code</button>
                    </div>
                    <button class="btn-copy" onclick="copyCode('uni-code-block')">Copy Python Code</button>
                </div>
                <pre><code id="uni-code-block"># Python code will appear here...</code></pre>
            </div>
        </section>

        <!-- 4. BIVARIATE ANALYSIS TAB -->
        <section id="pane-bivariate" class="tab-pane">
            <div class="section-header">
                <h2 class="section-title">Bivariate Analysis (Two-Variable Relationships)</h2>
                <p class="section-desc">Explore associations, category comparisons, and numerical correlations between two features.</p>
            </div>

            <div class="control-panel">
                <div class="control-group">
                    <label class="control-label" for="bi-x-select">X-Axis Feature (Grouping / Predictor):</label>
                    <select id="bi-x-select" onchange="autoConfigureBivariate()"></select>
                </div>

                <div class="control-group">
                    <label class="control-label" for="bi-y-select">Y-Axis Feature (Continuous Metric):</label>
                    <select id="bi-y-select"></select>
                </div>

                <div class="control-group">
                    <label class="control-label">Graph Type:</label>
                    <div class="btn-group">
                        <button class="btn-toggle active" id="btn-bi-bar" onclick="setBivariateChart('bar_chart')">Bar Chart</button>
                        <button class="btn-toggle" id="btn-bi-scatter" onclick="setBivariateChart('scatter_plot')">Scatter Plot</button>
                        <button class="btn-toggle" id="btn-bi-box" onclick="setBivariateChart('box_plot')">Box Plot by Cat</button>
                        <button class="btn-toggle" id="btn-bi-line" onclick="setBivariateChart('line_chart')">Line Chart</button>
                    </div>
                </div>

                <div class="control-group" style="max-width: 180px;">
                    <label class="control-label">Plotting Engine:</label>
                    <div class="btn-group">
                        <button class="btn-toggle active" id="btn-bi-sns" onclick="setBivariateLibrary('seaborn')">Seaborn</button>
                        <button class="btn-toggle" id="btn-bi-mpl" onclick="setBivariateLibrary('matplotlib')">Matplotlib</button>
                    </div>
                </div>

                <button class="btn-action" onclick="fetchBivariateChart()">Update Chart</button>
            </div>

            <div class="chart-display-grid">
                <div class="chart-frame">
                    <div id="bi-loader" class="chart-loader">Rendering Chart...</div>
                    <img id="bi-chart-img" src="" alt="Bivariate Chart Display">
                </div>

                <div class="explanation-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="info-pill" id="bi-pill-name">Chart</span>
                        <span class="tag" id="bi-pill-type">Bivariate</span>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">📘 What is this graph?</div>
                        <div class="info-text" id="bi-exp-what">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">🎯 Primary Purpose</div>
                        <div class="info-text" id="bi-exp-purpose">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">⏱️ When to use it?</div>
                        <div class="info-text" id="bi-exp-when">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">🔍 What should students observe?</div>
                        <div class="info-text" id="bi-exp-observe">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">💡 Example Classroom Question</div>
                        <div class="info-text" id="bi-exp-question" style="font-style: italic; color: #1e40af;">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">⚠️ Limitations & Cautions</div>
                        <div class="info-text" id="bi-exp-limitations" style="color: #991b1b;">--</div>
                    </div>
                </div>
            </div>

            <div class="code-container">
                <div class="code-header">
                    <div class="code-tabs">
                        <button class="code-tab-btn active" id="btn-bi-code-sns" onclick="switchBiCodeTab('seaborn')">Seaborn Code</button>
                        <button class="code-tab-btn" id="btn-bi-code-mpl" onclick="switchBiCodeTab('matplotlib')">Matplotlib Code</button>
                    </div>
                    <button class="btn-copy" onclick="copyCode('bi-code-block')">Copy Python Code</button>
                </div>
                <pre><code id="bi-code-block"># Python code will appear here...</code></pre>
            </div>
        </section>

        <!-- 5. MULTIVARIATE ANALYSIS TAB -->
        <section id="pane-multivariate" class="tab-pane">
            <div class="section-header">
                <h2 class="section-title">Multivariate Analysis (Multiple Feature Exploration)</h2>
                <p class="section-desc">Examine correlation heatmaps, pairwise bivariate matrices, and multi-variable score breakdowns.</p>
            </div>

            <div class="control-panel">
                <div class="control-group">
                    <label class="control-label">Multivariate Visual:</label>
                    <div class="btn-group">
                        <button class="btn-toggle active" id="btn-multi-heat" onclick="setMultiChart('heatmap')">Correlation Heatmap</button>
                        <button class="btn-toggle" id="btn-multi-pair" onclick="setMultiChart('pairplot')">Pair Plot (Scatter Matrix)</button>
                    </div>
                </div>

                <div class="control-group" id="multi-hue-group">
                    <label class="control-label" for="multi-hue-select">Stratify / Color Group (Hue):</label>
                    <select id="multi-hue-select"></select>
                </div>

                <div class="control-group" id="multi-lib-group" style="max-width: 180px;">
                    <label class="control-label">Plotting Engine:</label>
                    <div class="btn-group">
                        <button class="btn-toggle active" id="btn-multi-sns" onclick="setMultiLibrary('seaborn')">Seaborn</button>
                        <button class="btn-toggle" id="btn-multi-mpl" onclick="setMultiLibrary('matplotlib')">Matplotlib</button>
                    </div>
                </div>

                <button class="btn-action" onclick="fetchMultiChart()">Generate Multivariate View</button>
            </div>

            <div class="chart-display-grid">
                <div class="chart-frame">
                    <div id="multi-loader" class="chart-loader">Computing Multivariate Plot...</div>
                    <img id="multi-chart-img" src="" alt="Multivariate Chart Display">
                </div>

                <div class="explanation-card">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span class="info-pill" id="multi-pill-name">Heatmap</span>
                        <span class="tag tag-num">Multivariate</span>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">📘 What is this graph?</div>
                        <div class="info-text" id="multi-exp-what">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">🎯 Primary Purpose</div>
                        <div class="info-text" id="multi-exp-purpose">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">⏱️ When to use it?</div>
                        <div class="info-text" id="multi-exp-when">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">🔍 What should students observe?</div>
                        <div class="info-text" id="multi-exp-observe">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">💡 Example Classroom Question</div>
                        <div class="info-text" id="multi-exp-question" style="font-style: italic; color: #1e40af;">--</div>
                    </div>

                    <div class="info-section">
                        <div class="info-heading">⚠️ Limitations & Cautions</div>
                        <div class="info-text" id="multi-exp-limitations" style="color: #991b1b;">--</div>
                    </div>
                </div>
            </div>

            <div class="code-container">
                <div class="code-header">
                    <div class="code-tabs">
                        <button class="code-tab-btn active" id="btn-multi-code-sns" onclick="switchMultiCodeTab('seaborn')">Seaborn Code</button>
                        <button class="code-tab-btn" id="btn-multi-code-mpl" onclick="switchMultiCodeTab('matplotlib')">Matplotlib Code</button>
                    </div>
                    <button class="btn-copy" onclick="copyCode('multi-code-block')">Copy Python Code</button>
                </div>
                <pre><code id="multi-code-block"># Python code will appear here...</code></pre>
            </div>
        </section>

        <!-- 6. MATPLOTLIB VS SEABORN TAB -->
        <section id="pane-mpl_vs_sns" class="tab-pane">
            <div class="section-header">
                <h2 class="section-title">Matplotlib vs Seaborn (Side-by-Side Code Showdown)</h2>
                <p class="section-desc">Learn the practical differences between Python's foundational canvas and high-level statistical library.</p>
            </div>

            <div class="control-panel">
                <div class="control-group">
                    <label class="control-label">Select Graph Type to Compare:</label>
                    <div class="btn-group">
                        <button class="btn-toggle active" id="btn-cmp-hist" onclick="setComparisonChart('histogram')">Histogram</button>
                        <button class="btn-toggle" id="btn-cmp-bar" onclick="setComparisonChart('bar_chart')">Bar Chart</button>
                        <button class="btn-toggle" id="btn-cmp-box" onclick="setComparisonChart('box_plot')">Box Plot</button>
                        <button class="btn-toggle" id="btn-cmp-scatter" onclick="setComparisonChart('scatter_plot')">Scatter Plot</button>
                        <button class="btn-toggle" id="btn-cmp-heat" onclick="setComparisonChart('heatmap')">Heatmap</button>
                    </div>
                </div>
            </div>

            <div class="comparison-grid">
                <div class="code-container" style="margin-top: 0;">
                    <div class="code-header">
                        <span style="color: #93c5fd; font-weight: 700; font-size: 13px;">Matplotlib (Imperative & Low-Level)</span>
                        <button class="btn-copy" onclick="copyCode('side-mpl-code')">Copy</button>
                    </div>
                    <pre><code id="side-mpl-code"># Matplotlib code loading...</code></pre>
                </div>

                <div class="code-container" style="margin-top: 0;">
                    <div class="code-header">
                        <span style="color: #6ee7b7; font-weight: 700; font-size: 13px;">Seaborn (Declarative & High-Level)</span>
                        <button class="btn-copy" onclick="copyCode('side-sns-code')">Copy</button>
                    </div>
                    <pre><code id="side-sns-code"># Seaborn code loading...</code></pre>
                </div>
            </div>

            <div class="card">
                <h3 style="font-size: 17px; font-weight: 700; margin-bottom: 16px;">Core Architectural Differences for Beginners</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
                    <div class="edu-card" style="border-left-color: #3b82f6;">
                        <h4>1. Syntax & Boilerplate</h4>
                        <p><strong>Matplotlib</strong> requires manual setup of figure frames, loops for grouping, separate axis formatting calls, and explicit color mappings.</p>
                        <p style="margin-top: 6px;"><strong>Seaborn</strong> abstracts common statistical tasks into single declarative functions with intuitive parameters like <code>hue</code>, <code>kde</code>, and <code>palette</code>.</p>
                    </div>

                    <div class="edu-card" style="border-left-color: #10b981;">
                        <h4>2. Data Aggregation</h4>
                        <p><strong>Matplotlib</strong> requires you to pre-calculate means or frequencies using Pandas (e.g. <code>df.groupby()</code>) before feeding arrays to <code>plt.bar()</code>.</p>
                        <p style="margin-top: 6px;"><strong>Seaborn</strong> accepts the raw DataFrame directly and calculates group averages, confidence intervals, and category counts automatically under the hood.</p>
                    </div>

                    <div class="edu-card" style="border-left-color: #8b5cf6;">
                        <h4>3. Aesthetics & Theming</h4>
                        <p><strong>Matplotlib</strong> defaults to basic retro styling, requiring manual grid and spine configuration.</p>
                        <p style="margin-top: 6px;"><strong>Seaborn</strong> automatically applies modern statistical themes (e.g., <code>whitegrid</code>, <code>darkgrid</code>) and scientifically formulated color palettes.</p>
                    </div>
                </div>
            </div>
        </section>

        <!-- 7. GRAPH COMPARISON TABLE TAB -->
        <section id="pane-comparison" class="tab-pane">
            <div class="section-header">
                <h2 class="section-title">Graph Comparison Reference Sheet</h2>
                <p class="section-desc">A comprehensive summary matrix outlining when to use each of the 9 required EDA visualizations.</p>
            </div>

            <div class="card" style="padding: 0; overflow: hidden;">
                <div class="data-table-container" style="border: none;">
                    <table class="cheat-sheet-table">
                        <thead>
                            <tr>
                                <th style="width: 140px;">Graph Name</th>
                                <th style="width: 200px;">Purpose</th>
                                <th style="width: 200px;">When to Use</th>
                                <th style="width: 220px;">Example Classroom Question</th>
                                <th style="width: 140px;">Data Type</th>
                                <th>Limitations / Pitfalls</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td><span class="graph-badge">Bar Chart</span></td>
                                <td>Compare aggregated statistical values (mean, sum) across distinct groups.</td>
                                <td>Bivariate analysis comparing numeric averages across categories.</td>
                                <td>What is the average math score across different parental education tiers?</td>
                                <td><span class="tag tag-cat">Categorical X</span><span class="tag tag-num">Numeric Y</span></td>
                                <td>Hides variance, distribution spread, and outliers inside each group.</td>
                            </tr>
                            <tr>
                                <td><span class="graph-badge">Line Chart</span></td>
                                <td>Visualize continuous progression, sequential change, or ordered ranks.</td>
                                <td>Ordered categories (e.g. education levels) or continuous series.</td>
                                <td>Does academic performance steadily improve as parental education rises?</td>
                                <td><span class="tag tag-cat">Ordinal X</span><span class="tag tag-num">Numeric Y</span></td>
                                <td>Invalid for unordered nominal categories where line connections suggest a false sequence.</td>
                            </tr>
                            <tr>
                                <td><span class="graph-badge">Histogram</span></td>
                                <td>Show the frequency distribution and shape of a single continuous variable.</td>
                                <td>Univariate analysis of continuous scores or measures.</td>
                                <td>Are student reading scores normally distributed or skewed?</td>
                                <td><span class="tag tag-num">Continuous Numeric</span></td>
                                <td>Appearance varies heavily based on chosen bin width; obscures individual values.</td>
                            </tr>
                            <tr>
                                <td><span class="graph-badge">Box Plot</span></td>
                                <td>Summarize 5-number distribution (Min, Q1, Median, Q3, Max) and isolate outliers.</td>
                                <td>Univariate spread or bivariate group comparisons with outlier screening.</td>
                                <td>Are there extreme low outliers in math, and does spread vary by lunch type?</td>
                                <td><span class="tag tag-num">Numeric Y</span><span class="tag tag-cat">Optional Cat X</span></td>
                                <td>Can hide bimodal or multimodal peaks (combine with KDE or violin plot).</td>
                            </tr>
                            <tr>
                                <td><span class="graph-badge">Scatter Plot</span></td>
                                <td>Examine relationship, correlation direction, and clustering between two variables.</td>
                                <td>Bivariate analysis between two continuous numerical features.</td>
                                <td>How strongly does reading comprehension predict writing exam score?</td>
                                <td><span class="tag tag-num">Numeric X</span><span class="tag tag-num">Numeric Y</span></td>
                                <td>Points can suffer from overplotting when thousands of values overlap; correlation != causation.</td>
                            </tr>
                            <tr>
                                <td><span class="graph-badge">Heatmap</span></td>
                                <td>Color-coded correlation matrix displaying pairwise relationships.</td>
                                <td>Multivariate screening across all numerical features simultaneously.</td>
                                <td>Which subject pairs show the strongest pairwise correlation?</td>
                                <td><span class="tag tag-num">Numeric Matrix</span></td>
                                <td>Only measures linear correlation (Pearson); non-linear dependencies register near zero.</td>
                            </tr>
                            <tr>
                                <td><span class="graph-badge">Count Plot</span></td>
                                <td>Display raw frequency counts of discrete categories.</td>
                                <td>Univariate categorical inspection to check class balance.</td>
                                <td>How many students completed the test preparation course vs none?</td>
                                <td><span class="tag tag-cat">Categorical</span></td>
                                <td>Shows sample counts only; does not display numerical performance metrics.</td>
                            </tr>
                            <tr>
                                <td><span class="graph-badge">Pie Chart</span></td>
                                <td>Display proportional percentage share of a whole (100%).</td>
                                <td>Simple compositions with 2 to 5 distinct categories.</td>
                                <td>What proportion of students receive subsidized free/reduced lunch?</td>
                                <td><span class="tag tag-cat">Categorical (&le;5 levels)</span></td>
                                <td>Human vision poorly estimates slice angles and area; ineffective for many categories.</td>
                            </tr>
                            <tr>
                                <td><span class="graph-badge">Pair Plot</span></td>
                                <td>Grid of pairwise scatter plots and univariate distributions across all features.</td>
                                <td>Multivariate initial screening, optionally stratified by a category.</td>
                                <td>Do male and female students cluster differently across all three subjects?</td>
                                <td><span class="tag tag-num">Multiple Numeric</span><span class="tag tag-cat">Hue</span></td>
                                <td>Computationally heavy for large feature spaces; visual clutter beyond 5-6 features.</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    </main>

    <script>
        // Global state tracking
        const state = {
            summary: null,
            univariate: {
                column: 'math score',
                chartType: 'histogram',
                library: 'seaborn',
                codeTab: 'seaborn',
                code: { seaborn: '', matplotlib: '' }
            },
            bivariate: {
                x: 'gender',
                y: 'math score',
                chartType: 'bar_chart',
                library: 'seaborn',
                codeTab: 'seaborn',
                code: { seaborn: '', matplotlib: '' }
            },
            multivariate: {
                chartType: 'heatmap',
                hue: 'gender',
                library: 'seaborn',
                codeTab: 'seaborn',
                code: { seaborn: '', matplotlib: '' }
            },
            comparison: {
                chartType: 'histogram'
            }
        };

        // Navigation tab switching
        function switchTab(tabId) {
            document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));

            const targetPane = document.getElementById('pane-' + tabId);
            if (targetPane) targetPane.classList.add('active');

            const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.getAttribute('onclick').includes(tabId));
            if (activeBtn) activeBtn.classList.add('active');
        }

        // Initialize application on load
        window.addEventListener('DOMContentLoaded', () => {
            fetchSummary();
        });

        // 1. Fetch Dataset Summary & Populate UI
        async function fetchSummary() {
            try {
                const res = await fetch('/api/summary');
                const data = await res.json();

                if (data.error) {
                    document.getElementById('missing-file-alert').style.display = 'flex';
                    return;
                }

                state.summary = data;

                // Populate metric cards
                document.getElementById('card-students').textContent = data.total_students.toLocaleString();
                document.getElementById('card-columns').textContent = data.total_columns;
                document.getElementById('card-math').textContent = data.avg_math_score ?? 'N/A';
                document.getElementById('card-reading').textContent = data.avg_reading_score ?? 'N/A';
                document.getElementById('card-writing').textContent = data.avg_writing_score ?? 'N/A';
                document.getElementById('card-missing').textContent = data.total_missing;
                document.getElementById('card-duplicates').textContent = data.duplicate_count;

                // Populate feature classification lists
                const numContainer = document.getElementById('num-cols-list');
                numContainer.innerHTML = '';
                data.numeric_columns.forEach(col => {
                    const tag = document.createElement('span');
                    tag.className = 'tag tag-num';
                    tag.textContent = col;
                    numContainer.appendChild(tag);
                });

                const catContainer = document.getElementById('cat-cols-list');
                catContainer.innerHTML = '';
                data.categorical_columns.forEach(col => {
                    const tag = document.createElement('span');
                    tag.className = 'tag tag-cat';
                    tag.textContent = col;
                    catContainer.appendChild(tag);
                });

                // Populate preview table
                populatePreviewTable(data.preview, data.column_names);

                // Populate dropdowns
                populateDropdowns(data);

                // Load initial views
                fetchUnivariateChart();
                fetchBivariateChart();
                fetchMultiChart();
                updateComparisonCode();

            } catch (err) {
                console.error("Failed to load dataset summary:", err);
                document.getElementById('missing-file-alert').style.display = 'flex';
            }
        }

        function populatePreviewTable(rows, columns) {
            const headTr = document.getElementById('preview-head');
            headTr.innerHTML = '';
            columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                headTr.appendChild(th);
            });

            const tbody = document.getElementById('preview-body');
            tbody.innerHTML = '';
            rows.forEach(row => {
                const tr = document.createElement('tr');
                columns.forEach(col => {
                    const td = document.createElement('td');
                    td.textContent = row[col];
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
        }

        function populateDropdowns(data) {
            // Univariate column select
            const uniSelect = document.getElementById('uni-col-select');
            uniSelect.innerHTML = '';

            const numGroup = document.createElement('optgroup');
            numGroup.label = "Numerical Features";
            data.numeric_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c + " (Numeric)";
                numGroup.appendChild(opt);
            });
            uniSelect.appendChild(numGroup);

            const catGroup = document.createElement('optgroup');
            catGroup.label = "Categorical Features";
            data.categorical_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c + " (Categorical)";
                catGroup.appendChild(opt);
            });
            uniSelect.appendChild(catGroup);

            // Bivariate X select
            const biXSelect = document.getElementById('bi-x-select');
            biXSelect.innerHTML = '';
            const biCatGroup = document.createElement('optgroup');
            biCatGroup.label = "Categorical (Grouping)";
            data.categorical_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                biCatGroup.appendChild(opt);
            });
            biXSelect.appendChild(biCatGroup);

            const biNumGroup = document.createElement('optgroup');
            biNumGroup.label = "Numerical (Continuous)";
            data.numeric_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                biNumGroup.appendChild(opt);
            });
            biXSelect.appendChild(biNumGroup);

            // Bivariate Y select
            const biYSelect = document.getElementById('bi-y-select');
            biYSelect.innerHTML = '';
            data.numeric_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                biYSelect.appendChild(opt);
            });

            // Multivariate Hue select
            const multiHue = document.getElementById('multi-hue-select');
            multiHue.innerHTML = '';
            data.categorical_columns.forEach(c => {
                const opt = document.createElement('option');
                opt.value = c;
                opt.textContent = c;
                multiHue.appendChild(opt);
            });
        }

        // --- UNIVARIATE ANALYSIS LOGIC ---
        function onUnivariateColumnChange() {
            const col = document.getElementById('uni-col-select').value;
            state.univariate.column = col;
            const isNumeric = state.summary && state.summary.numeric_columns.includes(col);

            if (isNumeric && (state.univariate.chartType === 'count_plot' || state.univariate.chartType === 'pie_chart')) {
                setUnivariateChart('histogram');
            } else if (!isNumeric && (state.univariate.chartType === 'histogram' || state.univariate.chartType === 'box_plot')) {
                setUnivariateChart('count_plot');
            } else {
                fetchUnivariateChart();
            }
        }

        function setUnivariateChart(chartType) {
            state.univariate.chartType = chartType;
            ['hist', 'box', 'count', 'pie'].forEach(t => {
                const btn = document.getElementById('btn-uni-' + t);
                if (btn) btn.classList.remove('active');
            });
            const activeId = chartType === 'histogram' ? 'btn-uni-hist' :
                             chartType === 'box_plot' ? 'btn-uni-box' :
                             chartType === 'count_plot' ? 'btn-uni-count' : 'btn-uni-pie';
            document.getElementById(activeId).classList.add('active');
            fetchUnivariateChart();
        }

        function setUnivariateLibrary(lib) {
            state.univariate.library = lib;
            document.getElementById('btn-uni-sns').classList.toggle('active', lib === 'seaborn');
            document.getElementById('btn-uni-mpl').classList.toggle('active', lib === 'matplotlib');
            fetchUnivariateChart();
        }

        async function fetchUnivariateChart() {
            const loader = document.getElementById('uni-loader');
            loader.style.display = 'block';

            const col = document.getElementById('uni-col-select').value || state.univariate.column;
            state.univariate.column = col;

            const url = `/api/chart?type=${state.univariate.chartType}&column=${encodeURIComponent(col)}&library=${state.univariate.library}`;
            try {
                const res = await fetch(url);
                const data = await res.json();
                loader.style.display = 'none';

                if (data.status === 'success') {
                    document.getElementById('uni-chart-img').src = data.chart;
                    renderExplanation('uni', data.explanation);
                    state.univariate.code.seaborn = data.seaborn_code;
                    state.univariate.code.matplotlib = data.matplotlib_code;
                    updateUniCodeDisplay();
                } else {
                    alert(data.message || "Failed to render chart.");
                }
            } catch (err) {
                loader.style.display = 'none';
                console.error(err);
            }
        }

        function switchUniCodeTab(lib) {
            state.univariate.codeTab = lib;
            document.getElementById('btn-uni-code-sns').classList.toggle('active', lib === 'seaborn');
            document.getElementById('btn-uni-code-mpl').classList.toggle('active', lib === 'matplotlib');
            updateUniCodeDisplay();
        }

        function updateUniCodeDisplay() {
            const code = state.univariate.codeTab === 'seaborn' ? state.univariate.code.seaborn : state.univariate.code.matplotlib;
            document.getElementById('uni-code-block').textContent = code || "# Code will appear here...";
        }

        // --- BIVARIATE ANALYSIS LOGIC ---
        function autoConfigureBivariate() {
            const x = document.getElementById('bi-x-select').value;
            const isNumericX = state.summary && state.summary.numeric_columns.includes(x);

            if (isNumericX && state.bivariate.chartType !== 'scatter_plot') {
                setBivariateChart('scatter_plot');
            } else if (!isNumericX && state.bivariate.chartType === 'scatter_plot') {
                setBivariateChart('bar_chart');
            } else {
                fetchBivariateChart();
            }
        }

        function setBivariateChart(chartType) {
            state.bivariate.chartType = chartType;
            ['bar', 'scatter', 'box', 'line'].forEach(t => {
                const btn = document.getElementById('btn-bi-' + t);
                if (btn) btn.classList.remove('active');
            });
            const activeId = chartType === 'bar_chart' ? 'btn-bi-bar' :
                             chartType === 'scatter_plot' ? 'btn-bi-scatter' :
                             chartType === 'box_plot' ? 'btn-bi-box' : 'btn-bi-line';
            document.getElementById(activeId).classList.add('active');
            fetchBivariateChart();
        }

        function setBivariateLibrary(lib) {
            state.bivariate.library = lib;
            document.getElementById('btn-bi-sns').classList.toggle('active', lib === 'seaborn');
            document.getElementById('btn-bi-mpl').classList.toggle('active', lib === 'matplotlib');
            fetchBivariateChart();
        }

        async function fetchBivariateChart() {
            const loader = document.getElementById('bi-loader');
            loader.style.display = 'block';

            const x = document.getElementById('bi-x-select').value || state.bivariate.x;
            const y = document.getElementById('bi-y-select').value || state.bivariate.y;
            state.bivariate.x = x;
            state.bivariate.y = y;

            const url = `/api/chart?type=${state.bivariate.chartType}&x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}&library=${state.bivariate.library}`;
            try {
                const res = await fetch(url);
                const data = await res.json();
                loader.style.display = 'none';

                if (data.status === 'success') {
                    document.getElementById('bi-chart-img').src = data.chart;
                    renderExplanation('bi', data.explanation);
                    state.bivariate.code.seaborn = data.seaborn_code;
                    state.bivariate.code.matplotlib = data.matplotlib_code;
                    updateBiCodeDisplay();
                } else {
                    alert(data.message || "Failed to render bivariate chart.");
                }
            } catch (err) {
                loader.style.display = 'none';
                console.error(err);
            }
        }

        function switchBiCodeTab(lib) {
            state.bivariate.codeTab = lib;
            document.getElementById('btn-bi-code-sns').classList.toggle('active', lib === 'seaborn');
            document.getElementById('btn-bi-code-mpl').classList.toggle('active', lib === 'matplotlib');
            updateBiCodeDisplay();
        }

        function updateBiCodeDisplay() {
            const code = state.bivariate.codeTab === 'seaborn' ? state.bivariate.code.seaborn : state.bivariate.code.matplotlib;
            document.getElementById('bi-code-block').textContent = code || "# Code will appear here...";
        }

        // --- MULTIVARIATE ANALYSIS LOGIC ---
        function setMultiChart(type) {
            state.multivariate.chartType = type;
            document.getElementById('btn-multi-heat').classList.toggle('active', type === 'heatmap');
            document.getElementById('btn-multi-pair').classList.toggle('active', type === 'pairplot');

            // Hue is relevant for pairplot
            document.getElementById('multi-hue-group').style.display = type === 'pairplot' ? 'flex' : 'none';
            document.getElementById('multi-lib-group').style.display = type === 'heatmap' ? 'flex' : 'none';

            fetchMultiChart();
        }

        function setMultiLibrary(lib) {
            state.multivariate.library = lib;
            document.getElementById('btn-multi-sns').classList.toggle('active', lib === 'seaborn');
            document.getElementById('btn-multi-mpl').classList.toggle('active', lib === 'matplotlib');
            fetchMultiChart();
        }

        async function fetchMultiChart() {
            const loader = document.getElementById('multi-loader');
            loader.style.display = 'block';

            const hue = document.getElementById('multi-hue-select').value || state.multivariate.hue;
            state.multivariate.hue = hue;

            const url = `/api/chart?type=${state.multivariate.chartType}&hue=${encodeURIComponent(hue)}&library=${state.multivariate.library}`;
            try {
                const res = await fetch(url);
                const data = await res.json();
                loader.style.display = 'none';

                if (data.status === 'success') {
                    document.getElementById('multi-chart-img').src = data.chart;
                    renderExplanation('multi', data.explanation);
                    state.multivariate.code.seaborn = data.seaborn_code;
                    state.multivariate.code.matplotlib = data.matplotlib_code;
                    updateMultiCodeDisplay();
                } else {
                    alert(data.message || "Failed to render multivariate chart.");
                }
            } catch (err) {
                loader.style.display = 'none';
                console.error(err);
            }
        }

        function switchMultiCodeTab(lib) {
            state.multivariate.codeTab = lib;
            document.getElementById('btn-multi-code-sns').classList.toggle('active', lib === 'seaborn');
            document.getElementById('btn-multi-code-mpl').classList.toggle('active', lib === 'matplotlib');
            updateMultiCodeDisplay();
        }

        function updateMultiCodeDisplay() {
            const code = state.multivariate.codeTab === 'seaborn' ? state.multivariate.code.seaborn : state.multivariate.code.matplotlib;
            document.getElementById('multi-code-block').textContent = code || "# Code will appear here...";
        }

        // --- MATPLOTLIB VS SEABORN COMPARISON LOGIC ---
        function setComparisonChart(type) {
            state.comparison.chartType = type;
            ['hist', 'bar', 'box', 'scatter', 'heat'].forEach(t => {
                const btn = document.getElementById('btn-cmp-' + t);
                if (btn) btn.classList.remove('active');
            });
            const activeId = type === 'histogram' ? 'btn-cmp-hist' :
                             type === 'bar_chart' ? 'btn-cmp-bar' :
                             type === 'box_plot' ? 'btn-cmp-box' :
                             type === 'scatter_plot' ? 'btn-cmp-scatter' : 'btn-cmp-heat';
            document.getElementById(activeId).classList.add('active');
            updateComparisonCode();
        }

        async function updateComparisonCode() {
            const type = state.comparison.chartType;
            const x = type === 'histogram' ? 'math score' : (type === 'scatter_plot' ? 'reading score' : 'gender');
            const y = type === 'scatter_plot' ? 'writing score' : 'math score';
            const url = `/api/chart_code?type=${type}&x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}&hue=gender`;

            try {
                const res = await fetch(url);
                const data = await res.json();
                document.getElementById('side-mpl-code').textContent = data.matplotlib_code;
                document.getElementById('side-sns-code').textContent = data.seaborn_code;
            } catch (e) {
                console.error(e);
            }
        }

        // Render explanation cards
        function renderExplanation(prefix, exp) {
            if (!exp) return;
            document.getElementById(`${prefix}-pill-name`).textContent = exp.title;
            document.getElementById(`${prefix}-pill-type`).textContent = exp.data_type || 'Statistical';
            document.getElementById(`${prefix}-exp-what`).textContent = exp.what_is_it;
            document.getElementById(`${prefix}-exp-purpose`).textContent = exp.purpose;
            document.getElementById(`${prefix}-exp-when`).textContent = exp.when_to_use;
            document.getElementById(`${prefix}-exp-observe`).textContent = exp.what_to_observe;
            document.getElementById(`${prefix}-exp-question`).textContent = `"${exp.example_question}"`;
            document.getElementById(`${prefix}-exp-limitations`).textContent = exp.limitations;
        }

        // Copy code to clipboard helper
        function copyCode(elementId) {
            const text = document.getElementById(elementId).textContent;
            navigator.clipboard.writeText(text).then(() => {
                alert("Python code copied to clipboard!");
            }).catch(() => {
                alert("Please manually select and copy the code.");
            });
        }
    </script>
</body>
</html>
"""


def create_flask_routes(app):
    """Registers all web application routes and API endpoints on the Flask instance."""

    @app.route('/')
    def index():
        df = load_dataset()
        file_missing = df is None
        return render_template_string(create_dashboard_html(), file_missing=file_missing)

    @app.route('/api/summary')
    def api_summary():
        df = load_dataset()
        if df is None:
            return jsonify({
                "error": "Dataset file 'StudentsPerformance.csv' not found.",
                "message": "Please ensure StudentsPerformance.csv is placed in the same directory as eda_dashboard.py."
            }), 404
        return jsonify(get_dataset_summary(df))

    @app.route('/api/chart')
    def api_chart():
        df = load_dataset()
        if df is None:
            return jsonify({"status": "error", "message": "Dataset not found. Please place StudentsPerformance.csv in directory."}), 404

        chart_type = request.args.get('type', 'histogram')
        column = request.args.get('column', 'math score')
        x_col = request.args.get('x', column)
        y_col = request.args.get('y', 'reading score')
        hue_col = request.args.get('hue', 'gender')
        library = request.args.get('library', 'seaborn').lower()

        try:
            if chart_type == 'histogram':
                b64 = create_histogram(df, column, library=library)
            elif chart_type == 'bar_chart':
                b64 = create_bar_chart(df, x_col, y_col, library=library)
            elif chart_type == 'line_chart':
                b64 = create_line_chart(df, x_col, y_col, library=library)
            elif chart_type == 'box_plot':
                is_bivariate = x_col and x_col in df.columns and x_col != y_col and not pd.api.types.is_numeric_dtype(df[x_col])
                b64 = create_box_plot(df, y_column=y_col if is_bivariate else column, x_column=x_col if is_bivariate else None, library=library)
            elif chart_type == 'scatter_plot':
                b64 = create_scatter_plot(df, x_col, y_col, hue_column=hue_col, library=library)
            elif chart_type == 'heatmap':
                b64 = create_heatmap(df, library=library)
            elif chart_type == 'count_plot':
                b64 = create_count_plot(df, column, library=library)
            elif chart_type == 'pie_chart':
                b64 = create_pie_chart(df, column)
            elif chart_type == 'pairplot':
                b64 = create_pairplot(df, hue_column=hue_col)
            else:
                return jsonify({"status": "error", "message": f"Unsupported chart type: {chart_type}"}), 400

            explanation = get_chart_explanation(chart_type)
            mpl_code = generate_matplotlib_code(chart_type, x_column=x_col, y_column=y_col, hue_column=hue_col)
            sns_code = generate_seaborn_code(chart_type, x_column=x_col, y_column=y_col, hue_column=hue_col)

            return jsonify({
                "status": "success",
                "chart": f"data:image/png;base64,{b64}",
                "explanation": explanation,
                "matplotlib_code": mpl_code,
                "seaborn_code": sns_code
            })

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    @app.route('/api/chart_code')
    def api_chart_code():
        chart_type = request.args.get('type', 'histogram')
        x_col = request.args.get('x', 'math score')
        y_col = request.args.get('y', 'reading score')
        hue_col = request.args.get('hue', 'gender')

        mpl_code = generate_matplotlib_code(chart_type, x_column=x_col, y_column=y_col, hue_column=hue_col)
        sns_code = generate_seaborn_code(chart_type, x_column=x_col, y_column=y_col, hue_column=hue_col)

        return jsonify({
            "status": "success",
            "matplotlib_code": mpl_code,
            "seaborn_code": sns_code
        })


def main():
    """Main entrypoint: creates Flask application, registers routes, and starts server."""
    app = Flask(__name__)
    create_flask_routes(app)

    print("=" * 65)
    print("  EXPLORATORY DATA ANALYSIS (EDA) DASHBOARD STARTED")
    print("=" * 65)
    print(f"  * Dataset Location : {os.path.abspath(CSV_PATH)}")
    print(f"  * Dataset Exists   : {os.path.exists(CSV_PATH)}")
    print("  * Dashboard URL    : http://127.0.0.1:5000")
    print("=" * 65)
    print("  Press Ctrl+C to stop the Flask server.")
    print("=" * 65)

    app.run(host="127.0.0.1", port=5000, debug=True)


if __name__ == "__main__":
    main()
