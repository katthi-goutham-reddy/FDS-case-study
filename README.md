# Fundamentals of Data Science (FDS) Case Studies 2 & 5

<p align="center">
  <img src="cbit_logo.jpeg" alt="CBIT Logo" width="120"/>
</p>

**Institution:** Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad  
**Department:** Department of Computer Science and Engineering  
**Course:** Fundamentals of Data Science (22CSC59)  
**Academic Year:** III Year, V Semester (2026–27)  
**Student Name:** Katthi Goutham Reddy  
**Roll Number:** 160124733179  
**Section:** CSE-3  
**Faculty In-Charge:** Dr. S. Durga Devi  
**GitHub Repository:** [https://github.com/katthi-goutham-reddy/FDS-case-study](https://github.com/katthi-goutham-reddy/FDS-case-study)

---

## Overview

This repository contains the deliverables for the practical assessment in **Fundamentals of Data Science (22CSC59)** covering two assigned case studies:

1. **Case Study 2: Customer Segmentation using Clustering Analysis (East-West Airlines)**
   - Unsupervised machine learning applying **K-Means**, **Hierarchical (Agglomerative)**, and **DBSCAN** clustering on 3,999 frequent flyer accounts.
   - Comprehensive data preprocessing including log transformations (`log1p`) for heavily right-skewed mileage features, standard z-score scaling, and 2D Principal Component Analysis (PCA).
   - Hyperparameter optimization via the Elbow method (SSE inertia), Silhouette coefficient analysis, linkage dendrogram evaluations (Ward, Complete, Average, Single), and k-distance knee plots for DBSCAN.
   - Distinct profiling of 4 business segments: *Dormant Base (Segment 0)*, *Elite Transactors (Segment 1)*, *Infrequent Card Spenders (Segment 2)*, and *High-Mileage Premium Flyers (Segment 3)*.

2. **Case Study 5: Wine Quality Prediction via Multiple Linear Regression (Red Vinho Verde)**
   - Statistical testing and predictive modeling on 1,599 red wine physicochemical records.
   - Descriptive statistics, skewness diagnostics, and Coefficient of Variation (CV) dispersion analysis.
   - Formal two-tailed Pearson correlation hypothesis testing ($r = 0.4762$, $p = 2.83 \times 10^{-91}$) rejecting $H_0$ with extreme significance.
   - Full Multiple Linear Regression (OLS) with 80/20 train-test evaluation, 5-fold cross-validation stability ($R^2 = 0.352 \pm 0.054$), Multicollinearity assessment (all VIF < 8.0), and standardized regression coefficients ranking feature impacts.

---

## Repository Files

```text
.
├── .gitattributes                               # Git line-ending and binary file tracking configuration
├── .gitignore                                   # Ignore rules for virtual environments, caches, and temp files
├── cbit_logo.jpeg                               # CBIT institutional shield logo
├── EastWestAirlines.xlsx                        # Dataset for Case Study 2 (East-West Airlines frequent flyers)
├── winequality-red.csv                          # Dataset for Case Study 5 (Vinho Verde red wine physicochemical attributes)
├── FDS_Case_Studies_2_and_5_160124733179.ipynb  # Unified, fully executed Jupyter Notebook for both case studies
├── FDS_Case_Studies_2_and_5_160124733179.py     # Standalone executable Python script with all analysis pipelines
├── FDS_Case_Studies_Report_160124733179.pdf     # Official 21-page CBIT-format institutional technical report
├── README.md                                    # Project documentation and summary
└── requirements.txt                             # Pinned Python dependencies
```

---

## Deliverables Summary

| Deliverable | File Name | Description |
| :--- | :--- | :--- |
| **Official Report** | `FDS_Case_Studies_Report_160124733179.pdf` | Complete 21-page institutional report following CBIT format (Certificate, Abstract, TOC, List of Figures/Tables, Chapters 1–4, and References). |
| **Interactive Notebook** | `FDS_Case_Studies_2_and_5_160124733179.ipynb` | Google Colab / Jupyter-ready notebook with markdown documentation, code cells, and inline figures. |
| **Executable Script** | `FDS_Case_Studies_2_and_5_160124733179.py` | Complete Python source code implementing both analytical workflows end-to-end. |
| **Datasets** | `EastWestAirlines.xlsx`, `winequality-red.csv` | Original datasets used for reproducing all results and model metrics. |

---

## Setup & Execution

### 1. Environment Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/katthi-goutham-reddy/FDS-case-study.git
cd FDS-case-study
pip install -r requirements.txt
```

### 2. Running the Python Script

Execute the full data analysis pipeline directly:

```bash
python FDS_Case_Studies_2_and_5_160124733179.py
```

### 3. Launching the Jupyter Notebook

Open and run the interactive notebook in Jupyter Lab or Notebook:

```bash
jupyter notebook FDS_Case_Studies_2_and_5_160124733179.ipynb
```

---

## Author & Academic Information

- **Student Name:** Katthi Goutham Reddy
- **Roll Number:** 160124733179
- **Branch & Section:** Computer Science and Engineering (CSE-3)
- **Institution:** Chaitanya Bharathi Institute of Technology (Autonomous), Gandipet, Hyderabad - 500075
