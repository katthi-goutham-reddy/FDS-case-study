# Fundamentals of Data Science (FDS) Case Studies

**Course:** Fundamentals of Data Science (22CSC59)  
**Institution:** Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad  
**Department:** Computer Science and Engineering  
**Academic Year:** III Year, V Semester (2026-27)  
**Student Name:** Katthi Goutham Reddy  
**Roll Number:** 160124733179  
**Section:** CSE-3  
**Faculty In-Charge:** Dr. S. Durga Devi  
**Repository:** [https://github.com/katthi-goutham-reddy/FDS-case-study](https://github.com/katthi-goutham-reddy/FDS-case-study)

---

## Overview

This repository contains the complete analytical implementations, executable code, Jupyter notebooks, datasets, exported visualizations, and institutional reports for:
1. **Case Study 2:** Customer Segmentation using Clustering Analysis (East-West Airlines)
2. **Case Study 5:** Wine Quality Prediction via Multiple Linear Regression (Red Vinho Verde)

---

## Repository Structure

```text
.
├── FDS_Case_Studies_2_and_5_160124733179_CBIT_format.pdf   # Official 21-page combined CBIT format report
├── FDS case study - 2,5.pdf                               # Combined report alias
├── fds_case_studies_2_and_5.ipynb                         # Unified Jupyter notebook for both case studies
├── build_cbit_report.py                                   # Automated ReportLab report generator
├── cbit_logo.png                                          # CBIT shield logo for PDF header
├── requirements.txt                                       # Pinned Python package dependencies
├── README.md                                              # Project documentation
│
├── Case Study 2 — Airline Customer Segmentation/          # Dedicated Case Study 2 directory
│   ├── FDS_Case_Study_2_Airline_Clustering.ipynb          # Jupyter notebook for customer clustering
│   ├── FDS_Case_Study_2_Airline_Clustering.py             # Executable Python script
│   ├── FDS_Case_Study_2_Airline_Clustering_Report.pdf     # Individual Case Study 2 PDF report
│   └── EastWestAirlines.xlsx                              # Airline mileage dataset
│
├── Case Study 5 — Wine Quality Assessment/                # Dedicated Case Study 5 directory
│   ├── FDS_Case_Study_5_Wine_Quality.ipynb                # Jupyter notebook for wine quality regression
│   ├── FDS_Case_Study_5_Wine_Quality.py                   # Executable Python script
│   ├── FDS_Case_Study_5_Wine_Quality_Report.pdf           # Individual Case Study 5 PDF report
│   └── winequality-red.csv                                # Red wine physicochemical dataset
│
├── casestudies-jupyternb/                                 # Centralized Jupyter Notebooks folder
│   ├── FDS_Case_Study_2_Airline_Clustering.ipynb          # Case Study 2 notebook
│   ├── FDS_Case_Study_5_Wine_Quality.ipynb                # Case Study 5 notebook
│   └── fds_case_studies_2_and_5.ipynb                     # Combined comprehensive notebook
│
├── data/                                                  # Shared raw datasets
│   ├── EastWestAirlines.xlsx
│   └── winequality-red.csv
├── figures/                                               # Generated figures and charts
└── results/                                               # Exported CSV metric summaries and tables
```

---

## Case Study 2: Customer Segmentation using Clustering (East-West Airlines)

### Objective
Segment 3,999 frequent flyer accounts of East-West Airlines based on flight activity, mileage accrual patterns, credit card partner usage, and account tenure using unsupervised learning algorithms: K-Means, Agglomerative Hierarchical clustering, and DBSCAN.

### Key Findings & Methodology
1. **Preprocessing:** Severe positive skewness in mileage features was handled using natural log transformation `np.log1p`. Outliers were identified using $|z| > 3$ on log-transformed features (removing 376 extreme records, 9.40%), yielding 3,623 clean records without dropping legitimate high-tier flyers. Standard scaling (`StandardScaler`) was then applied.
2. **K-Means Clustering:** Both inertia elbow analysis and silhouette score profiles indicated optimal clustering at $K=4$ (Silhouette score = 0.3199, Calinski-Harabasz = 1575.63, Davies-Bouldin = 0.9021).
3. **Hierarchical Clustering:** Ward linkage produced the most cohesive tree with the highest silhouette score (0.3317 at $K=4$). Complete linkage achieved 0.2453, and Single linkage exhibited chaining (0.2641).
4. **DBSCAN:** Using 5-nearest neighbor knee analysis ($\text{eps} = 0.879$, $\text{min\_samples} = 10$), DBSCAN identified 3 dense customer cores while setting aside 220 records (6.07%) as noise.
5. **Business Personas Identified:**
   - **Cluster 0 (Active Frequent Flyers, 27.7%):** High flight miles (1,076 mi/yr), 54.2% award flight redemption rate. *Strategy:* Retain tier status with airport lounge access and seat upgrades.
   - **Cluster 1 (Credit Card Loyalty Earners, 45.0%):** High credit card bonus miles (20,900 mi) but zero recent flights. *Strategy:* Travel redemption discounts to convert mileage to flights.
   - **Cluster 2 (Inactive / Low Engagement, 27.2%):** Low balances (22,418 mi), minimal activity, 11.8% award rate. *Strategy:* Re-activation campaigns and milestone bonus challenges.
   - **Cluster 3 (Elite Outliers, <0.1%):** High qualifying flight activity requiring dedicated corporate concierge support.

---

## Case Study 5: Wine Quality Prediction via Multiple Linear Regression

### Objective
Analyze physicochemical properties of red Vinho Verde wine (1,599 samples) to model and predict sensory quality ratings (scale 0-10) using statistical tests and Multiple Linear Regression.

### Key Findings & Methodology
1. **Data Cleaning:** 240 duplicate rows (15.01%) were removed to eliminate data leakage and avoid optimistic evaluation bias, leaving 1,359 unique observations.
2. **Statistical Analysis:**
   - **Relative Variability:** Residual sugar exhibited the highest Coefficient of Variation (CV = 53.59%) compared to Volatile Acidity (34.57%) and Alcohol (10.37%).
   - **Hypothesis Testing:** Two-tailed Pearson correlation test between alcohol and quality yielded $r = +0.4803$ ($t = 20.34$, $p = 2.28 \times 10^{-79}$), rejecting $H_0$ at $\alpha = 0.05$.
3. **Multicollinearity & Diagnostics:**
   - Fixed acidity (VIF = 7.95) and density (VIF = 6.31) showed moderate-to-high multicollinearity ($\text{VIF} > 5$).
   - Standardized coefficients ranked **Alcohol** ($\beta = +0.3340$) and **Volatile Acidity** ($\beta = -0.1785$) as the two most impactful predictors.
4. **Model Performance:**
   - Multiple Linear Regression on 80:20 train-test split: $\text{MAE} = 0.5041$, $\text{MSE} = 0.4310$, $\text{RMSE} = 0.6565$ quality points, Test $R^2 = 0.3915$, Train $R^2 = 0.3520$.
5. **Business Recommendations:**
   - Recommend automated vat screening for Alcohol (grape maturity) and Volatile Acidity (bacterial spoilage).
   - Because $R^2 \approx 0.39$ and RMSE is 0.66 on a discrete 3-8 rating scale, the model serves as an automated pre-screening filter to detect flawed batches rather than replacing certified human sensory panels.

---

## Execution Instructions

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Unified Jupyter Notebook
To execute the combined notebook top-to-bottom and update in-place:
```bash
jupyter nbconvert --to notebook --execute --inplace fds_case_studies_2_and_5.ipynb
```

### 3. Generating the PDF Report
To build the 21-page institutional CBIT format PDF report:
```bash
python build_cbit_report.py
```
Output PDF is saved as `FDS_Case_Studies_2_and_5_160124733179_CBIT_format.pdf`.

---

## References
[1] G. Shmueli and P. C. Bruce, *Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python*, John Wiley & Sons, 2016.  
[2] P. Cortez, A. Cerdeira, F. Almeida, T. Matos, and J. Reis, “Modeling wine preferences by data mining from physicochemical properties,” *Decision Support Systems*, vol. 47, no. 4, pp. 547-553, 2009.  
[3] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.  
[4] S. Seabold and J. Perktold, “Statsmodels: Econometric and statistical modeling with Python,” *Proceedings of the 9th Python in Science Conference*, 2010.  
