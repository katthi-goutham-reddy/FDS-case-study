# Fundamentals of Data Science (FDS) Case Studies

**Course:** Fundamentals of Data Science  
**Institution:** Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad  
**Academic Year:** III Year, V Semester (2026-27)  

This repository contains the complete implementation, executed Jupyter Notebooks, visualization artifacts, generated datasets, and formal PDF reports for Case Studies 2 and 5.

---

## Repository Structure

```text
.
├── 2/                                              # Case Study 2: Clustering Analysis
│   ├── EastWestAirlines.xlsx                       # Dataset (sheet 'data' & 'Description')
│   ├── Clustering Analysis.docx                    # Assignment brief
│   ├── case_study_2_clustering_eastwest_airlines.ipynb  # Executed notebook
│   ├── build_report.py                             # ReportLab PDF report generation script
│   ├── figures/                                    # Exported visualization plots & workflow
│   ├── results/                                    # Metrics JSON and tabular CSV summaries
│   └── report/                                     # Generated formal PDF report
├── 5/                                              # Case Study 5: Wine Quality Regression
│   ├── wine+quality.zip                            # Source dataset archive
│   ├── data/                                       # Extracted CSV datasets
│   ├── Wine_Quality.docx                           # Assignment brief
│   ├── case_study_5_wine_quality_regression.ipynb  # Executed notebook
│   ├── build_report.py                             # ReportLab PDF report generation script
│   ├── figures/                                    # Exported visualization plots & workflow
│   ├── results/                                    # Metrics JSON and tabular CSV summaries
│   └── report/                                     # Generated formal PDF report
├── requirements.txt                                # Pinned package versions
├── CLAUDE.md                                       # Authoritative project guidelines & spec
├── PROMPT.md                                       # Mentorship instructions & workflow
└── README.md                                       # Project documentation
```

---

## Case Study 2: Customer Segmentation using Clustering

### Objective
Segment 3,999 frequent flyer customers of East-West Airlines based on flight activity, mileage accrual patterns, credit card partner usage, and account tenure using unsupervised learning algorithms: K-Means, Agglomerative Hierarchical clustering, and DBSCAN.

### Key Findings
1. **Preprocessing:** Eight features exhibited severe positive skewness (cc3_miles: 17.20, Balance: 5.00). Applying `np.log1p` compressed long tails. Outlier removal using $|z| > 3$ on log-transformed features removed 376 extreme records (9.40%), yielding 3,623 clean records without dropping legitimate high-tier flyers.
2. **K-Means Clustering:** Both inertia elbow analysis and silhouette score profiles peaked at $K=4$ (Silhouette score = 0.3199, Calinski-Harabasz = 1575.63, Davies-Bouldin = 0.9021).
3. **Hierarchical Clustering:** Ward linkage produced the most cohesive tree with the highest silhouette score (0.3317 at $K=4$). Single linkage exhibited chaining (silhouette = 0.2641).
4. **DBSCAN:** Using $k=5$ nearest-neighbour knee analysis ($\text{eps} = 0.879$, $\text{min\_samples} = 10$), DBSCAN identified 3 dense customer cores while setting aside 220 records (6.07%) as noise.
5. **Business Personas Identified:**
   - **Cluster 0 (Active Frequent Flyers, 27.7%):** High flight miles (1,076 mi/yr), 54.2% award flight redemption rate. Strategy: Tier retention and lounge perks.
   - **Cluster 1 (Credit Card Loyalty Earners, 45.0%):** High credit card bonus miles (20,900 mi) but zero recent flights. Strategy: Travel redemption discounts to convert mileage to flights.
   - **Cluster 2 (Inactive / Low Engagement, 27.2%):** Low balances (22,418 mi), minimal activity, 11.8% award rate. Strategy: Re-engagement campaigns.
   - **Cluster 3 (Outlier Account, <0.1%):** High qualifying flight activity.

---

## Case Study 5: Wine Quality Prediction via Multiple Linear Regression

### Objective
Analyze physicochemical properties of red Vinho Verde wine (1,599 samples) to model and predict sensory quality ratings (scale 0-10) using statistical tests and Multiple Linear Regression.

### Key Findings
1. **Data Cleaning:** 240 duplicate rows (15.01%) were removed to eliminate data leakage and avoid optimistic evaluation bias, leaving 1,359 unique observations.
2. **Statistical Analysis:**
   - **Relative Variability:** Residual sugar exhibited the highest Coefficient of Variation (CV = 53.59%) compared to Volatile Acidity (34.57%) and Alcohol (10.37%).
   - **Hypothesis Testing:** Two-tailed Pearson correlation test between alcohol and quality yielded $r = +0.4803$ ($p = 2.28 \times 10^{-79}$), rejecting $H_0$ at $\alpha = 0.05$.
3. **Multicollinearity & Diagnostics:**
   - Fixed acidity (VIF = 7.95) and density (VIF = 6.31) showed moderate-to-high multicollinearity ($\text{VIF} > 5$).
   - Standardized coefficients ranked **Alcohol** ($\beta = +0.3340$) and **Volatile Acidity** ($\beta = -0.1785$) as the two most impactful predictors.
4. **Model Performance:**
   - Multiple Linear Regression on 80:20 train-test split: $\text{MAE} = 0.5041$, $\text{RMSE} = 0.6565$ quality points, Test $R^2 = 0.3547$, Train $R^2 = 0.3601$.
5. **Business Recommendations:**
   - Recommend automated vat screening for Alcohol (grape maturity) and Volatile Acidity (bacterial spoilage).
   - Because $R^2 \approx 0.35$ and RMSE is 0.66 on a discrete 3-8 rating scale, the model cannot replace sensory tasting panels; it serves as a cost-effective pre-screening tool.

---

## Environment Setup & Execution

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Running Jupyter Notebooks
To run the notebooks top-to-bottom and update in-place:
```bash
# Case Study 2
cd 2
jupyter nbconvert --to notebook --execute --inplace case_study_2_clustering_eastwest_airlines.ipynb

# Case Study 5
cd ../5
jupyter nbconvert --to notebook --execute --inplace case_study_5_wine_quality_regression.ipynb
```

### 3. Generating PDF Reports
The ReportLab PDF scripts dynamically read outputs from `figures/` and `results/`:
```bash
# Case Study 2 Report
cd 2
python build_report.py "STUDENT_NAME" "ROLL_NO" "SECTION" "FACULTY_NAME" "GITHUB_URL"

# Case Study 5 Report
cd ../5
python build_report.py "STUDENT_NAME" "ROLL_NO" "SECTION" "FACULTY_NAME" "GITHUB_URL"
```

---

## References
1. Shmueli, G., and Bruce, P. C. (2016). *Data Mining for Business Analytics: Concepts, Techniques, and Applications in Python*. John Wiley & Sons.
2. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., and Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, Elsevier, 47(4):547-553.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
