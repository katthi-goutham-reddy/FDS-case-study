# FDS Case Studies 2 and 5 (Batch 2)

Subject: Fundamentals of Data Science | College: Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad
Student: <NAME> | Roll No: <ROLL NO> | Section: <SECTION> | Year: III Year, V Semester (2026-27) | Faculty: <FACULTY NAME>
Deadline: 26-10-2026. Ask the user for every <PLACEHOLDER> above before building the PDFs; never leave one in a deliverable.

## Faculty submission guidelines (must all be satisfied)
1. Report title page: case study title, Roll no, Name, section, year, subject, faculty name.
2. Report sections: problem statement, methodology, statistical findings, model evaluation, conclusion.
3. Every graph has a title, axis labels and a written interpretation. Never a graph alone.
4. Statistical findings and model performance are explained, not just displayed.
5. Deliver an executed Jupyter Notebook (.ipynb) with code, outputs and visualizations.
6. Deliver a PDF report with the complete analysis.
7. Notebook + Python code pushed to GitHub; repo link submitted with the PDF.
- No code without explaining its purpose and the observation obtained.
- AI assistance must be understood, verified and attributed (see Attribution below).

## Environment (Windows, VS Code)
- Create a venv at repo root: `py -m venv .venv`, activate it, `pip install -r requirements.txt`.
- After installing, pin the exact installed versions of the listed packages in requirements.txt.

## Rules for BOTH notebooks
- Structure: title cell (all title-page details above) -> objective -> numbered sections mirroring the task list of the brief.
- Before every code cell: a short markdown cell "Purpose:" saying what the cell does and why.
- After every code cell with output: a markdown cell "Observation:" written ONLY from the actual executed output (real numbers). Never invent or guess numbers.
- Workflow: write the notebook, execute it with `jupyter nbconvert --to notebook --execute --inplace <file>.ipynb`, read the outputs, write/update every Observation cell with the real values, re-execute to confirm it runs top to bottom with no errors.
- Plain, clear student voice. No em dashes anywhere in markdown or report text.
- `random_state=42` everywhere it applies.
- Every plot: title, x/y axis labels, legend where there is more than one series, readable tick labels. `plt.tight_layout()`, save to `figures/NN_short_name.png` (dpi=150, inside the case study folder), then `plt.show()`.
- Save key results for the report into `results/` inside the case study folder (metrics as JSON/CSV, tables as CSV).
- Final cells: Conclusion (markdown) and Attribution: "Code structure drafted with AI assistance (Claude); executed, verified and interpreted by <NAME>. Dataset sources: ...".
- Use paths relative to the notebook's folder (notebooks live inside `2/` and `5/`).

---------------------------------------------------------------
## Case Study 2: Clustering Analysis (K-Means, Hierarchical, DBSCAN)
Folder `2/`. Brief: `2/Clustering Analysis.docx`. Data: `2/EastWestAirlines.xlsx`, sheet `data` (sheet `Description` holds the data dictionary; use it to describe each field). Needs `openpyxl`.
Notebook: `2/case_study_2_clustering_eastwest_airlines.ipynb`.
Business context (from the Description sheet): East-West Airlines wants to identify customer segments from frequent flyer behaviour; source: Shmueli and Bruce (2016), company names changed.

1. Data Preprocessing:
   - Load, shape, head, dtypes, describe, data dictionary table. Missing values (count; handle if any). Duplicates.
   - Drop `ID#` from features. Keep `Award?` OUT of the clustering features and use it only to profile clusters afterwards (explain why). cc1/cc2/cc3_miles are ordinal bins (1 to 5); keep as numeric and say so.
   - Skewness table; the mile/balance columns are heavily right-skewed, so apply `np.log1p` to skewed columns (skew > 1) and explain.
   - Outliers: show IQR counts per feature on raw data (explain why IQR removal on raw skewed data would discard too much), then remove rows with |z| > 3 on the log-transformed features; report how many rows were removed and the % of data.
   - StandardScaler on the final feature matrix.
2. EDA (multiple visualizations): histograms raw vs log-transformed, box plots, correlation heatmap, pair plot of 4 to 5 key features, PCA 2D scatter of the scaled data (with explained variance %) to look for visible groupings. Write the patterns seen.
3. K-Means: elbow curve (inertia for K = 1 to 10) AND silhouette score for K = 2 to 10; choose K and justify with both. Fit final model (n_init=10).
4. Hierarchical (Agglomerative): dendrograms (truncated, `scipy.cluster.hierarchy`) for ward, complete, average and single linkage; table of silhouette scores for each linkage at the chosen number of clusters; pick the best linkage and justify (note single linkage chaining if seen).
5. DBSCAN: k-distance plot (k = min_samples) to choose eps; grid over eps (around the knee) and min_samples (e.g. 5, 10, 2 x n_features); table of n_clusters, noise points and %, silhouette (computed on non-noise points only, say so). Pick final parameters and justify.
6. Visualization: PCA 2D scatter for each final model, clusters in different colours (noise in grey for DBSCAN), with legends.
7. Evaluation: comparison table of the three final models: number of clusters, silhouette score (required), plus Davies-Bouldin and Calinski-Harabasz as supporting metrics (explain each: higher/lower is better). Save to `2/results/clustering_comparison.csv`.
8. Cluster Analysis and Interpretation: cluster sizes; profile table of mean ORIGINAL (unscaled) feature values per cluster plus Award? rate; heatmap of standardized cluster means; give each K-Means cluster a business name (e.g. "frequent flyers", "dormant members") with the evidence; compare how the three algorithms segment customers; insights and marketing recommendations in short comments.
9. Conclusion and limitations (e.g. choice of K is partly subjective, results depend on the transformation and scaling, DBSCAN struggles with varying density in this data, PCA 2D view loses information).

---------------------------------------------------------------
## Case Study 5: Wine Quality with Multiple Linear Regression
Folder `5/`. Brief: `5/Wine_Quality.docx`. Data: `5/wine+quality.zip` containing `winequality-red.csv` and `winequality-white.csv` (`sep=';'`) and `winequality.names` (citation and attribute info). Extract to `5/data/`.
Dataset choice: use the RED wine file (1,599 samples, 11 physicochemical inputs + quality) as the primary dataset, and state why in the notebook (brief asks for the 11 attributes; mixing red and white would need a 12th "type" variable and they have different chemistry). <If the user asks for white or combined, change this.>
Notebook: `5/case_study_5_wine_quality_regression.ipynb`.

0. Data understanding: load, shape, head, dtypes, describe, missing values, duplicate rows (count them; drop for modelling and explain, report the shape after).
Task 1 Statistical analysis:
 a. Mean, median, mode, standard deviation and skewness for alcohol, volatile acidity and residual sugar in one table (if mode is not unique, report the first and note it). Interpret each skewness.
 b. Coefficient of variation (std / mean x 100) for the three attributes; state which has the greatest relative variability and why CV (not std) is the right measure (different units/scales).
 c. Pearson correlation between alcohol and quality with `scipy.stats.pearsonr`: r, p-value, H0/H1, alpha = 0.05, decision; interpret direction and strength.
Task 2 EDA:
 a. Histogram (bar chart of counts per score) of quality; interpret shape, most common scores, imbalance.
 b. Box plot of alcohol grouped by quality score; describe the trend and outliers.
 c. Correlation heatmap of all 11 attributes + quality (annotated).
 d. Rank attributes by absolute Pearson r with quality; name the top two, sign and strength.
Task 3 Predictive modelling:
 a/b. Multiple linear regression (scikit-learn `LinearRegression`) on all 11 attributes; 80:20 split with random_state=42.
 c. Test-set MAE, MSE, RMSE, R2 in a table; explain each in units of quality points. Also show train R2 for an over/underfitting check.
 d. Standardized coefficients: fit on StandardScaler-transformed X (fit on train only); bar chart ranked by |coef|; name the two most influential predictors. Multicollinearity BEFORE interpreting: VIF table (`statsmodels`), point out strongly correlated pairs (e.g. fixed acidity, density, citric acid, pH) and what that means for interpretation. Also fit `statsmodels` OLS on the training data to report coefficient p-values.
 e. Scatter plot of predicted vs actual quality on the test set with the y = x line (add small jitter to actual scores and say so); residual plot. Explain limitations: discrete ordinal target, predictions cluster around 5 to 6, extremes (3, 4, 8) are poorly predicted, linear assumptions.
Task 4 Business decision-making:
 a. Recommend two measurements for lab screening, backed by correlation, standardized coefficients and p-values.
 b. Is the model reliable enough to replace sensory evaluation? Answer using test R2 and RMSE (what an error of that size means on a 0 to 10 scale). Expected answer: use as a screening aid, not a replacement, unless the numbers say otherwise.
 c. Two limitations (e.g. only red Vinho Verde, quality is a subjective median of expert scores, class imbalance in quality, no grape/brand/price data, linear model).

---------------------------------------------------------------
## Repo / GitHub
Repo root = this folder (suggested name `fds-case-studies-2-5`). README.md describing both case studies, setup, how to run the notebooks and the report scripts, key findings. Commit executed notebooks (with outputs), `build_report.py` scripts, figures, results, report PDFs, data files. Do not commit `.venv`.

## PDF reports (built by Claude Code AFTER both notebooks are executed and verified)
Tool: `reportlab` (pure Python, no LaTeX). Build each report with `<folder>/build_report.py` that reads ONLY the saved `figures/` and `results/` files, so every number in the PDF comes from the executed notebook. Never type a metric into the report by hand.
Output: `2/report/CS2_Clustering_Analysis_<ROLL NO>.pdf` and `5/report/CS5_Wine_Quality_Regression_<ROLL NO>.pdf`.

Layout: A4, 2 cm margins, one clean font, page numbers in the footer, numbered section headings, figures captioned "Figure N: ..." and tables "Table N: ...". Figures about 14 to 16 cm wide. No em dashes.

Structure (faculty prescribed):
1. Title page: case study title, Name, Roll No, Section, Year, Subject, Faculty name, College, date of submission. Nothing else on the page.
2. Problem Statement and Objectives (from the brief, in the student's own words).
3. Dataset Description: source, rows/columns, feature list with types/meaning, data quality issues found.
4. Methodology: preprocessing steps and why, algorithms/model and parameters, evaluation metrics and why. Include one methodology workflow diagram drawn with matplotlib in hand-drawn style (`plt.xkcd()`), saved in figures/.
5. Statistical Findings: key EDA/statistics figures and tables, each followed by an interpretation paragraph with real numbers (CS-5: Task 1 table, CV, Pearson test, Task 2 plots).
6. Model Evaluation: CS-2 elbow/silhouette, dendrogram, k-distance, cluster scatter plots, comparison table. CS-5 metrics table, standardized coefficients, VIF, predicted vs actual, residuals.
7. Interpretation: CS-2 cluster profiles and business segment names. CS-5 Task 4 business decisions.
8. Conclusion and Limitations.
9. References: dataset source (CS-5: Cortez et al., 2009 citation exactly as in `winequality.names`; CS-2: Shmueli and Bruce, 2016 as stated in the xlsx Description sheet), libraries used, GitHub repository link. No invented references.
10. Short AI-assistance acknowledgement line (same wording as the notebook attribution).

Length: body about 4 to 7 pages each (title page excluded). Choose the most informative figures instead of every plot.
After building, render pages to PNG (pymupdf) and inspect: readable figures, no overflow, no placeholders, numbers match the notebook. Ask for missing title-page details and the GitHub repo URL before the final build.
