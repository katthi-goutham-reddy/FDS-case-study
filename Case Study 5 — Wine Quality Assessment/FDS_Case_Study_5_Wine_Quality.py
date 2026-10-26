# %% [markdown]
# # Case Study 5: Wine Quality Analysis with Multiple Linear Regression
# 
# 
# 
# | Field | Details |
# 
# |---|---|
# 
# | **Subject** | Fundamentals of Data Science |
# 
# | **College** | Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad |
# 
# | **Student** | Katthi Goutham Reddy |
# 
# | **Roll No** | 160124733179 |
# 
# | **Section** | CSE-3 |
# 
# | **Year** | III Year, V Semester (2026-27) |
# 
# | **Faculty** | Dr. S. Durga Devi |
# 
# | **Date** | 26-10-2026 |
# 
# | **GitHub** | https://github.com/katthi-goutham-reddy/FDS-case-study |
# 
# 
# 
# ## Objective
# 
# 
# 
# Perform statistical analysis and build a multiple linear regression model to predict red wine quality from 11 physicochemical features. Provide evidence-based recommendations for winery lab screening.
# 
# 
# 
# **Dataset choice:** We use the red wine file (`winequality-red.csv`, 1,599 samples). The brief asks for the 11 physicochemical attributes; mixing red and white would require a 12th 'type' variable and they have different chemistry, which would complicate interpretation without adding value to this analysis.

# %% [markdown]
# ## Section 0: Data Understanding

# %% [markdown]
# **Purpose:** Extract the dataset from the zip file, load the red wine CSV, and perform initial exploration: shape, head, dtypes, describe, missing values, and duplicates.

# %%
import zipfile, os, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Extract zip
zip_path = 'wine+quality.zip'
with zipfile.ZipFile(zip_path, 'r') as z:
    z.extractall('data')
print('Extracted files:', os.listdir('data'))

df = pd.read_csv('data/winequality-red.csv', sep=';')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
df.head()

# %% [markdown]
# **Observation:** The red wine dataset has 1,599 rows and 12 columns (11 physicochemical features and quality). Quality scores range from 3 to 8 with mean 5.64.

# %% [markdown]
# **Purpose:** Check data types, descriptive statistics, missing values, and duplicate rows. Drop duplicates before modelling and report the shape after.

# %%
print('=== Data Types ===')
print(df.dtypes)
print()
print('=== Descriptive Statistics ===')
print(df.describe())
print()
print('=== Missing Values ===')
print(df.isnull().sum())
print()
n_dups = df.duplicated().sum()
print(f'=== Duplicates: {n_dups} rows ===')
df = df.drop_duplicates().reset_index(drop=True)
print(f'Shape after dropping duplicates: {df.shape}')
print('Duplicates are removed because identical rows inflate effective sample size and bias statistical tests.')

# %% [markdown]
# **Observation:** All 11 physicochemical features are continuous float64, and quality is integer int64. There are no missing values. There are 240 duplicate rows (15.01%), which are removed to avoid data leakage and artificial inflation of model performance, leaving 1,359 unique observations.

# %% [markdown]
# ## Task 1: Statistical Analysis

# %% [markdown]
# **Purpose:** Compute mean, median, mode, standard deviation, and skewness for alcohol, volatile acidity, and residual sugar in a single table. Interpret each skewness value.

# %%
target_cols = ['alcohol', 'volatile acidity', 'residual sugar']
stat_rows = []
for col in target_cols:
    mode_val = df[col].mode()
    mode_str = str(round(mode_val.iloc[0], 4))
    if len(mode_val) > 1:
        mode_str += ' (first; not unique)'
    stat_rows.append({
        'Attribute': col,
        'Mean': round(df[col].mean(), 4),
        'Median': round(df[col].median(), 4),
        'Mode': mode_str,
        'Std_Dev': round(df[col].std(), 4),
        'Skewness': round(df[col].skew(), 4)
    })
stat_df = pd.DataFrame(stat_rows)
print('=== Task 1a: Descriptive Statistics ===')
print(stat_df.to_string(index=False))
stat_df.to_csv('results/task1_stats.csv', index=False)
print()
print('Skewness interpretation:')
print('  alcohol: moderate right-skew if > 0.5, meaning a tail of high-alcohol wines.')
print('  volatile acidity: if right-skewed, most wines have low VA but a few have very high VA.')
print('  residual sugar: typically highly right-skewed; most wines are dry with a few sweet outliers.')

# %% [markdown]
# **Observation:** For alcohol, mean is 10.43%, median is 10.20%, mode is 9.50%, std is 1.08%, and skewness is +0.86 (moderately right-skewed). For volatile acidity, mean is 0.53 g/dm3, median is 0.52, mode is 0.50, std is 0.18, and skewness is +0.73 (moderately right-skewed). For residual sugar, mean is 2.52 g/dm3, median is 2.20, mode is 2.00, std is 1.35, and skewness is +4.55 (strongly right-skewed, showing most wines are dry with few sweet outliers).

# %% [markdown]
# **Purpose:** Compute the Coefficient of Variation (CV = std/mean * 100) for the three attributes. CV is the right measure here because the three attributes have different units and scales; comparing raw standard deviations would be misleading.

# %%
cv_rows = []
for col in target_cols:
    cv = (df[col].std() / df[col].mean()) * 100
    cv_rows.append({'Attribute': col, 'Mean': round(df[col].mean(), 4),
                    'Std_Dev': round(df[col].std(), 4), 'CV(%)': round(cv, 2)})
cv_df = pd.DataFrame(cv_rows)
print('=== Task 1b: Coefficient of Variation ===')
print(cv_df.to_string(index=False))
cv_df.to_csv('results/task1_cv.csv', index=False)
print()
max_cv_col = cv_df.loc[cv_df['CV(%)'].idxmax(), 'Attribute']
print(f'Greatest relative variability: {max_cv_col}')
print('CV is preferred over std because it normalises variability relative to the mean,'
      ' making comparisons across different units and scales valid.')

# %% [markdown]
# **Observation:** Coefficient of variation is 10.37% for alcohol, 34.57% for volatile acidity, and 53.59% for residual sugar. Residual sugar shows the greatest relative variability. CV is the appropriate comparative measure because it normalizes standard deviation by the mean, allowing meaningful scale-free comparison across attributes measured in different units.

# %% [markdown]
# **Purpose:** Compute Pearson correlation between alcohol and quality with a formal hypothesis test. H0: no linear relationship (r=0). H1: there is a linear relationship. Alpha=0.05.

# %%
r, p_val = pearsonr(df['alcohol'], df['quality'])
alpha = 0.05
decision = 'Reject H0' if p_val < alpha else 'Fail to reject H0'
print('=== Task 1c: Pearson Correlation: alcohol vs quality ===')
print(f'H0: r = 0 (no linear relationship)')
print(f'H1: r != 0 (linear relationship exists)')
print(f'Alpha: {alpha}')
print(f'Pearson r: {r:.4f}')
print(f'p-value: {p_val:.6e}')
print(f'Decision: {decision}')
strength = 'strong' if abs(r) > 0.5 else ('moderate' if abs(r) > 0.3 else 'weak')
direction = 'positive' if r > 0 else 'negative'
print(f'Interpretation: {strength} {direction} linear relationship between alcohol and quality.')
pearson_result = {'r': round(r, 4), 'p_value': float(f'{p_val:.6e}'), 'decision': decision}
with open('results/task1_pearson.json', 'w') as f:
    json.dump(pearson_result, f, indent=2)

# %% [markdown]
# **Observation:** Pearson correlation between alcohol and quality is r = +0.4803 with p-value = 2.28e-79. At alpha = 0.05, we reject H0 (r = 0) and conclude there is a statistically significant, moderate positive linear association between alcohol content and wine quality.

# %% [markdown]
# ## Task 2: Exploratory Data Analysis

# %% [markdown]
# **Purpose:** Bar chart of quality score counts to understand the distribution. This shows the most common scores, the shape of the distribution, and any class imbalance.

# %%
fig, ax = plt.subplots(figsize=(9, 5.2))
quality_counts = df['quality'].value_counts().sort_index()
total_wines = len(df)
bars = ax.bar(quality_counts.index.astype(str), quality_counts.values, 
              color=['#6baed6', '#3182bd', '#722f37', '#8e3a43', '#3182bd', '#6baed6'], 
              edgecolor='white', linewidth=1.2)

# Annotate each bar with Count and Percentage
for i, (score, count) in enumerate(quality_counts.items()):
    pct = (count / total_wines) * 100
    ax.text(i, count + 12, f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

# Highlight central 81.8% concentration
ax.axvspan(1.5, 3.5, color='#722f37', alpha=0.08, label='Core Grades 5 & 6 (81.8% of dataset)')
ax.set_ylim(0, max(quality_counts.values) + 85)
ax.set_title('Distribution of Red Wine Sensory Quality Ratings\n(Demonstrating Heavy Imbalance and Concentration around Scores 5 and 6)', 
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Quality Score (Sensory Expert Median Rating: 0 - 10 scale)', fontsize=10.5, fontweight='bold')
ax.set_ylabel('Number of Wine Samples', fontsize=10.5, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(loc='upper right', fontsize=9)
plt.tight_layout()
plt.savefig('figures/01_quality_distribution.png', dpi=150)
plt.show()
print('Quality score distribution:')
print(quality_counts)
print(f'Most common score: {quality_counts.idxmax()} ({quality_counts.max()} wines, {quality_counts.max()/total_wines*100:.1f}%)')

# %% [markdown]
# **Observation:** Quality scores range from 3 to 8, with scores 5 (577 wines, 42.5%) and 6 (535 wines, 39.4%) accounting for 81.8% of the entire dataset. Scores 3 (10 wines, 0.7%) and 8 (17 wines, 1.3%) are extreme tails. This heavy concentration in mid-range tiers creates substantial class imbalance and foreshadows regression shrinkage toward the mean.
# 

# %% [markdown]
# **Purpose:** Box plot of alcohol grouped by quality score to see whether higher-quality wines tend to have higher alcohol content, and to spot outliers per quality level.

# %%
fig, ax = plt.subplots(figsize=(10, 5.8))
quality_levels = sorted(df['quality'].unique())
quality_groups = [df.loc[df['quality'] == q, 'alcohol'].values for q in quality_levels]

wine_palette = ['#e0ecf4', '#bfd3e6', '#9ebcda', '#8c96c6', '#8c6bb1', '#810f7c']
bp = ax.boxplot(quality_groups, tick_labels=[str(q) for q in quality_levels], patch_artist=True,
                showmeans=True, meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='yellow', markersize=6),
                medianprops=dict(color='black', linewidth=1.8),
                flierprops=dict(marker='o', markersize=4, markerfacecolor='#722f37', alpha=0.5))

for patch, color in zip(bp['boxes'], wine_palette):
    patch.set_facecolor(color)
    patch.set_edgecolor('#333333')

# Annotate trend
medians = [np.median(g) for g in quality_groups]
ax.plot(range(1, len(quality_levels)+1), medians, 'r--', alpha=0.6, label='Median Trend')
ax.annotate('Systematic upward trend:\nMedian alcohol rises from 9.6% (score 5)\nto 12.5% (score 8)', 
            xy=(5, medians[4]), xytext=(2.2, 13.8),
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5),
            fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='#fee2e2', edgecolor='#d62728', alpha=0.85))

ax.set_title('Alcohol Content by Wine Quality Rating (Task 2b)\n(Yellow diamonds = mean; black lines = median; red dashed line = median trend)', 
             fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Quality Score', fontsize=10.5, fontweight='bold')
ax.set_ylabel('Alcohol Content (% vol)', fontsize=10.5, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('figures/02_alcohol_by_quality_boxplot.png', dpi=150)
plt.show()
print('Saved: figures/02_alcohol_by_quality_boxplot.png')

# %% [markdown]
# **Observation:** Box plots display an upward trend in median alcohol content as quality increases: median alcohol rises systematically from 9.60% at score 5 to 10.50% at score 6, 11.60% at score 7, and 12.50% at score 8. A collection of high-alcohol outliers appears in score 5 (reaching up to 14.9%), explaining why alcohol alone cannot perfectly discriminate quality without complementary chemical markers.
# 

# %% [markdown]
# **Purpose:** Annotated correlation heatmap of all 11 attributes + quality. This shows which features are most correlated with quality and with each other (multicollinearity).

# %%
fig, ax = plt.subplots(figsize=(12, 9.5))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            vmin=-0.7, vmax=0.7, linewidths=0.6, linecolor='white',
            annot_kws={'size': 8.5}, cbar_kws={'label': 'Pearson Correlation Coefficient (r)'}, ax=ax)
ax.set_title('Correlation Heatmap: 11 Physicochemical Attributes and Sensory Quality\n(Annotated with Pearson r; showing key multicollinear pairs and target associations)', 
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('figures/03_correlation_heatmap.png', dpi=150)
plt.show()
print('Saved: figures/03_correlation_heatmap.png')

# %% [markdown]
# **Observation:** Correlation analysis reveals that quality correlates most positively with alcohol (+0.48) and sulphates (+0.25), and most negatively with volatile acidity (-0.40). Strong inter-feature correlations exist between fixed acidity and citric acid (+0.67), fixed acidity and density (+0.67), and fixed acidity and pH (-0.68).

# %% [markdown]
# **Purpose:** Rank all 11 physicochemical attributes by absolute Pearson correlation with quality to identify the most predictive features.

# %%
feature_cols = [c for c in df.columns if c != 'quality']
corr_with_quality = df[feature_cols].corrwith(df['quality']).abs().sort_values(ascending=False)
corr_signed = df[feature_cols].corrwith(df['quality']).sort_values(key=abs, ascending=False)
rank_df = pd.DataFrame({'Attribute': corr_signed.index, 'Pearson_r': corr_signed.values.round(4),
                        'Abs_r': corr_with_quality.values.round(4)})
print('=== Task 2d: Attributes ranked by |Pearson r| with quality ===')
print(rank_df.to_string(index=False))
rank_df.to_csv('results/task2_feature_correlation_rank.csv', index=False)
top1 = rank_df.iloc[0]
top2 = rank_df.iloc[1]
print(f'Top 1: {top1["Attribute"]} (r={top1["Pearson_r"]:.4f})')
print(f'Top 2: {top2["Attribute"]} (r={top2["Pearson_r"]:.4f})')

# %% [markdown]
# **Observation:** Ranked by absolute Pearson r with quality: 1. Alcohol (|r| = 0.4803, positive), 2. Volatile acidity (|r| = 0.3952, negative), 3. Sulphates (|r| = 0.2488, positive), 4. Citric acid (|r| = 0.2281, positive), 5. Density (|r| = 0.1843, negative).

# %% [markdown]
# ## Task 3: Predictive Modelling

# %% [markdown]
# **Purpose:** Split the data 80:20, fit multiple linear regression on all 11 attributes, and compute test-set metrics. Train R2 is also shown for an over/underfitting check.

# %%
X = df[feature_cols].values
y = df['quality'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}')

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2_test = r2_score(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)

metrics_df = pd.DataFrame([
    {'Metric': 'MAE', 'Value': round(mae, 4), 'Meaning': 'Average absolute error in quality points'},
    {'Metric': 'MSE', 'Value': round(mse, 4), 'Meaning': 'Average squared error (penalises large errors more)'},
    {'Metric': 'RMSE', 'Value': round(rmse, 4), 'Meaning': 'Root MSE in quality points (same unit as target)'},
    {'Metric': 'R2 (test)', 'Value': round(r2_test, 4), 'Meaning': 'Fraction of quality variance explained by the model'},
    {'Metric': 'R2 (train)', 'Value': round(r2_train, 4), 'Meaning': 'Train R2 for overfitting check'}
])
print('=== Task 3c: Regression Metrics ===')
print(metrics_df.to_string(index=False))
metrics_df.to_csv('results/task3_regression_metrics.csv', index=False)

# %% [markdown]
# **Observation:** Multiple linear regression on 80:20 split (1,087 train, 272 test) yielded MAE = 0.5041, MSE = 0.4310, RMSE = 0.6565, and test R2 = 0.3547. Train R2 was 0.3601, indicating consistent generalization without overfitting.

# %% [markdown]
# **Purpose:** Check multicollinearity using the Variance Inflation Factor (VIF) before interpreting coefficients. VIF > 5 indicates moderate multicollinearity; VIF > 10 is a concern. Then compute standardized coefficients (fit on scaled X) and rank by |coefficient| to identify the most influential predictors. Also fit statsmodels OLS to get coefficient p-values.

# %%
# VIF on training data
X_train_df = pd.DataFrame(X_train, columns=feature_cols)
vif_data = pd.DataFrame()
vif_data['Feature'] = feature_cols
vif_data['VIF'] = [variance_inflation_factor(X_train_df.values, i) for i in range(len(feature_cols))]
vif_data['VIF'] = vif_data['VIF'].round(2)
vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)
print('=== VIF Table (Training Data) ===')
print(vif_data.to_string(index=False))
print('VIF > 5: moderate multicollinearity. VIF > 10: high multicollinearity.')
high_vif = vif_data[vif_data['VIF'] > 5]['Feature'].tolist()
print(f'Features with VIF > 5: {high_vif}')
print('These features are correlated with each other, so individual coefficients may be unstable.')
vif_data.to_csv('results/task3_vif.csv', index=False)

# %% [markdown]
# **Observation:** VIF analysis on training data shows fixed acidity (VIF = 7.95) and density (VIF = 6.31) exhibit moderate-to-high multicollinearity (VIF > 5). Remaining 9 features have VIF < 5 (pH: 3.42, alcohol: 3.16, citric acid: 3.11). This collinearity among acids and density means individual partial slope coefficients must be interpreted with caution.

# %%
# Standardized coefficients
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)

lr_std = LinearRegression()
lr_std.fit(X_train_std, y_train)

std_coef_df = pd.DataFrame({'Feature': feature_cols, 'Std_Coef': lr_std.coef_.round(4)})
std_coef_df['Abs_Coef'] = std_coef_df['Std_Coef'].abs()
std_coef_df = std_coef_df.sort_values('Abs_Coef', ascending=False).reset_index(drop=True)
print('=== Standardized Coefficients ===')
print(std_coef_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10.5, 6))
# Sort ascending so top feature is plotted at top via horizontal bar chart
std_coef_plot = std_coef_df.sort_values('Abs_Coef', ascending=True).reset_index(drop=True)
colors_bar = ['#2b5c8f' if c >= 0 else '#d9534f' for c in std_coef_plot['Std_Coef']]
bars = ax.barh(std_coef_plot['Feature'], std_coef_plot['Std_Coef'], color=colors_bar, edgecolor='white', height=0.65)
ax.axvline(0, color='black', linewidth=1)

# Annotate each bar with value
for bar, val in zip(bars, std_coef_plot['Std_Coef']):
    offset = 0.008 if val >= 0 else -0.008
    ha = 'left' if val >= 0 else 'right'
    sign = '+' if val > 0 else ''
    ax.text(val + offset, bar.get_y() + bar.get_height()/2, f'{sign}{val:.4f}', 
            va='center', ha=ha, fontsize=9, fontweight='bold')

ax.set_xlim(-0.24, 0.40)
ax.set_title('Standardized Regression Coefficients (Ranked by Absolute Magnitude |β|)\n(Blue = Positive impact on quality; Red = Negative impact; fit on standardized training features)', 
             fontsize=11.5, fontweight='bold', pad=12)
ax.set_xlabel('Standardized Regression Coefficient (β)', fontsize=10.5, fontweight='bold')
ax.set_ylabel('Physicochemical Feature', fontsize=10.5, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.4)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2b5c8f', label='Positive Predictor (Increases Quality)'),
    Patch(facecolor='#d9534f', label='Negative Predictor (Decreases Quality / Spoilage)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig('figures/04_standardized_coefficients.png', dpi=150)
plt.show()
std_coef_df.to_csv('results/task3_standardized_coefs.csv', index=False)
print(f'Top predictor: {std_coef_df.iloc[0]["Feature"]} (coef={std_coef_df.iloc[0]["Std_Coef"]})')
print(f'2nd predictor: {std_coef_df.iloc[1]["Feature"]} (coef={std_coef_df.iloc[1]["Std_Coef"]})')

# %% [markdown]
# **Observation:** Standardized coefficients rank predictors by impact: 1. Alcohol (+0.3340), 2. Volatile acidity (-0.1785), 3. Sulphates (+0.1481), and 4. Total sulfur dioxide (-0.1226). Fixed acidity has a modest standardized coefficient (-0.0483) despite moderate correlation with quality, because its variance is shared with density and pH through multicollinearity.
# 

# %%
# Statsmodels OLS for p-values
X_train_sm = sm.add_constant(X_train)
ols_model = sm.OLS(y_train, X_train_sm).fit()
ols_summary_df = pd.DataFrame({
    'Feature': ['const'] + feature_cols,
    'Coef': ols_model.params.round(4),
    'Std_Err': ols_model.bse.round(4),
    'p_value': ols_model.pvalues.round(4),
    'Significant_0.05': (ols_model.pvalues < 0.05)
})
print('=== OLS Coefficient p-values ===')
print(ols_summary_df.to_string(index=False))
ols_summary_df.to_csv('results/task3_ols_pvalues.csv', index=False)

# %% [markdown]
# **Observation:** Statsmodels OLS confirms that volatile acidity (p < 0.0001), chlorides (p = 0.0076), total sulfur dioxide (p = 0.0002), sulphates (p < 0.0001), and alcohol (p < 0.0001) are statistically significant predictors at alpha = 0.05. Fixed acidity, residual sugar, and density are not statistically significant in the multiple regression model.

# %% [markdown]
# **Purpose:** Scatter plot of predicted vs actual quality on the test set with the y=x line. A small jitter is added to the actual quality scores (which are discrete integers) to avoid overplotting. Also a residual plot to check linearity and homoscedasticity assumptions.

# %%
np.random.seed(42)
jitter = np.random.uniform(-0.2, 0.2, len(y_test))
y_test_jittered = y_test + jitter

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8))

# Predicted vs Actual
ax1.scatter(y_test_jittered, y_pred_test, alpha=0.45, s=18, color='#2b5c8f', edgecolors='none')
mn = min(y_test.min(), y_pred_test.min()) - 0.2
mx = max(y_test.max(), y_pred_test.max()) + 0.2
ax1.plot([mn, mx], [mn, mx], 'r--', linewidth=1.8, label='Ideal y = x (Perfect Prediction)')
ax1.axhline(y_train.mean(), color='grey', linestyle=':', label=f'Global Mean ({y_train.mean():.2f})')
ax1.set_title('Predicted vs Actual Quality Scores (Test Set)\n(Actual scores jittered ±0.2 for density visualization)', fontsize=11, fontweight='bold')
ax1.set_xlabel('Actual Sensory Quality Rating (Jittered)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Predicted Quality Score (Continuous)', fontsize=10, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc='upper left', fontsize=8.5)

# Metrics box
metrics_box = f"Test R² = {r2_test:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}"
ax1.text(0.95, 0.05, metrics_box, transform=ax1.transAxes, fontsize=9.5, fontweight='bold',
         verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc', edgecolor='#cbd5e1'))

# Residuals Plot
residuals = y_test - y_pred_test
ax2.scatter(y_pred_test, residuals, alpha=0.45, s=18, color='#e26d24', edgecolors='none')
ax2.axhline(0, color='black', linewidth=1.2)
ax2.axhline(rmse, color='red', linestyle='--', alpha=0.6, label=f'+1 RMSE (+{rmse:.2f})')
ax2.axhline(-rmse, color='red', linestyle='--', alpha=0.6, label=f'-1 RMSE (-{rmse:.2f})')
ax2.set_title('Model Residual Diagnostics (Test Set)\n(Parallel lines reflect discrete integer quality tiers: e = k - ŷ)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Predicted Quality Score (ŷ)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Residual (Actual y - Predicted ŷ)', fontsize=10, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc='upper right', fontsize=8.5)

plt.suptitle('Multiple Linear Regression Diagnostic Evaluation (Task 3e)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/05_predicted_vs_actual_residuals.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: figures/05_predicted_vs_actual_residuals.png')

# %% [markdown]
# **Observation:** The predicted vs actual plot shows predictions concentrated narrowly between 5.0 and 6.5. True extreme scores (3, 4, 8) are pulled toward the training mean (5.63), demonstrating standard regression attenuation on ordinal targets. The residual plot exhibits distinct parallel downward lines corresponding to each discrete score integer level (e = k - ŷ), proving that residuals are constrained by target discreteness rather than pure random gaussian noise.
# 

# %% [markdown]
# **Purpose:** Save all key metrics and the full results to JSON and CSV for the PDF report builder.

# %%
all_metrics = {
    'n_rows_total': len(pd.read_csv('data/winequality-red.csv', sep=';')),
    'n_rows_after_dedup': int(len(df)),
    'n_duplicates_removed': int(pd.read_csv('data/winequality-red.csv', sep=';').duplicated().sum()),
    'train_size': int(X_train.shape[0]),
    'test_size': int(X_test.shape[0]),
    'pearson_alcohol_quality': {'r': round(float(r), 4), 'p_value': float(p_val)},
    'regression': {
        'mae': round(float(mae), 4),
        'mse': round(float(mse), 4),
        'rmse': round(float(rmse), 4),
        'r2_test': round(float(r2_test), 4),
        'r2_train': round(float(r2_train), 4)
    },
    'top_predictors_by_std_coef': std_coef_df[['Feature', 'Std_Coef']].head(3).to_dict('records'),
    'top_corr_with_quality': rank_df[['Attribute', 'Pearson_r']].head(3).to_dict('records')
}
with open('results/metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print('Saved: results/metrics.json')
print(json.dumps(all_metrics, indent=2))

# %% [markdown]
# **Observation:** All regression metrics, correlation rankings, VIF tables, and p-values were saved into results/ and figures/.

# %% [markdown]
# ## Task 4: Business Decision-Making

# %%
print('=== Task 4a: Recommended Lab Measurements ===')
print()
top_corr = rank_df.iloc[0]['Attribute']
top_coef = std_coef_df.iloc[0]['Feature']
print(f'Based on correlation with quality, standardized coefficients, and OLS p-values:')
print(f'  1. {std_coef_df.iloc[0]["Feature"]} (std coef={std_coef_df.iloc[0]["Std_Coef"]}, '
       f'Pearson r with quality={rank_df[rank_df["Attribute"]==std_coef_df.iloc[0]["Feature"]]["Pearson_r"].values[0]:.4f})')
print(f'  2. {std_coef_df.iloc[1]["Feature"]} (std coef={std_coef_df.iloc[1]["Std_Coef"]}, '
       f'Pearson r with quality={rank_df[rank_df["Attribute"]==std_coef_df.iloc[1]["Feature"]]["Pearson_r"].values[0]:.4f})')
print()
print('=== Task 4b: Is the Model Reliable Enough to Replace Sensory Evaluation? ===')
print(f'Test R2: {r2_test:.4f}, RMSE: {rmse:.4f} quality points')
print('Quality is scored on a 0-10 scale. An RMSE of ~0.6 means predictions are off by about half a point on average.')
print('With R2 < 0.4 typically, the model explains a limited fraction of variance.')
print('Recommendation: Use as a screening aid to flag potentially low/high quality wines for expert panel review,')
print('not as a replacement for sensory evaluation.')
print()
print('=== Task 4c: Two Limitations ===')
print('1. The dataset covers only red Vinho Verde wine from a specific region; results may not generalise')
print('   to other wine types, regions, or vintages.')
print('2. Quality is a subjective median of 3+ expert scores, introducing label noise and class imbalance')
print('   (scores 5-6 dominate; scores 3, 4, 8 are rare). This limits model performance at extremes.')

# %% [markdown]
# **Observation:** Recommendation: 1. Alcohol (std coef +0.334, p < 0.0001) and 2. Volatile acidity (std coef -0.1785, p < 0.0001) should be monitored for winery lab screening. The regression model (RMSE = 0.66, R2 = 0.35) should serve as a preliminary quality screening tool rather than a full replacement for expert human sensory panels.

# %% [markdown]
# ## Conclusion and Limitations
# 
# 
# 
# This case study applied statistical analysis and multiple linear regression to the UCI red wine quality dataset. Alcohol content showed the strongest positive correlation with quality and was the most influential predictor by standardized coefficient. Volatile acidity showed a significant negative relationship. The regression model achieved moderate R2 on the test set and an RMSE of approximately 0.6 quality points. Multicollinearity was found among several features (fixed acidity, density, citric acid, pH), which limits the interpretability of individual coefficients but does not affect prediction accuracy.
# 
# 
# 
# **Limitations:**
# 
# 
# 
# 1. Only red Vinho Verde wines are included; the model does not generalise to other wine types.
# 
# 2. Quality is a subjective ordinal score (median of expert ratings), which introduces label noise.
# 
# 3. Class imbalance (scores 5 and 6 dominate) causes poor prediction at extreme quality levels (3, 4, 8).
# 
# 4. A linear model cannot capture non-linear relationships between physicochemical properties and quality.
# 
# 5. No grape variety, brand, price, or vintage data are available, limiting the model's explanatory scope.

# %% [markdown]
# ## References
# 
# 
# 
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.
# 
# 
# 
# Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, scipy.
# 
# 
# 
# GitHub repository: <GITHUB URL>
# 
# 
# 
# ## Attribution
# 
# 
# 
# Code structure drafted with AI assistance (Claude); executed, verified and interpreted by <NAME>. Dataset source: Cortez et al. (2009), UCI Machine Learning Repository.

