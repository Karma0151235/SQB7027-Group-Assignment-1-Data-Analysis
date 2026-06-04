"""
OFD Usage, Dietary Patterns & Healthy Food Perception Among Adults in Selangor
Statistical Analysis Script
Consultation for: Nur Liyana Binti Azman, Faculty of Health Science, UiTM
Prepared by: Farzana Akter, Nur Alya Izzati Binti Azman, Stephen Tee Wei Xin
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, chi2_contingency, kendalltau, f_oneway, mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})

# ==============================================================================
# CONFIGURATION
# ==============================================================================
script_dir  = os.path.dirname(os.path.abspath(__file__))
file_path   = os.path.join(script_dir, 'RespondentData.xlsx')
plot_dir    = os.path.join(script_dir, 'Analysis_Plots')
os.makedirs(plot_dir, exist_ok=True)

# ==============================================================================
# PHASE 0: DATA INGESTION
# ==============================================================================
print("=" * 65)
print("PHASE 0: DATA INGESTION")
print("=" * 65)
print(f"Loading data from: {file_path}")
df_raw = pd.read_excel(file_path)
print(f"Raw shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

# ==============================================================================
# PHASE 1: COLUMN-SHIFT-AWARE DATA CLEANING
# ==============================================================================
print("\n" + "=" * 65)
print("PHASE 1: DATA CLEANING & COLUMN SHIFT CORRECTION")
print("=" * 65)
"""
Root cause: Google Forms inserted a Selangor-residency confirmation
question (Q3) for 60 respondents, shifting all subsequent answers one
column to the right relative to the column headers.  We detect these
rows by the presence of 'Ya / Yes' (the consent answer) in column 1
(the Gender header column) and apply row-type-specific column offsets.
"""

shifted_mask = df_raw.iloc[:, 1].astype(str).str.strip() == 'Ya / Yes'
n_shifted    = shifted_mask.sum()
n_unshifted  = (~shifted_mask).sum()
print(f"  Standard rows  : {n_unshifted}")
print(f"  Shifted rows   : {n_shifted}  (Selangor residency Q inserted)")


def extract_col(df, mask, unshifted_idx, shifted_idx):
    """Concatenate a variable from the two row populations."""
    return pd.concat([
        df[~mask].iloc[:, unshifted_idx],
        df[mask].iloc[:, shifted_idx],
    ]).reset_index(drop=True)


gender       = extract_col(df_raw, shifted_mask, 1, 2)
age          = extract_col(df_raw, shifted_mask, 2, 3)
district     = extract_col(df_raw, shifted_mask, 3, 4)
ethnicity    = extract_col(df_raw, shifted_mask, 4, 5)
employment   = extract_col(df_raw, shifted_mask, 5, 6)
income       = extract_col(df_raw, shifted_mask, 6, 7)
app_used     = extract_col(df_raw, shifted_mask, 7, 8)
usage_freq   = extract_col(df_raw, shifted_mask, 8, 9)
self_percept = extract_col(df_raw, shifted_mask, 9, 10)

# Perception Likert items sit at fixed absolute columns for ALL rows
perc_q1 = pd.to_numeric(df_raw.iloc[:, 60], errors='coerce').reset_index(drop=True)
perc_q2 = pd.to_numeric(df_raw.iloc[:, 61], errors='coerce').reset_index(drop=True)
perc_q3 = pd.to_numeric(df_raw.iloc[:, 62], errors='coerce').reset_index(drop=True)

# ==============================================================================
# PHASE 2: FFQ SCORING (Ghazali & Isa 2023 methodology)
# ==============================================================================
print("\n" + "=" * 65)
print("PHASE 2: FFQ DIETARY SCORE COMPUTATION")
print("=" * 65)
"""
Method: Convert categorical frequency responses to daily frequency
conversion factors (Ghazali & Isa, 2023).  Score = Σ(daily_factor)
summed across all 38 food items.  Portion-size columns are excluded
because they contain free-text values that are too heterogeneous for
reliable quantification.  The resulting score is a continuous index of
overall dietary frequency load — higher values indicate more frequent
consumption across more food groups.
"""

# FFQ food-item column indices for standard rows (excludes portion-size cols)
FFQ_IDX_UNSHIFTED = [
    11, 12, 13, 14, 15, 16, 17, 18,  # Cereals & cereal products (8 items)
    20, 21, 22, 23, 24,               # Fast foods (5 items)
    26, 27,                           # Meat & meat products (2 items)
    29, 30, 31, 32,                   # Fish & seafood (4 items)
    34,                               # Eggs (1 item)
    36,                               # Legumes (1 item)
    38,                               # Dairy (1 item)
    40, 41,                           # Vegetables (2 items)
    43, 44, 45, 46, 47,               # Beverages (5 items)
    49, 50, 51, 52, 53, 54,           # Confectionery (6 items)
    56, 57, 58,                       # Condiments & flavourings (3 items)
]  # 38 items total
FFQ_IDX_SHIFTED = [i + 1 for i in FFQ_IDX_UNSHIFTED]

# Frequency string → times per day conversion factors
FREQ_MAP = {
    'tidak pernah': 0.0,
    'never':        0.0,
    '1-3 kali sebulan':   2  / 30,   # mid-point 2 of 1-3
    '1-3 times per month': 2 / 30,
    'sekali seminggu':    1  / 7,
    'once a week':        1  / 7,
    '2-4 kali seminggu':  3  / 7,    # mid-point 3 of 2-4
    '2-4 times per week': 3  / 7,
    '5-6 kali seminggu':  5.5 / 7,   # mid-point 5.5 of 5-6
    '5-6 times per week': 5.5 / 7,
    'sekali sehari':      1.0,
    'once a day':         1.0,
    '2-3 kali sehari':    2.5,        # mid-point 2.5 of 2-3
    '2-3 times per day':  2.5,
    '>=4 kali sehari':    4.0,
    '4 kali sehari':      4.0,
    '5-6 kali sehari':    5.5,
    '5-6 times per day':  5.5,
}


def parse_freq(cell):
    s = str(cell).lower().strip()
    for key, val in FREQ_MAP.items():
        if key in s:
            return val
    return np.nan


def score_ffq(frame):
    return frame.apply(lambda col: col.map(parse_freq)).sum(axis=1, skipna=True)


unshifted_dietary = score_ffq(df_raw[~shifted_mask].iloc[:, FFQ_IDX_UNSHIFTED])
shifted_dietary   = score_ffq(df_raw[shifted_mask].iloc[:,   FFQ_IDX_SHIFTED])
dietary_score     = pd.concat([unshifted_dietary, shifted_dietary]).reset_index(drop=True)

print(f"  Items in FFQ   : {len(FFQ_IDX_UNSHIFTED)}")
print(f"  Score range    : {dietary_score.min():.2f} – {dietary_score.max():.2f}")
print(f"  Mean ± SD      : {dietary_score.mean():.2f} ± {dietary_score.std():.2f}")

# ==============================================================================
# PHASE 3: ENCODE ANALYTICAL VARIABLES
# ==============================================================================
print("\n" + "=" * 65)
print("PHASE 3: VARIABLE ENCODING")
print("=" * 65)

# OFD usage frequency → ordinal (1–5)
USAGE_ORDINAL_MAP = {
    '1 – 2 kali sebulan / times monthly': 1,
    '3 – 4 kali sebulan / times monthly': 2,
    '5 – 6 kali sebulan / times monthly': 3,
    '7 – 8 kali sebulan / times monthly': 4,
    '9 kali sebulan / times monthly':     5,
}
usage_ordinal = usage_freq.map(USAGE_ORDINAL_MAP)
print(f"  usage_ordinal range : {usage_ordinal.min()} – {usage_ordinal.max()}")
print(f"  Null usage_ordinal  : {usage_ordinal.isna().sum()}")

# Perception composite score (sum of 3 Likert items, range 3–15)
# Replace out-of-range values (0) with NaN
perc_q1 = perc_q1.where(perc_q1.between(1, 5))
perc_q2 = perc_q2.where(perc_q2.between(1, 5))
perc_q3 = perc_q3.where(perc_q3.between(1, 5))
perception_score = perc_q1 + perc_q2 + perc_q3   # continuous, 3–15
print(f"  perception_score range : {perception_score.min():.0f} – {perception_score.max():.0f}")
print(f"  perception_score mean  : {perception_score.mean():.2f}")

# Categorical cut for chi-square: Low (3–9) / High (10–15)
perception_cat = pd.cut(perception_score, bins=[2, 9, 15],
                        labels=['Low', 'High'])
print(f"  Perception Low/High    : {perception_cat.value_counts().to_dict()}")

# Age → ordinal bracket
AGE_ORDINAL_MAP = {
    '30-34 tahun / years old':   1,
    '35-39 tahun / years old':   2,
    '40-44 tahun / years old':   3,
    '45-49 tahun / years old':   4,
    '50-54 tahun / years old':   5,
    '55 – 59 tahun / years old': 6,
}
age_ordinal = age.map(AGE_ORDINAL_MAP)

# Income → ordinal bracket
INCOME_ORDINAL_MAP = {
    '≤RM5,000':           1,
    'RM5,000 - RM11,000': 2,
    'RM11,000 - RM14,000':3,
    '≥RM15,000':          4,
}
income_ordinal = income.map(INCOME_ORDINAL_MAP)

# Binary: Grab vs FoodPanda (for ecosystem comparison)
app_binary = app_used.map(lambda x: 'Grab Food'  if 'Grab'  in str(x)
                          else ('Food Panda' if 'Panda' in str(x) else np.nan))

# ==============================================================================
# PHASE 4: ASSEMBLE ANALYTICAL DATAFRAME
# ==============================================================================
cdf = pd.DataFrame({
    'gender':          gender.values,
    'age':             age.values,
    'age_ordinal':     age_ordinal.values,
    'district':        district.values,
    'ethnicity':       ethnicity.values,
    'employment':      employment.values,
    'income':          income.values,
    'income_ordinal':  income_ordinal.values,
    'app_used':        app_used.values,
    'app_binary':      app_binary.values,
    'usage_freq':      usage_freq.values,
    'usage_ordinal':   usage_ordinal.values,
    'self_percept':    self_percept.values,
    'perc_q1':         perc_q1.values,
    'perc_q2':         perc_q2.values,
    'perc_q3':         perc_q3.values,
    'perception_score':perception_score.values,
    'perception_cat':  perception_cat.values,
    'dietary_score':   dietary_score.values,
})

# Empirical diet classification by median split
median_diet = cdf['dietary_score'].median()
cdf['diet_class'] = np.where(cdf['dietary_score'] > median_diet,
                              'Higher Intake', 'Lower Intake')

print(f"\n  Final analytical sample: n = {len(cdf)}")

# ==============================================================================
# PHASE 5: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
print("\n" + "=" * 65)
print("PHASE 5: EXPLORATORY DATA ANALYSIS")
print("=" * 65)

PALETTE = sns.color_palette("muted")

# --- Plot 1: Age Distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
age_order = [k for k in AGE_ORDINAL_MAP.keys() if k in cdf['age'].values]
age_counts = cdf['age'].value_counts().reindex(age_order)
bars = ax.barh(age_order, age_counts.values, color=PALETTE[0], edgecolor='white', height=0.6)
for bar, val in zip(bars, age_counts.values):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'n={val}', va='center', fontsize=10)
ax.set_xlabel('Number of Respondents')
ax.set_title('Age Distribution of Respondents (n=300)', fontweight='bold', pad=12)
ax.set_xlim(0, age_counts.max() * 1.15)
ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '01_Age_Distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 01] Age distribution saved")

# --- Plot 2: Market Share Pie ---
fig, ax = plt.subplots(figsize=(7, 7))
app_counts = cdf['app_used'].value_counts()
colors_pie = sns.color_palette("pastel", len(app_counts))
wedges, texts, autotexts = ax.pie(
    app_counts.values, labels=app_counts.index,
    autopct='%1.1f%%', startangle=140, colors=colors_pie,
    pctdistance=0.82, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
for at in autotexts:
    at.set_fontsize(10)
ax.set_title('Most Frequently Used OFD Application\n(n=300)', fontweight='bold', pad=16)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '02_App_Market_Share.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 02] Market share pie saved")

# --- Plot 3: Usage Frequency Distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
usage_labels = [
    '1–2 times/month', '3–4 times/month', '5–6 times/month',
    '7–8 times/month', '≥9 times/month'
]
usage_counts = cdf['usage_ordinal'].value_counts().sort_index()
bars = ax.bar(usage_labels, usage_counts.values, color=PALETTE[1], edgecolor='white', width=0.6)
for bar, val in zip(bars, usage_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'n={val}', ha='center', fontsize=10)
ax.set_ylabel('Number of Respondents')
ax.set_xlabel('OFD Usage Frequency')
ax.set_title('OFD Usage Frequency Among Adults in Selangor (n=300)', fontweight='bold', pad=12)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '03_Usage_Frequency_Distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 03] Usage frequency distribution saved")

# --- Plot 4: Income Distribution ---
fig, ax = plt.subplots(figsize=(8, 5))
income_order = ['≤RM5,000', 'RM5,000 - RM11,000', 'RM11,000 - RM14,000', '≥RM15,000']
income_counts = cdf['income'].value_counts().reindex(income_order, fill_value=0)
bars = ax.bar(income_order, income_counts.values, color=PALETTE[2], edgecolor='white', width=0.6)
for bar, val in zip(bars, income_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'n={val}', ha='center', fontsize=10)
ax.set_ylabel('Number of Respondents')
ax.set_xlabel('Monthly Income / Allowance')
ax.set_title('Income Distribution of Respondents (n=300)', fontweight='bold', pad=12)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '04_Income_Distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 04] Income distribution saved")

# --- Plot 5: Perception Score Distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
valid_perc = cdf['perception_score'].dropna()
ax.hist(valid_perc, bins=15, color=PALETTE[3], edgecolor='white', alpha=0.85)
ax.axvline(valid_perc.mean(), color='crimson', linestyle='--', linewidth=1.8,
           label=f'Mean = {valid_perc.mean():.1f}')
ax.axvline(valid_perc.median(), color='navy', linestyle=':', linewidth=1.8,
           label=f'Median = {valid_perc.median():.1f}')
ax.set_xlabel('Composite Perception Score (3 = most negative, 15 = most positive)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Healthy Food Perception Score\n(Sum of 3 Likert items; n=300)', fontweight='bold', pad=12)
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '05_Perception_Score_Distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 05] Perception score distribution saved")

# --- Plot 6: Dietary Score Distribution ---
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(cdf['dietary_score'], bins=25, kde=True, ax=ax,
             color=PALETTE[4], edgecolor='white', alpha=0.75)
ax.axvline(cdf['dietary_score'].mean(), color='crimson', linestyle='--', linewidth=1.8,
           label=f'Mean = {cdf["dietary_score"].mean():.2f}')
ax.set_xlabel('Dietary Score (Sum of Daily Frequency Factors, 38 food items)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Computed Dietary Score (n=300)', fontweight='bold', pad=12)
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '06_Dietary_Score_Distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 06] Dietary score distribution saved")

# --- Plot 7: Gender x Usage Frequency stacked bar ---
fig, ax = plt.subplots(figsize=(9, 5))
cross_gender_usage = pd.crosstab(cdf['usage_ordinal'], cdf['gender'], normalize='index') * 100
cross_gender_usage.index = usage_labels
cross_gender_usage.plot(kind='bar', stacked=True, ax=ax,
                        color=[PALETTE[0], PALETTE[3]], edgecolor='white', width=0.6)
ax.set_xlabel('OFD Usage Frequency')
ax.set_ylabel('Proportion (%)')
ax.set_title('OFD Usage Frequency by Gender (Percentage, n=300)', fontweight='bold', pad=12)
ax.legend(title='Gender')
ax.grid(axis='y', alpha=0.4)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '07_Gender_by_Usage.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 07] Gender by usage frequency saved")

# ==============================================================================
# PHASE 6: BASELINE ANALYSIS (CLIENT'S PROPOSED METHODOLOGY)
# ==============================================================================
print("\n" + "=" * 65)
print("PHASE 6: BASELINE ANALYSIS — CLIENT'S PROPOSED METHODOLOGY")
print("=" * 65)

# -- Objective 4: Spearman's correlation (usage_freq × dietary_score) --
valid_obj4 = cdf[['usage_ordinal', 'dietary_score']].dropna()
rho, p_spearman = spearmanr(valid_obj4['usage_ordinal'], valid_obj4['dietary_score'])
print(f"\n  [Objective 4] Spearman Rank Correlation")
print(f"    n          = {len(valid_obj4)}")
print(f"    rho        = {rho:.4f}")
print(f"    p-value    = {p_spearman:.4f}")
print(f"    Significant (p<0.05): {'YES' if p_spearman < 0.05 else 'NO'}")

# -- Objective 5: Pearson Chi-Square (usage_freq × perception category) --
valid_obj5 = cdf[['usage_ordinal', 'perception_cat']].dropna()
contingency_table = pd.crosstab(valid_obj5['usage_ordinal'], valid_obj5['perception_cat'])
contingency_table.index = [usage_labels[i-1] for i in contingency_table.index]
print(f"\n  [Objective 5] Pearson Chi-Square Test")
print(f"    Contingency table (usage_freq × perception_cat):")
print("    " + contingency_table.to_string().replace('\n', '\n    '))
chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)
print(f"    chi2       = {chi2:.4f}")
print(f"    dof        = {dof}")
print(f"    p-value    = {p_chi2:.4f}")
print(f"    Significant (p<0.05): {'YES' if p_chi2 < 0.05 else 'NO'}")

# Plot: Baseline scatter (Obj 4)
fig, ax = plt.subplots(figsize=(9, 6))
sns.stripplot(x='usage_ordinal', y='dietary_score', data=cdf,
              jitter=True, alpha=0.5, palette='muted', ax=ax)
means = cdf.groupby('usage_ordinal')['dietary_score'].mean()
ax.plot(range(len(means)), means.values, 'r-o', linewidth=2,
        markersize=7, label='Group mean', zorder=5)
ax.set_xticks(range(5))
ax.set_xticklabels(usage_labels, rotation=20, ha='right')
ax.set_xlabel('OFD Usage Frequency')
ax.set_ylabel('Dietary Score (Daily Frequency Factor Sum)')
ax.set_title(
    f'Dietary Score vs OFD Usage Frequency\n'
    f'Spearman ρ = {rho:.3f}, p = {p_spearman:.3f}',
    fontweight='bold', pad=12)
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '08_Obj4_Spearman_Scatter.png'), dpi=200, bbox_inches='tight')
plt.close()
print("\n  [Plot 08] Spearman scatter (Obj 4) saved")

# Plot: Baseline Chi-Square heatmap (Obj 5)
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='Blues',
            linewidths=0.5, ax=ax, cbar=True, annot_kws={'size': 12})
ax.set_title(
    f'OFD Usage Frequency × Perception Category\n'
    f'Chi-Square = {chi2:.3f}, df = {dof}, p = {p_chi2:.3f}',
    fontweight='bold', pad=12)
ax.set_xlabel('Perception of Healthy Food Availability')
ax.set_ylabel('OFD Usage Frequency')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '09_Obj5_ChiSquare_Heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 09] Chi-square heatmap (Obj 5) saved")

# ==============================================================================
# PHASE 7: IMPROVED METHODOLOGY
# ==============================================================================
print("\n" + "=" * 65)
print("PHASE 7: IMPROVED METHODOLOGY")
print("=" * 65)

# -- 7A: ANCOVA (OLS regression with covariates, replaces Spearman for Obj 4) --
print("\n  [7A] ANCOVA — OLS Regression (Obj 4 Upgrade)")
print("       DV: dietary_score | IV: usage_ordinal | Covariates: age_ordinal, income_ordinal")

ancova_data = cdf[['dietary_score', 'usage_ordinal', 'age_ordinal', 'income_ordinal']].dropna()
X = ancova_data[['usage_ordinal', 'age_ordinal', 'income_ordinal']].values
y = ancova_data['dietary_score'].values

# Centre X for interpretability
X_mean = X.mean(axis=0)
X_c    = X - X_mean

# OLS via numpy
X_aug = np.column_stack([np.ones(len(X_c)), X_c])
beta  = np.linalg.lstsq(X_aug, y, rcond=None)[0]

# Compute residuals and t-statistics
y_hat   = X_aug @ beta
resid   = y - y_hat
n, k    = len(y), X_aug.shape[1]
MSE     = (resid ** 2).sum() / (n - k)
var_b   = MSE * np.linalg.inv(X_aug.T @ X_aug).diagonal()
se_b    = np.sqrt(var_b)
t_stats = beta / se_b
p_vals  = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))
SS_tot  = ((y - y.mean()) ** 2).sum()
SS_res  = (resid ** 2).sum()
R2      = 1 - SS_res / SS_tot
adj_R2  = 1 - (1 - R2) * (n - 1) / (n - k)

coef_labels = ['Intercept', 'Usage Frequency', 'Age Bracket', 'Income Bracket']
print(f"\n    n={len(ancova_data)}, R²={R2:.4f}, Adj. R²={adj_R2:.4f}")
print(f"\n    {'Variable':<22} {'Coeff':>8} {'SE':>8} {'t':>8} {'p':>8}")
print("    " + "-" * 56)
for lab, b, se, t, p in zip(coef_labels, beta, se_b, t_stats, p_vals):
    sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    print(f"    {lab:<22} {b:>8.3f} {se:>8.3f} {t:>8.3f} {p:>8.4f} {sig}")

# -- 7B: Kendall's Tau-b (replaces Chi-Square for Obj 5) --
print("\n  [7B] Kendall's Tau-b — Ordinal-Ordinal Association (Obj 5 Upgrade)")
print("       IV: usage_ordinal | DV: perception_score (continuous)")

valid_tau = cdf[['usage_ordinal', 'perception_score']].dropna()
tau, p_tau = kendalltau(valid_tau['usage_ordinal'], valid_tau['perception_score'])
print(f"    n         = {len(valid_tau)}")
print(f"    tau-b     = {tau:.4f}")
print(f"    p-value   = {p_tau:.4f}")
print(f"    Significant (p<0.05): {'YES' if p_tau < 0.05 else 'NO'}")

# Plot ANCOVA: usage vs dietary by age group
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Left: usage × dietary coloured by age
scatter_data = cdf.dropna(subset=['usage_ordinal', 'dietary_score', 'age_ordinal'])
age_labels_short = {1:'30-34', 2:'35-39', 3:'40-44', 4:'45-49', 5:'50-54', 6:'55-59'}
for age_val, grp in scatter_data.groupby('age_ordinal'):
    axes[0].scatter(grp['usage_ordinal'] + np.random.uniform(-0.15, 0.15, len(grp)),
                    grp['dietary_score'], alpha=0.5, s=25,
                    label=age_labels_short.get(age_val, str(age_val)))
axes[0].set_xticks(range(1, 6))
axes[0].set_xticklabels(usage_labels, rotation=20, ha='right', fontsize=8)
axes[0].set_xlabel('OFD Usage Frequency')
axes[0].set_ylabel('Dietary Score')
axes[0].set_title('Usage × Dietary Score\n(coloured by Age Group)', fontweight='bold', pad=10)
axes[0].legend(title='Age bracket', fontsize=8, title_fontsize=8)
axes[0].grid(alpha=0.3)

# Right: coefficient plot
coef_no_int = beta[1:]
se_no_int   = se_b[1:]
p_no_int    = p_vals[1:]
labels_plot = ['Usage\nFrequency', 'Age\nBracket', 'Income\nBracket']
colors_coef = ['steelblue' if p < 0.05 else 'lightgrey' for p in p_no_int]
axes[1].barh(labels_plot, coef_no_int, xerr=se_no_int * 1.96, color=colors_coef,
             edgecolor='white', height=0.5)
axes[1].axvline(0, color='black', linewidth=1.2, linestyle='--')
axes[1].set_xlabel('OLS Coefficient (Δ Dietary Score per bracket)')
axes[1].set_title('ANCOVA Coefficients\n(Blue = significant at p<0.05)', fontweight='bold', pad=10)
axes[1].grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '10_ANCOVA_Results.png'), dpi=200, bbox_inches='tight')
plt.close()
print("\n  [Plot 10] ANCOVA visualisation saved")

# Plot: perception score vs usage (Kendall's)
fig, ax = plt.subplots(figsize=(9, 6))
for usage_val, grp in cdf.dropna(subset=['usage_ordinal', 'perception_score']).groupby('usage_ordinal'):
    ax.scatter([usage_val] * len(grp) + np.random.uniform(-0.15, 0.15, len(grp)),
               grp['perception_score'], alpha=0.4, s=25, color=PALETTE[usage_val - 1])
means_p = cdf.groupby('usage_ordinal')['perception_score'].mean()
ax.plot(means_p.index, means_p.values, 'k-o', linewidth=2, markersize=7,
        label='Group mean', zorder=5)
ax.set_xticks(range(1, 6))
ax.set_xticklabels(usage_labels, rotation=20, ha='right')
ax.set_xlabel('OFD Usage Frequency')
ax.set_ylabel('Composite Perception Score (3–15)')
ax.set_title(
    f'Perception Score vs OFD Usage Frequency\n'
    f"Kendall's tau-b = {tau:.3f}, p = {p_tau:.3f}",
    fontweight='bold', pad=12)
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '11_Obj5_Kendall_Scatter.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 11] Kendall Tau-b scatter (Obj 5) saved")

# ==============================================================================
# PHASE 8: DEPTH-OF-RESEARCH EXPLORATIONS
# ==============================================================================
print("\n" + "=" * 65)
print("PHASE 8: DEPTH-OF-RESEARCH EXPLORATIONS")
print("=" * 65)

# -- 8A: One-Way ANOVA — Income × Dietary Score --
print("\n  [8A] One-Way ANOVA: Income Bracket vs Dietary Score")
income_groups = [
    grp['dietary_score'].dropna().values
    for _, grp in cdf.dropna(subset=['income_ordinal']).groupby('income_ordinal')
]
F_inc, p_inc = f_oneway(*income_groups)
print(f"    F-statistic = {F_inc:.4f}")
print(f"    p-value     = {p_inc:.4f}")
print(f"    Significant (p<0.05): {'YES' if p_inc < 0.05 else 'NO'}")

# -- 8B: Mann-Whitney U — GrabFood vs FoodPanda dietary scores --
print("\n  [8B] Mann-Whitney U Test: GrabFood vs FoodPanda Dietary Score")
grab_scores  = cdf[cdf['app_binary'] == 'Grab Food']['dietary_score'].dropna()
panda_scores = cdf[cdf['app_binary'] == 'Food Panda']['dietary_score'].dropna()
U_stat, p_mwu = mannwhitneyu(grab_scores, panda_scores, alternative='two-sided')
print(f"    GrabFood  : n={len(grab_scores)}, median={grab_scores.median():.2f}")
print(f"    FoodPanda : n={len(panda_scores)}, median={panda_scores.median():.2f}")
print(f"    U stat    = {U_stat:.1f}")
print(f"    p-value   = {p_mwu:.4f}")
print(f"    Significant (p<0.05): {'YES' if p_mwu < 0.05 else 'NO'}")

# -- 8C: Cognitive Dissonance Cross-Tabulation --
print("\n  [8C] Cognitive Dissonance: Self-Reported vs Empirical Dietary Class")
# 'Healthy' = does NOT contain 'tidak'; 'Unhealthy' = contains 'tidak'
valid_dissonance = cdf.dropna(subset=['self_percept']).copy()
valid_dissonance = valid_dissonance[
    valid_dissonance['self_percept'].astype(str).str.contains('Makanan', na=False)]
valid_dissonance['self_percept_en'] = np.where(
    ~valid_dissonance['self_percept'].astype(str).str.contains('tidak', case=False, na=False),
    'Self-reports:\nHealthy', 'Self-reports:\nUnhealthy')
dissonance_matrix = pd.crosstab(
    valid_dissonance['self_percept_en'],
    valid_dissonance['diet_class'])
print(f"\n    {dissonance_matrix.to_string()}")

# Compute dissonance rate among those who claim 'Healthy'
healthy_claimers = valid_dissonance[
    ~valid_dissonance['self_percept'].astype(str).str.contains('tidak', na=False)]
dissonant_n = (healthy_claimers['diet_class'] == 'Higher Intake').sum()
pct_dissonant = dissonant_n / len(healthy_claimers) * 100
print(f"\n    Respondents claiming 'Healthy': n={len(healthy_claimers)}")
print(f"    ...with Higher Intake score  : n={dissonant_n} ({pct_dissonant:.1f}%) — cognitive dissonance")

# -- 8D: Kruskal-Wallis across all usage groups (non-parametric ANOVA alternative) --
print("\n  [8D] Kruskal-Wallis Test: Usage Frequency vs Dietary Score")
kw_groups = [grp['dietary_score'].dropna().values
             for _, grp in cdf.groupby('usage_ordinal')]
H_stat, p_kw = stats.kruskal(*kw_groups)
print(f"    H-statistic = {H_stat:.4f}")
print(f"    p-value     = {p_kw:.4f}")
print(f"    Significant (p<0.05): {'YES' if p_kw < 0.05 else 'NO'}")

# ==============================================================================
# PHASE 9: VISUALISATIONS — DEPTH EXPLORATIONS
# ==============================================================================

# Plot: Boxplot — Income vs Dietary Score
fig, ax = plt.subplots(figsize=(9, 6))
income_order_labels = ['≤RM5,000', 'RM5,000 - RM11,000', 'RM11,000 - RM14,000', '≥RM15,000']
plot_data = cdf[cdf['income'].isin(income_order_labels)].copy()
sns.boxplot(x='income', y='dietary_score', data=plot_data, order=income_order_labels,
            palette='Set2', ax=ax, width=0.55)
sns.stripplot(x='income', y='dietary_score', data=plot_data, order=income_order_labels,
              alpha=0.35, jitter=True, color='black', size=3, ax=ax)
ax.set_xlabel('Monthly Income / Allowance')
ax.set_ylabel('Dietary Score')
ax.set_title(
    f'Dietary Score Distribution Across Income Brackets\n'
    f'One-Way ANOVA: F = {F_inc:.3f}, p = {p_inc:.3f}',
    fontweight='bold', pad=12)
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '12_Income_vs_Dietary.png'), dpi=200, bbox_inches='tight')
plt.close()
print("\n  [Plot 12] Income vs dietary boxplot saved")

# Plot: KDE — GrabFood vs FoodPanda
fig, ax = plt.subplots(figsize=(9, 6))
sns.kdeplot(grab_scores,  label=f'Grab Food (n={len(grab_scores)})',
            fill=True, alpha=0.4, color='#2E86AB', ax=ax)
sns.kdeplot(panda_scores, label=f'Food Panda (n={len(panda_scores)})',
            fill=True, alpha=0.4, color='#E84855', ax=ax)
ax.axvline(grab_scores.median(),  color='#2E86AB', linestyle='--', linewidth=1.5)
ax.axvline(panda_scores.median(), color='#E84855', linestyle='--', linewidth=1.5)
ax.set_xlabel('Dietary Score (Daily Frequency Factor Sum)')
ax.set_ylabel('Density')
ax.set_title(
    f'Dietary Score Distribution: Grab Food vs Food Panda\n'
    f'Mann-Whitney U = {U_stat:.0f}, p = {p_mwu:.3f}',
    fontweight='bold', pad=12)
ax.legend()
ax.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '13_App_Dietary_KDE.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 13] App ecosystem KDE saved")

# Plot: Cognitive Dissonance Heatmap
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(dissonance_matrix, annot=True, fmt='d', cmap='RdYlGn_r',
            linewidths=0.5, ax=ax, cbar=True, annot_kws={'size': 14})
ax.set_title(
    'Self-Reported Food Choice vs Empirical Dietary Intake Class\n'
    '(Cognitive Dissonance Analysis)',
    fontweight='bold', pad=12)
ax.set_xlabel('Calculated Dietary Reality (Median Split)')
ax.set_ylabel('Self-Perceived Food Choice via OFD')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '14_Cognitive_Dissonance.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 14] Cognitive dissonance heatmap saved")

# Plot: Stacked bar — Perception score by usage group
fig, ax = plt.subplots(figsize=(10, 6))
perc_by_usage = cdf.dropna(subset=['usage_ordinal', 'perception_cat']).copy()
perc_cross = pd.crosstab(perc_by_usage['usage_ordinal'], perc_by_usage['perception_cat'],
                         normalize='index') * 100
perc_cross.index = usage_labels
perc_cross.plot(kind='bar', stacked=True, ax=ax,
                color=[PALETTE[0], PALETTE[2]], edgecolor='white', width=0.6)
ax.set_xlabel('OFD Usage Frequency')
ax.set_ylabel('Proportion (%)')
ax.set_title('Perception of Healthy Food Availability by OFD Usage Frequency', fontweight='bold', pad=12)
ax.legend(title='Perception Level')
ax.grid(axis='y', alpha=0.4)
plt.xticks(rotation=20, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, '15_Perception_by_Usage_Stacked.png'), dpi=200, bbox_inches='tight')
plt.close()
print("  [Plot 15] Perception by usage stacked bar saved")

# ==============================================================================
# PHASE 10: RESULTS SUMMARY
# ==============================================================================
print("\n" + "=" * 65)
print("PHASE 10: RESULTS SUMMARY")
print("=" * 65)

print("""
  ┌─────────────────────────────────────────────────────────────┐
  │ TEST                      STATISTIC    p-VALUE  SIGNIFICANT │
  ├─────────────────────────────────────────────────────────────┤""")
print(f"  │ Spearman ρ (Obj 4 base)    ρ={rho:+.3f}     p={p_spearman:.3f}    {'YES' if p_spearman<0.05 else 'NO':>3}         │")
print(f"  │ Chi-Square (Obj 5 base)    χ²={chi2:.3f}    p={p_chi2:.3f}    {'YES' if p_chi2<0.05 else 'NO':>3}         │")
print(f"  │ ANCOVA — Usage (Obj 4+)    β={beta[1]:+.3f}      p={p_vals[1]:.3f}    {'YES' if p_vals[1]<0.05 else 'NO':>3}         │")
print(f"  │ ANCOVA — Age               β={beta[2]:+.3f}     p={p_vals[2]:.3f}    {'YES' if p_vals[2]<0.05 else 'NO':>3}         │")
print(f"  │ Kendall τ-b (Obj 5+)       τ={tau:+.3f}     p={p_tau:.3f}    {'YES' if p_tau<0.05 else 'NO':>3}         │")
print(f"  │ ANOVA Income × Diet        F={F_inc:.3f}      p={p_inc:.3f}    {'YES' if p_inc<0.05 else 'NO':>3}         │")
print(f"  │ Mann-Whitney GvP           U={U_stat:.0f}      p={p_mwu:.3f}    {'YES' if p_mwu<0.05 else 'NO':>3}         │")
print(f"  │ Kruskal-Wallis Usage×Diet  H={H_stat:.3f}      p={p_kw:.3f}    {'YES' if p_kw<0.05 else 'NO':>3}         │")
print("  └─────────────────────────────────────────────────────────────┘")

print("\n  Cognitive Dissonance:")
print(f"    Of {len(healthy_claimers)} respondents who self-report ordering 'Healthy food',")
print(f"    {dissonant_n} ({pct_dissonant:.1f}%) fall into the empirically Higher Intake category —")
print(f"    suggesting substantial misalignment between self-perception and actual intake.")

print("\n--- ANALYSIS COMPLETE ---")
print(f"  Plots saved to: {plot_dir}")