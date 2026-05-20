# ==============================================================================
# COMPREHENSIVE DATA ANALYSIS PIPELINE: OFD USAGE, DIETARY PATTERNS & PERCEPTION
# Python Version: 3.10.6
# ==============================================================================

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 0. DATA INGESTION & PREPROCESSING
# ------------------------------------------------------------------------------
# Load the dataset. We use partial string matching for column selection to handle
# the often messy output of Google Forms CSV exports.
df_raw = pd.read_csv('RespondentData.csv')
df = df_raw.copy()


# Helper function to find columns dynamically based on keywords
def get_col(keyword):
    cols = [c for c in df.columns if keyword.lower() in c.lower()]
    return cols[0] if cols else None


# Map core columns for easier reference
cols_map = {
    'gender': get_col('Jantina'),
    'age': get_col('Umur'),
    'income': get_col('pendapatan'),
    'app_used': get_col('aplikasi OFD yang paling kerap'),
    'usage_freq': get_col('Berapa kerap anda menggunakan aplikasi'),
    'self_perception': get_col('makanan berkhasiat atau tidak'),
    'top_items': get_col('tiga makanan/minuman')
}

# ------------------------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# Rationale: Establishes the demographic baseline and identifies cohort imbalances.
# ------------------------------------------------------------------------------
print("\n--- SECTION 3: EXPLORATORY DATA ANALYSIS (EDA) ---\n")


def run_eda(dataframe, column_dict):
    print("A. Demographics:")
    print(f"Sample Size: {len(dataframe)}")
    print(dataframe[column_dict['gender']].value_counts(normalize=True) * 100, "\n")
    print("B. Age Distribution:")
    print(dataframe[column_dict['age']].value_counts(), "\n")
    print("C. Socioeconomic Distribution (Income):")
    print(dataframe[column_dict['income']].value_counts(), "\n")

    print("Behavioral:")
    print("A. Market Share Distribution:")
    print(dataframe[column_dict['app_used']].value_counts(), "\n")
    print("B. Usage Frequency:")
    print(dataframe[column_dict['usage_freq']].value_counts(), "\n")
    print("C. Perception Distribution (Healthy vs Unhealthy Self-Report):")
    print(dataframe[column_dict['self_perception']].value_counts(), "\n")


run_eda(df, cols_map)

# ------------------------------------------------------------------------------
# DATA TRANSFORMATIONS (Required for both Part 1 and Part 2)
# ------------------------------------------------------------------------------

# 1. Usage Frequency to Ordinal Mapping
# Rationale: Converts string responses into ranked integers for statistical testing.
freq_mapping = {
    'Tidak pernah / Never': 0,
    '1-3 kali sebulan / 1-3 times per month': 1,
    'Sekali seminggu / Once a week': 2,
    '2-4 kali seminggu / 2-4 times per week': 3,
    '5-6 kali seminggu/ 5-6 times per week': 4,
    'Setiap hari / Every day': 5
}
df['usage_ordinal'] = df[cols_map['usage_freq']].map(freq_mapping).fillna(0)

# 2. FFQ (Dietary Pattern) Scoring Simulation
# Rationale: The proposal requires calculating total grams. This maps categorical
# frequencies to numerical daily conversion factors (e.g., 1/7 = 0.14).
ffq_conversion = {
    'Tidak pernah / Never': 0.0,
    '1-3 kali sebulan / 1-3 times per month': 0.06,  # ~2 times / 30 days
    'Sekali seminggu / Once a week': 0.14,  # 1 / 7 days
    '2-4 kali seminggu / 2-4 times per week': 0.42,  # 3 / 7 days
    '5-6 kali seminggu / 5-6 times per week': 0.78,  # 5.5 / 7 days
    'Sekali sehari / Once a day': 1.0,
    '2-3 kali sehari / 2-3 times per day': 2.5
}

# Isolate FFQ columns (Assuming they contain '[' based on the CSV snippet)
ffq_cols = [c for c in df.columns if '[' in c and 'Cereals' in c]

# Calculate a continuous "Dietary Score" (Proxy for Total Grams)
# Note: In a full production script, you will multiply by the Standard Weight DB here.
df['dietary_score_continuous'] = 0
for col in ffq_cols:
    df['dietary_score_continuous'] += df[col].map(ffq_conversion).fillna(0) * 100  # Assuming avg 100g serving

# 3. Perception of Healthy Food Scoring
# Rationale: Captures Likert scale data for domains: Selection, Quality, Price.
# Assuming columns representing the 3 domains end with '1', '2', '3' or are located at the end.
perception_cols = df.columns[-3:]  # Selecting the last 3 cols as proxy for D1, D2, D3
for col in perception_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(3)  # Default to neutral

df['perception_score_continuous'] = df[perception_cols].sum(axis=1)  # Range 3 - 15

# Part 1 Specific Transformation: Degrading data to nominal categories.
df['perception_binned'] = pd.cut(df['perception_score_continuous'],
                                 bins=[0, 9, 15],
                                 labels=['Low', 'High'])

# ==============================================================================
# PART 1: EXECUTING CLIENT PROPOSAL (BASELINE)
# ==============================================================================
print("\n--- PART 1: CLIENT PROPOSAL STATISTICAL TESTS ---\n")

# Objective 4: Spearman’s Correlation (Usage vs Dietary Pattern)
# Flaw logic: Spearman correlates ranks. It forces continuous dietary data into ranks,
# discarding the exact gram differences between individuals.
spearman_corr, spearman_p = stats.spearmanr(df['usage_ordinal'], df['dietary_score_continuous'])
print(f"Objective 4 (Client - Spearman): Correlation={spearman_corr:.3f}, p-value={spearman_p:.3f}")
print("Comment: This test does not control for age or income, leaving the result exposed to confounding variables.")

# Objective 5: Pearson Chi-Square Test (Usage vs Categorized Perception)
# Flaw logic: Chi-Square treats 'Low' and 'High' as unrelated bins, ignoring that 'High' is strictly greater than 'Low'.
contingency_table = pd.crosstab(df['usage_ordinal'], df['perception_binned'])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Objective 5 (Client - Chi-Square): Chi2={chi2:.3f}, p-value={chi_p:.3f}")
print("Comment: Severe loss of precision by dropping the 3-15 continuous scale into nominal bins.")

# ==============================================================================
# PART 2: IMPROVED EXPERIMENT DESIGN (YOUR TEAM'S RECOMMENDATIONS)
# ==============================================================================
print("\n--- PART 2: IMPROVED METHODOLOGY (MULTIVARIATE) ---\n")

# Need numeric covariates for ANCOVA
income_mapping = {'≤RM5,000': 1, 'RM5,001 - RM10,000': 2, '≥RM10,001': 3}  # Simplified for example
df['income_num'] = df[cols_map['income']].map(income_mapping).fillna(1)
df['age_num'] = pd.factorize(df[cols_map['age']])[0]  # Simplistic encoding of age brackets

# 1. IMPROVED TESTS
# Objective 4: ANCOVA (via OLS Regression)
# Rationale: Tests if usage affects diet WHILE holding income and age constant.
print("1. IMPROVED TESTS")
ancova_model = smf.ols('dietary_score_continuous ~ C(usage_ordinal) + income_num + age_num', data=df).fit()
print("Objective 4 Upgrade (ANCOVA):")
print(ancova_model.summary().tables[1])  # Print just the coefficient table for brevity
print("Rationale: Directly isolates OFD usage impact, retaining the continuous nature of the FFQ data.\n")

# Objective 5: Ordinal Logistic Regression
# Rationale: Respects the native progression (ordinality) of the 3-15 perception score.
try:
    # Requires perception score to be categorical integers for the model
    df['perception_ordinal'] = pd.factorize(df['perception_score_continuous'], sort=True)[0]
    mod_prob = OrderedModel(df['perception_ordinal'], df[['usage_ordinal']], distr='logit')
    res_log = mod_prob.fit(disp=False)
    print("Objective 5 Upgrade (Ordinal Logistic Regression):")
    print(res_log.summary().tables[1])
except Exception as e:
    print(f"Ordinal Model skipped due to sample size constraints in prototype data: {e}")
print("Rationale: Perfectly suited for modeling Likert 'progression' data without destructive binning.\n")

# 2. IMPROVING DEPTH OF RESEARCH
print("2. DEPTH OF RESEARCH EXPLORATIONS")

# 2A. Income Stratification: ANOVA
# Rationale: Tests variance in diet across different economic strata.
income_groups = [group['dietary_score_continuous'].values for name, group in df.groupby('income_num')]
anova_f, anova_p = stats.f_oneway(*income_groups)
print(f"2A (ANOVA - Income vs Diet): F-statistic={anova_f:.3f}, p-value={anova_p:.3f}")
print("Hypothesis: Quantifying if purchasing power mitigates or accelerates digital food consumption patterns.\n")

# 2B. Application Comparisons: Mann-Whitney U Test
# Rationale: Non-parametric test comparing the means of two independent groups (Grab vs FoodPanda).
grab_diet = df[df[cols_map['app_used']].str.contains('Grab', na=False, case=False)]['dietary_score_continuous']
panda_diet = df[df[cols_map['app_used']].str.contains('Panda', na=False, case=False)]['dietary_score_continuous']
if len(grab_diet) > 0 and len(panda_diet) > 0:
    mwu_stat, mwu_p = stats.mannwhitneyu(grab_diet, panda_diet)
    print(f"2B (Mann-Whitney - Grab vs FoodPanda): U-statistic={mwu_stat:.3f}, p-value={mwu_p:.3f}")
print("Hypothesis: Determines if platform algorithms/UI drive differing consumptive outcomes.\n")

# 2C. Perception Discrepancy: Cross-Tabulation
# Rationale: Juxtaposes cognitive self-awareness against empirical FFQ data.
# We create a median split of the actual diet to compare against self-report.
median_diet = df['dietary_score_continuous'].median()
df['actual_diet_bin'] = np.where(df['dietary_score_continuous'] > median_diet, 'Empirically Higher Intake',
                                 'Empirically Lower Intake')

dissonance_matrix = pd.crosstab(df[cols_map['self_perception']], df['actual_diet_bin'])
print("2C (Cross-Tabulation - Self-Awareness vs Reality):")
print(dissonance_matrix)
print(
    "\nHypothesis: Highlights cognitive dissonance. Are respondents aware of the true quality of their dietary environment?")

# End of Pipeline