# SQB7027-Group-Assignment-1-Data-Analysis

# Comprehensive Data Analysis Pipeline: OFD Usage, Dietary Patterns & Perception

## Project Thesis
This analysis evaluates the impact of Online Food Delivery (OFD) applications on dietary patterns and consumer perceptions. This pipeline executes a two-part experimental design: first establishing a baseline by replicating the original bivariate statistical proposals, and second, applying an upgraded multivariate methodology to correct for variance loss and confounding variables.

## Part 1: Baseline Methodology (Client Proposal)
This segment executes the tests exactly as outlined in the initial research proposal. It systematically degrades continuous data into nominal or ordinal categories to satisfy basic statistical models.

* **Objective 4 (Usage vs. Dietary Pattern):** * **Test:** Spearman’s Rank Correlation.
  * **Mechanism:** Converts continuous dietary intake (total grams) into ranks. 
  * **Limitation:** Fails to account for critical demographic covariates (Age, Income), exposing the results to confounding variables.
* **Objective 5 (Usage vs. Perception):**
  * **Test:** Pearson Chi-Square Test.
  * **Mechanism:** Bins a continuous 3-15 point Likert composite score into 'Low' and 'High' nominal categories.
  * **Limitation:** Destroys the inherent hierarchical progression of the Likert scale data.

## Part 2: Upgraded Experiment Design (Multivariate Expansion)
This segment overrides the baseline by deploying advanced statistical models designed to maximize the predictive power of the original, un-binned data structures.

### 2.1 Improved Statistical Models
* **Objective 4 Upgrade:** ANCOVA (Analysis of Covariance).
  * **Rationale:** Utilizes the continuous dietary intake data as the dependent variable without transformation. It introduces Age and Income as covariates, isolating the true, unconfounded effect of OFD Usage Frequency on diet.
* **Objective 5 Upgrade:** Ordinal Logistic Regression.
  * **Rationale:** Natively respects the ordinal progression of the perception Likert data, eliminating the need for destructive data binning while providing tighter predictive accuracy.

### 2.2 Depth of Research Explorations
This section explores the peripheral variables within the dataset to provide a more holistic view of the digital food environment.
* **Income Stratification (ANOVA):** Quantifies whether purchasing power influences the frequency of usage and the types of food ordered.
* **Application Comparisons (Mann-Whitney U Test):** Compares dietary outcomes between specific app ecosystems (e.g., GrabFood vs. FoodPanda) to determine if differing UI/algorithms drive different consumptive behaviors.
* **Perception Discrepancy (Cross-Tabulation):** Juxtaposes self-reported health awareness against empirical Food Frequency Questionnaire (FFQ) data to measure cognitive dissonance within the cohort.

## Conclusion
By transitioning from the baseline Spearman and Chi-Square tests to ANCOVA and Ordinal Logistic Regression, this pipeline rescues the variance inherent in the continuous data formats. This strict dependency on un-binned data ensures the final analysis isolates the true variables and defends the thesis against demographic confounders.