# My Viva Notes

---

## Data Preprocessing & Imputation

**Q1: Why did you drop the row where gender = "Other"?**  
There was literally one row out of 5,110 with gender "Other". If I kept it, it would create a dummy variable with basically zero variance. That messes up Chi-Square tests (expected counts < 5) and can cause issues in logistic regression. Dropping one row doesn't hurt anything.

**Q2: Why use stratified median imputation for BMI instead of just the overall median?**  
BMI depends a lot on age and gender — a kid's BMI is very different from an elderly person's. If I just used the global median, it would wash out those patterns. So I grouped by age band and gender, then used each group's median. I checked and it performs about the same as KNN imputation but it's way simpler.

**Q3: Why did you create a `bmi_was_missing` flag?**  
This was actually one of the coolest findings. Patients with missing BMI had a 19.9% stroke rate compared to only 4.3% for everyone else — that's almost 5x higher. My guess is that missing BMI means the patient came in through the ER and was too sick to stand on a scale. So the missingness itself is a useful feature. The logistic regression confirmed it: OR = 3.70, p < 1e-9.

**Q4: How did you handle Unknown smoking status?**  
I looked at who had "Unknown" and found that 521 out of 543 kids under 12 had it. So for children, I just set it to "never smoked" — makes sense, they're kids. For the adults, I tried imputing vs. just keeping "Unknown" as its own category. Both gave basically the same AUC (~0.847), so I kept "Unknown" as-is because I didn't want to make up smoking histories for people.

**Q5: Why cap BMI at 65 instead of removing outliers?**  
There were 5 people with BMI > 65 (one was 97.6 which seems like a data error). They were all non-stroke cases. I didn't want to delete them because that would mess with the class balance. Capping at 65 keeps them in the dataset without letting extreme values pull the model around. I checked — it makes no difference to AUC either way.

---

## EDA & Statistical Tests

**Q6: Why not use accuracy as the main metric?**  
Because the dataset is 95% "no stroke." A dumb model that always says "no stroke" gets 95.1% accuracy but catches zero actual strokes. That's useless for a screening tool. I used ROC-AUC, PR-AUC, Recall, and F1 instead — these actually care about finding the minority class.

**Q7: Why Mann-Whitney U instead of just a t-test?**  
I ran normality tests (Shapiro-Wilk, D'Agostino) and they all rejected normality for age, glucose, and BMI. So a regular t-test's assumptions don't hold. Mann-Whitney is non-parametric — it works with ranks and doesn't need normal distributions. I reported both tests anyway, but Mann-Whitney is the one I trust more here.

**Q8: What does Cramér's V add beyond the Chi-Square p-value?**  
The p-value just says "is there a significant association or not." But with 5,000+ rows, even tiny associations can be statistically significant. Cramér's V tells you how strong the association actually is (0 = none, 1 = perfect). For example, Residence_type had p = 0.30 (not significant) with V = 0.015 (basically nothing), while bmi_was_missing had V = 0.139 which is a decent effect.

**Q9: How do you read the adjusted OR for hypertension?**  
The adjusted OR is 1.39 (CI: 1.00–1.92, p = 0.049). What that means is: after controlling for age, glucose, heart disease, BMI, smoking, and everything else, hypertensive patients have 39% higher odds of stroke. The confidence interval just barely avoids crossing 1.0, so it's borderline significant. That makes sense — a lot of hypertension's effect works through age and glucose, so once you adjust for those, the independent contribution gets smaller.

**Q10: What does the age OR of 2.02 per 10 years mean?**  
For every 10 extra years of age, the odds of stroke roughly double, holding everything else constant. It's the strongest predictor in the model by far. This matches what you see in the medical literature — stroke risk roughly doubles every decade after 55.

---

## Feature Engineering & Modeling

**Q11: Why split before fitting imputers and scalers?**  
If you fit your transformers on the whole dataset and then split, the model has indirectly "seen" the test data distributions. That's data leakage and it makes your performance numbers look better than they actually are. I split first, fit everything on the training set only, and then transform the test set.

**Q12: Why apply SMOTE inside CV folds, not before?**  
If you SMOTE the whole training set before cross-validation, some of the synthetic samples generated from one fold's data can leak into another fold's validation set. That gives you overly optimistic results. I used imbalanced-learn's Pipeline so SMOTE only runs on each fold's training portion. This is the correct way to do it.

**Q13: Why did balanced class weights beat SMOTE?**  
`class_weight='balanced'` just reweights the loss function so minority-class mistakes cost more. SMOTE creates synthetic data points by interpolating between existing ones. The problem is that with one-hot encoded features, interpolation gives you weird fractional values between 0 and 1 for binary dummies — those aren't real data points. For this dataset, the simpler reweighting approach worked better.

**Q14: Why is logistic regression competitive with XGBoost and Random Forest?**  
I think it's because stroke risk in this data is mostly driven by smooth, monotonic relationships — older age = more risk, higher glucose = more risk, etc. These are exactly the kind of patterns logistic regression handles well. The tree models add complexity but don't gain much because there aren't strong non-linear interactions that my engineered features (like age×glucose, cardiometabolic risk count) don't already capture. Plus LR gives you interpretable coefficients, which is nice for a clinical tool.

**Q15: What is Youden's J and how did you use it?**  
Youden's J = Sensitivity + Specificity − 1. It finds the threshold on the ROC curve where you get the best tradeoff between catching true positives and avoiding false positives. I calculated it on out-of-fold predictions from 5-fold CV (not on the test set) so I wouldn't overfit the threshold. The result was around 0.49–0.54, giving me 80–84% recall with 64–75% specificity.

---

## Streamlit App

**Q16: What are the 3 risk tiers?**  
- **Low Risk:** predicted probability below the screening threshold (~0.54). Just routine care.
- **Moderate Risk:** between the screening and priority thresholds (0.54–0.74). Should probably get checked out.
- **High Risk:** above the priority threshold (≥0.74). Needs urgent evaluation.

The thresholds come from the CV-calibrated cutoffs and you can tweak them in the app sidebar.

**Q17: How does the explainability tab work?**  
Since I'm using logistic regression, each prediction is just a sum of (feature value × coefficient). The bar chart in Tab 2 shows each feature's contribution to the final score — positive bars push risk up, negative bars reduce it. So you can see exactly why a specific patient got their risk level.

**Q18: Why include a disclaimer?**  
The model is trained on one retrospective dataset. It doesn't know about imaging, genetics, medications, or any causal relationships. It's just a school project, not a real medical tool. The disclaimer makes that clear so nobody mistakes it for actual clinical advice.

---

## GitHub & Project Management

**Q19: Why use feature branches?**  
Instead of committing everything to main and hoping nothing breaks, I used separate branches for each phase (cleaning, EDA, modeling, app, docs). This way I could review changes before merging and easily roll back if something went wrong. It's also just good practice for any real project.

**Q20: Why are data files gitignored?**  
CSVs can be large and you shouldn't put raw data in version control anyway. The `.gitignore` excludes CSVs, zip files, the venv folder, `__pycache__`, and notebook checkpoints. The data README has the Kaggle link and instructions so anyone can download it themselves.

---

## Harder Questions

**Q21: What if the imbalance was even worse, like 0.1% positive?**  
I'd look into cost-sensitive learning with higher penalties, anomaly detection approaches like Isolation Forest, ensemble methods designed for imbalance (BalancedBagging, EasyEnsemble), focal loss for gradient boosting, or maybe collecting more positive cases through enriched sampling.

**Q22: How would you validate this for a real hospital?**  
You'd need external validation on data from other hospitals and populations, prospective testing on future data, calibration analysis to check if the predicted probabilities actually match reality, fairness audits across demographics, privacy compliance checks, and integration testing with actual EHR systems. Basically a lot more work than a capstone project.

**Q23: ROC-AUC vs PR-AUC — when does each matter more?**  
ROC-AUC uses TPR vs FPR and is fairly stable across class imbalances since FPR is dominated by the big majority class. PR-AUC uses Precision vs Recall and is more sensitive to imbalance because precision directly measures what fraction of your flagged patients are actually positive. For a 5% prevalence problem like this, PR-AUC gives you a better picture of how useful the model would be in practice.

**Q24: Why StandardScaler instead of MinMaxScaler?**  
StandardScaler centers features to zero mean and unit variance. This matters for logistic regression because L2 regularization penalizes all coefficients equally — if features aren't on the same scale, the penalty is unfair. MinMaxScaler would squish everything into [0,1] but it's really sensitive to outliers, and my BMI and glucose columns have some extreme values.

**Q25: If you could add one external data source, what would it be?**  
Longitudinal blood pressure readings — like systolic and diastolic over time from medical records. Right now hypertension is just a yes/no variable, which throws away a lot of information about how severe it is, how long they've had it, and whether it's controlled. Having the actual BP trajectories would make the risk predictions way better and you could even do survival analysis with time-to-event modeling.
