# %% [markdown]
# # Credit risk: logistic regression vs. random forest
# **Goal:** predict which loan applicants will default ("bad"), using the UCI German Credit data.
#
# **Why not accuracy?** About 30% of applicants are "bad". A model that approves everyone scores ~70% accuracy
# while catching zero defaulters. And the two mistakes are not equally costly (see the written argument below).
# So we report precision, recall, F1, ROC-AUC, PR-AUC, and the *business cost* of each model.
#
# Run top to bottom. The summary at `reports/one_page_summary.md` is generated from the real results in the last cell.

# %%
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
if os.environ.get("HEADLESS") == "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, roc_curve,
                             precision_recall_curve, accuracy_score)
warnings.filterwarnings("ignore")
SEED = 42
os.makedirs("reports/figures", exist_ok=True)

# Cost matrix from the UCI documentation: approving a bad borrower (FN) costs 5x
# as much as rejecting a good one (FP). Change these to match a real lender's economics.
COST_FN, COST_FP = 5, 1
# Set True to drop attributes that are legally/ethically sensitive in lending.
DROP_SENSITIVE = False
# Which dataset: "german" (UCI, ~1k rows, 30% bad) or "gmsc" (Give Me Some Credit, 150k rows, ~7% bad).
# For gmsc, put cs-training.csv in data/ . Switch with:  DATASET=gmsc jupyter notebook
DATASET = os.environ.get("DATASET", "german")
if DATASET == "gmsc":
    COST_FN, COST_FP = 5, 1   # ASSUMPTION: GMSC ships no cost matrix. Replace with the lender's real numbers.
COST_NOTE = ("(the convention for this data set)" if DATASET == "german"
             else "(an assumption: this data set gives no costs, so a lender should substitute its own)")

# %% [markdown]
# ## 1. Load data
# Positive class (`bad = 1`) means the applicant defaulted. We treat "catch the defaulters" as the positive task.

# %%
def make_synthetic(n=1000, seed=SEED):
    """ONLY for testing the pipeline offline (ALLOW_SYNTHETIC=1). Not real data."""
    r = np.random.default_rng(seed)
    p = lambda opts, w=None: r.choice(opts, n, p=w)
    df = pd.DataFrame({
        "checking_status": p(["<0","0<=X<200",">=200","no checking"]),
        "duration": r.integers(4, 72, n),
        "credit_history": p(["critical/other existing credit","existing paid","delayed previously","no credits/all paid","all paid"]),
        "purpose": p(["radio/tv","new car","used car","furniture/equipment","business","education"]),
        "credit_amount": np.exp(r.normal(7.8, .8, n)).round(),
        "savings_status": p(["<100","100<=X<500","500<=X<1000",">=1000","no known savings"]),
        "employment": p(["unemployed","<1","1<=X<4","4<=X<7",">=7"]),
        "installment_commitment": r.integers(1, 5, n),
        "personal_status": p(["male single","female div/dep/mar","male mar/wid","male div/sep"]),
        "other_parties": p(["none","guarantor","co applicant"]),
        "residence_since": r.integers(1, 5, n),
        "property_magnitude": p(["real estate","life insurance","car","no known property"]),
        "age": r.integers(19, 75, n),
        "other_payment_plans": p(["none","bank","stores"]),
        "housing": p(["own","rent","for free"]),
        "existing_credits": r.integers(1, 5, n),
        "job": p(["skilled","unskilled resident","high qualif/self emp/mgmt","unemp/unskilled non res"]),
        "num_dependents": r.integers(1, 3, n),
        "own_telephone": p(["none","yes"]),
        "foreign_worker": p(["yes","no"]),
    })
    z = (-1.6 + (df.checking_status == "<0")*1.0 + df.duration/40 + np.log(df.credit_amount)/6
         + (df.savings_status == "<100")*.5 - (df.age/100) + r.normal(0, .8, n))
    df["bad"] = (z > np.quantile(z, .70)).astype(int)
    return df

def load_german_credit():
    try:
        from sklearn.datasets import fetch_openml
        df = fetch_openml("credit-g", version=1, as_frame=True).frame.copy()
        df["bad"] = (df["class"].astype(str) == "bad").astype(int)
        return df.drop(columns="class"), "UCI German Credit (via OpenML credit-g)"
    except Exception as e:
        if os.environ.get("ALLOW_SYNTHETIC") == "1":
            print("!! Using SYNTHETIC data for pipeline testing only:", e)
            return make_synthetic(), "SYNTHETIC (test only)"
        raise RuntimeError("Could not download German Credit. Check your connection, or place "
                           "the data as data/german_credit.csv with a 0/1 'bad' column.") from e

def make_synthetic_gmsc(n=20000, seed=SEED):
    """ONLY for testing the pipeline offline. Mimics the GMSC schema, quirks included. Not real data."""
    r = np.random.default_rng(seed)
    age = r.integers(21, 90, n)
    inc = np.exp(r.normal(8.6, .6, n)).round()
    df = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": np.clip(r.beta(.6, 1.5, n) * 1.3, 0, None),
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": r.poisson(.3, n),
        "DebtRatio": r.beta(1, 3, n) * 1.5,
        "MonthlyIncome": inc,
        "NumberOfOpenCreditLinesAndLoans": r.poisson(8, n),
        "NumberOfTimes90DaysLate": r.poisson(.12, n),
        "NumberRealEstateLoansOrLines": r.poisson(1, n),
        "NumberOfTime60-89DaysPastDueNotWorse": r.poisson(.1, n),
        "NumberOfDependents": r.poisson(.7, n).astype(float),
    })
    z = (-4.1 + 1.2*df.RevolvingUtilizationOfUnsecuredLines + 0.7*df["NumberOfTimes90DaysLate"]
         + 0.5*df["NumberOfTime30-59DaysPastDueNotWorse"] - 0.025*(age-50) + r.normal(0, 1, n))
    df["bad"] = (r.random(n) < 1/(1+np.exp(-z))).astype(int)
    df.loc[r.random(n) < .2, "MonthlyIncome"] = np.nan                      # ~20% missing income
    df.loc[r.random(n) < .02, "DebtRatio"] = r.integers(2000, 300000, (n,))[0]  # absurd ratios
    df.loc[r.random(n) < .002, ["NumberOfTimes90DaysLate", "NumberOfTime30-59DaysPastDueNotWorse"]] = 98  # sentinel code
    return df

def load_gmsc():
    for p in ("data/cs-training.csv", "data/gmsc.csv"):
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
            df["bad"] = df["SeriousDlqin2yrs"].astype(int)
            return df.drop(columns="SeriousDlqin2yrs"), "Give Me Some Credit (cs-training.csv)"
    if os.environ.get("ALLOW_SYNTHETIC") == "1":
        print("!! Using SYNTHETIC data for pipeline testing only")
        return make_synthetic_gmsc(), "SYNTHETIC (test only)"
    raise FileNotFoundError("Download cs-training.csv from kaggle.com/c/GiveMeSomeCredit and put it in data/")

if DATASET == "gmsc":
    raw, DATA_NAME = load_gmsc()
elif os.path.exists("data/german_credit.csv"):
    raw, DATA_NAME = pd.read_csv("data/german_credit.csv"), "data/german_credit.csv"
else:
    raw, DATA_NAME = load_german_credit()
print(DATA_NAME, raw.shape)
print("\nClass balance:\n", raw["bad"].value_counts(normalize=True).rename({0: "good", 1: "bad"}).round(3))

# %% [markdown]
# ## 2. Feature engineering
# Raw columns describe the applicant at one point in time. We add ratios and flags that capture
# *burden* (how heavy is this loan relative to the applicant?) and *stability*.

# %%
SAVINGS = {"<100": 1, "100<=X<500": 2, "500<=X<1000": 3, ">=1000": 4}          # "no known savings" -> NaN-> 0 flag below
EMPLOY = {"unemployed": 0, "<1": 0.5, "1<=X<4": 2.5, "4<=X<7": 5.5, ">=7": 8}  # approx. years in current job

def engineer_german(df):
    d = df.copy()
    for c in d.select_dtypes(["category"]).columns:
        d[c] = d[c].astype(str)
    d["monthly_payment"]   = d["credit_amount"] / d["duration"]
    d["log_credit_amount"] = np.log1p(d["credit_amount"])
    d["burden_score"]      = d["installment_commitment"] * d["monthly_payment"]   # rate x payment size
    d["credit_per_dependent"] = d["credit_amount"] / (1 + d["num_dependents"])
    d["long_loan"]         = (d["duration"] > 36).astype(int)
    d["young_borrower"]    = (d["age"] < 25).astype(int)
    d["savings_level"]     = d["savings_status"].map(SAVINGS).fillna(0)
    d["no_savings_info"]   = (~d["savings_status"].isin(SAVINGS)).astype(int)
    d["years_employed"]    = d["employment"].map(EMPLOY).fillna(2.5)
    d["overdrawn_checking"] = (d["checking_status"] == "<0").astype(int)
    d["has_other_plans"]   = (d["other_payment_plans"] != "none").astype(int)
    d["credit_x_age"]      = d["credit_amount"] / d["age"]
    if DROP_SENSITIVE:
        d = d.drop(columns=["personal_status", "foreign_worker", "age", "young_borrower", "credit_x_age"], errors="ignore")
    return d

def engineer_gmsc(df):
    """Features built from the Give Me Some Credit data dictionary. Missing values are left as NaN and
    imputed inside the model pipeline (so imputation statistics come from training folds only)."""
    d = df.copy()
    late30, late60, late90 = "NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse", "NumberOfTimes90DaysLate"
    # 96 / 98 are placeholder codes in this data set, not real counts
    d["late_code_flag"] = d[[late30, late60, late90]].isin([96, 98]).any(axis=1).astype(int)
    for c in (late30, late60, late90):
        d[c] = d[c].where(~d[c].isin([96, 98]))
    d["age"] = d["age"].where(d["age"] >= 18)                                   # age 0 is a data error
    d["income_missing"] = d["MonthlyIncome"].isna().astype(int)                 # missingness can itself be informative
    d["utilization"] = d["RevolvingUtilizationOfUnsecuredLines"].clip(0, 2)      # >2 is almost surely an error
    d["over_limit"] = (d["RevolvingUtilizationOfUnsecuredLines"] > 1).astype(int)
    d["debt_ratio"] = d["DebtRatio"].clip(0, 5)
    d["debt_ratio_absurd"] = (d["DebtRatio"] > 5).astype(int)                    # usually means income was missing
    d["log_income"] = np.log1p(d["MonthlyIncome"])
    d["income_per_dependent"] = d["MonthlyIncome"] / (1 + d["NumberOfDependents"])
    d["total_past_due"] = d[[late30, late60, late90]].sum(axis=1, min_count=1)
    d["any_90d_late"] = (d[late90] > 0).astype(int)
    d["weighted_lateness"] = d[late30] + 2 * d[late60] + 3 * d[late90]           # later = worse
    d["real_estate_share"] = d["NumberRealEstateLoansOrLines"] / d["NumberOfOpenCreditLinesAndLoans"].clip(lower=1)
    d["many_credit_lines"] = (d["NumberOfOpenCreditLinesAndLoans"] > 15).astype(int)
    d["young_borrower"] = (d["age"] < 30).astype(int)
    d = d.drop(columns=["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"])
    if DROP_SENSITIVE:
        d = d.drop(columns=["age", "young_borrower"], errors="ignore")
    return d

engineer = engineer_gmsc if DATASET == "gmsc" else engineer_german
data = engineer(raw)
X, y = data.drop(columns="bad"), data["bad"]
num_cols = X.select_dtypes("number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
if not cat_cols:
    cat_cols = []
print(f"{len(num_cols)} numeric + {len(cat_cols)} categorical features")

# %% [markdown]
# ## 3. Split, preprocessing, and how class imbalance is handled
# * **Stratified** train/test split keeps the ~30% bad rate in both sets.
# * **Class weights** make each missed defaulter count more during training (no synthetic rows needed at this size).
# * **Threshold tuning** (section 5) is the second, and usually bigger, lever: the default 0.5 cut-off is wrong when errors have unequal costs.
# * The test set is touched only once, at the end.

# %%
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

def prep():
    return ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])

models = {
    "Logistic regression": Pipeline([("prep", prep()),
        ("clf", LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))]),
    "Random forest": Pipeline([("prep", prep()),
        ("clf", RandomForestClassifier(n_estimators=300 if len(X) > 5000 else 500, min_samples_leaf=30 if len(X) > 5000 else 5, max_features="sqrt",
                                       class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))]),
}
baseline = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)

# %% [markdown]
# ## 4. Metrics at the default threshold (and why accuracy misleads)

# %%
def cost(yt, yp):
    tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
    return COST_FN * fn + COST_FP * fp, dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

def report(yt, proba, thr):
    yp = (proba >= thr).astype(int)
    c, cm = cost(yt, yp)
    return dict(threshold=round(float(thr), 3), accuracy=accuracy_score(yt, yp),
                precision=precision_score(yt, yp, zero_division=0), recall=recall_score(yt, yp),
                f1=f1_score(yt, yp), roc_auc=roc_auc_score(yt, proba),
                pr_auc=average_precision_score(yt, proba), cost=int(c), **cm)

skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
oof = {}
for name, m in models.items():
    oof[name] = cross_val_predict(m, X_tr, y_tr, cv=skf, method="predict_proba")[:, 1]
    m.fit(X_tr, y_tr)

rows = {"Approve everyone (baseline)": report(y_te, np.zeros(len(y_te)), 0.5)}
rows["Approve everyone (baseline)"]["roc_auc"] = 0.5
for name, m in models.items():
    rows[name + " @0.5"] = report(y_te, m.predict_proba(X_te)[:, 1], 0.5)
pd.DataFrame(rows).T.round(3)

# %% [markdown]
# Read the baseline row: high accuracy, **recall = 0**. That is why accuracy alone is meaningless here.

# %% [markdown]
# ## 5. Choose the decision threshold by cost
# We pick the threshold on **out-of-fold training predictions** (never the test set) that minimises
# `5 x missed defaulters + 1 x wrongly rejected good customers`.

# %%
grid = np.linspace(0.05, 0.95, 91)
best_thr, curves = {}, {}
for name in models:
    costs = [cost(y_tr, (oof[name] >= t).astype(int))[0] for t in grid]
    curves[name] = costs
    best_thr[name] = float(grid[int(np.argmin(costs))])
    print(f"{name}: best threshold {best_thr[name]:.2f}  (train OOF cost {min(costs)})")

plt.figure(figsize=(6, 3.5))
for name, c in curves.items(): plt.plot(grid, c, label=name)
plt.axvline(0.5, ls=":", c="grey"); plt.xlabel("Decision threshold (P(default) above which we reject)")
plt.ylabel("Total cost (train, cross-validated)"); plt.legend(); plt.tight_layout()
plt.savefig("reports/figures/threshold_cost.png", dpi=150); plt.show()

# %% [markdown]
# ## 6. Final test-set comparison

# %%
final = {"Approve everyone (baseline)": rows["Approve everyone (baseline)"]}
test_proba = {}
for name, m in models.items():
    test_proba[name] = m.predict_proba(X_te)[:, 1]
    final[f"{name} @0.5"] = rows[f"{name} @0.5"]
    final[f"{name} @cost-optimal"] = report(y_te, test_proba[name], best_thr[name])
results = pd.DataFrame(final).T
results[["threshold","accuracy","precision","recall","f1","roc_auc","pr_auc","cost","tp","fn","fp","tn"]].round(3)

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name in models:
    fpr, tpr, _ = roc_curve(y_te, test_proba[name]); ax[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y_te, test_proba[name]):.2f})")
    p, r, _ = precision_recall_curve(y_te, test_proba[name]); ax[1].plot(r, p, label=f"{name} (AP {average_precision_score(y_te, test_proba[name]):.2f})")
ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(xlabel="False positive rate", ylabel="Recall", title="ROC")
ax[1].axhline(y_te.mean(), c="k", ls=":"); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (dotted = base rate)")
ax[0].legend(); ax[1].legend(); plt.tight_layout(); plt.savefig("reports/figures/roc_pr.png", dpi=150); plt.show()

# %% [markdown]
# Because the test set has only ~250 rows (~75 bad), single-split numbers are noisy. Cross-validated spread:

# %%
from sklearn.model_selection import cross_validate
cvres = {}
for name, m in models.items():
    s = cross_validate(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1),
                       scoring=["roc_auc", "average_precision"])
    cvres[name] = dict(roc_auc_mean=s["test_roc_auc"].mean(), roc_auc_sd=s["test_roc_auc"].std(),
                       pr_auc_mean=s["test_average_precision"].mean())
pd.DataFrame(cvres).T.round(3)

# %% [markdown]
# ## 7. Which error is worse? (written argument)
#
# Let "positive" = applicant will default.
#
# * **False negative (we approve someone who defaults)**: the lender loses principal, collection costs and the
#   opportunity to lend that money elsewhere. On a typical consumer loan that is most of the loan amount.
# * **False positive (we reject someone who would have repaid)**: the lender loses only the *interest margin*
#   it would have earned, plus a bit of goodwill. That is a fraction of the loan amount.
#
# So **the false negative is the worse error**, in the data's own cost matrix by a factor of 5. The practical consequences:
# 1. We should accept more false positives (reject some good customers) in exchange for catching more defaulters, so the
#    model is tuned for **high recall** on the "bad" class, then precision is managed rather than maximised.
# 2. The right cut-off is far below 0.5. With a 5:1 cost ratio, the break-even is roughly "reject if P(default) > 1/6 = 0.17" when
#    probabilities are well calibrated; the empirical search above finds the practical version of this.
# 3. **Caveats:** the 5:1 ratio is a convention from the dataset documentation. Real ratios depend on loan size, interest margin,
#    and recovery rate, and rejecting good customers also has fairness and growth consequences that this simple cost does not capture.
#    Rejections should also be reviewable, since a rejected customer is the one who never gets to prove the model wrong.

# %% [markdown]
# ## 8. What drives the predictions? (plain language)
# **Permutation importance**: shuffle one original column at a time on the test set and measure how much
# ROC-AUC drops. A bigger drop means the model leaned on that information more. Works the same for both models.

# %%
def perm(m, n=None):
    Xp, yp = (X_te, y_te) if len(X_te) <= 8000 else (X_te.sample(8000, random_state=SEED), None)
    yp = y_te if yp is None else y_te.loc[Xp.index]
    n = n or (30 if len(X_te) <= 8000 else 8)
    r = permutation_importance(m, Xp, yp, scoring="roc_auc", n_repeats=n, random_state=SEED, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X_te.columns).sort_values(ascending=False)

imp = {name: perm(m) for name, m in models.items()}
imp_df = pd.DataFrame(imp).sort_values("Random forest", ascending=False)
display_top = imp_df.head(12)
display_top.round(4)

# %%
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
for a, name in zip(ax, models):
    s = imp[name].head(10)[::-1]; a.barh(s.index, s.values); a.set_title(f"{name}\n(drop in ROC-AUC when shuffled)")
plt.tight_layout(); plt.savefig("reports/figures/importance.png", dpi=150); plt.show()

# %% [markdown]
# **Direction of effect (logistic regression).** Odds ratios tell you whether a feature *raises or lowers* default risk.
# For numeric features the ratio is per one standard deviation; for categories it is relative to the average applicant.

# %%
lr = models["Logistic regression"]
names = lr.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=names)
odds = np.exp(coef).rename("odds_ratio").sort_values()
print("Strongest risk REDUCERS:\n", odds.head(6).round(2), "\n\nStrongest risk RAISERS:\n", odds.tail(6)[::-1].round(2))

# %% [markdown]
# ## 9. Generate the one-page summary from the real numbers

# %%
FRIENDLY = {
 "checking_status": "the state of their checking account", "duration": "how long the loan runs",
 "credit_amount": "the size of the loan", "credit_history": "their past repayment record",
 "savings_status": "their savings balance", "age": "their age", "purpose": "what the loan is for",
 "employment": "how long they've held their job", "monthly_payment": "the monthly repayment amount",
 "burden_score": "how heavy the repayments are relative to income", "log_credit_amount": "the size of the loan",
 "overdrawn_checking": "whether their checking account is overdrawn", "savings_level": "their savings balance",
 "installment_commitment": "repayments as a share of income", "property_magnitude": "what property they own",
 "other_payment_plans": "other instalment plans", "housing": "their housing situation",
 "years_employed": "years in current job", "other_parties": "co-signers or guarantors",
 "utilization": "how much of their available credit they are already using", "over_limit": "being over their credit limit",
 "debt_ratio": "debts relative to income", "debt_ratio_absurd": "unusable debt-to-income figures",
 "log_income": "monthly income", "MonthlyIncome": "monthly income", "income_missing": "whether income was reported",
 "income_per_dependent": "income per person supported", "total_past_due": "how often they have paid late",
 "any_90d_late": "ever being 90+ days late", "weighted_lateness": "how often and how badly they have paid late",
 "NumberOfTimes90DaysLate": "times 90+ days late", "NumberOfTime30-59DaysPastDueNotWorse": "times 30-59 days late",
 "NumberOfTime60-89DaysPastDueNotWorse": "times 60-89 days late", "NumberOfOpenCreditLinesAndLoans": "number of open credit lines",
 "NumberRealEstateLoansOrLines": "number of mortgages / real-estate loans", "NumberOfDependents": "number of dependents",
 "real_estate_share": "share of credit that is real-estate", "many_credit_lines": "having many open credit lines",
 "young_borrower": "being a younger borrower", "late_code_flag": "placeholder codes in the lateness fields"}
fr = lambda c: FRIENDLY.get(c, c.replace("_", " "))

cost_models = {n: final[f"{n} @cost-optimal"] for n in models}
winner = min(cost_models, key=lambda n: min(cost(y_tr, (oof[n] >= t).astype(int))[0] for t in grid))
w, w05 = cost_models[winner], final[f"{winner} @0.5"]
base_cost = final["Approve everyone (baseline)"]["cost"]
saving = 100 * (1 - w["cost"] / base_cost)
top = list(dict.fromkeys(fr(c) for c in imp[winner].head(6).index))[:3]
n_te, n_bad = len(y_te), int(y_te.sum())
clean = lambda n: fr(n.split("__", 1)[-1])
raisers = ", ".join(clean(n) for n in odds.tail(3)[::-1].index)
reducers = ", ".join(clean(n) for n in odds.head(3).index)
lo = cvres["Logistic regression"]["roc_auc_mean"]; rf = cvres["Random forest"]["roc_auc_mean"]

LIM_DATA = ("Small, old data. About 1,000 applicants from the 1990s; results will wobble and must be re-checked on current data."
            if DATASET == "german" else
            "Rare outcome, messy data. Only about 7% defaulted, and fields such as income are often missing or implausible, "
            "so we cleaned and flagged them. Results describe this historical sample, not any particular lender's book.")
SENS = ("Age, personal status and nationality are in this data." if DATASET == "german" else "Age is in this data.")
summary = f"""# Can we predict who will default on a loan?
*One-page summary for non-technical readers. Data: {DATA_NAME}, {len(raw)} past applicants, {raw['bad'].mean():.0%} of whom defaulted.*

## The question
Given what we know about an applicant at the time they apply, can we flag the ones who are likely to stop paying,
without turning away too many people who would have paid us back?

## Why "percent correct" is the wrong yardstick
Only about {raw['bad'].mean():.0%} of applicants default. A "model" that approves everybody is therefore right
about {final['Approve everyone (baseline)']['accuracy']:.0%} of the time, and catches **none** of the people we most need to catch.
We judged the models on *how many defaulters they catch* (recall), *how often a flag is right* (precision),
and the **money lost to mistakes**.

## Which mistake is worse
Approving someone who then defaults means losing most of the loan. Turning away someone who would have repaid means losing
only the profit on that loan. We treated the first mistake as **{COST_FN} times as costly** as the second {COST_NOTE}.
So we deliberately accept some wrongly-rejected good customers in order to catch more defaulters.

## What we did
We built two models: a simple, transparent one (logistic regression) and a more flexible one (random forest, many decision trees voting).
Both were tested on {n_te} applicants ({n_bad} defaulters) they had never seen.

## Results (on those {n_te} unseen applicants)
| | Catches defaulters (recall) | Flag is right (precision) | Overall ranking skill (ROC-AUC) | Cost vs. approving everyone |
|---|---|---|---|---|
| Approve everyone | 0% | n/a | 0.50 | 100% |
| {winner}, standard cut-off | {w05['recall']:.0%} | {w05['precision']:.0%} | {w05['roc_auc']:.2f} | {100*w05['cost']/base_cost:.0f}% |
| **{winner}, cost-aware cut-off** | **{w['recall']:.0%}** | **{w['precision']:.0%}** | **{w['roc_auc']:.2f}** | **{100*w['cost']/base_cost:.0f}%** |

Moving the cut-off to reflect the real cost of mistakes cut estimated losses by about **{saving:.0f}%** compared with approving everyone.
Across repeated cross-validation, the two models ranked applicants about equally well
(logistic regression ROC-AUC {lo:.2f}, random forest {rf:.2f}), so the simpler, easier-to-explain model is a strong default
unless the forest wins clearly on your own data.

## What the model pays attention to
The information that mattered most was: **{top[0]}**, **{top[1]}**, and **{top[2]}**. We measured this by scrambling each piece of
information in turn and seeing how much worse the predictions got. In the simple model, the factors that most raised the odds of default were: {raisers}; the factors that most lowered it were: {reducers}.
Patterns like these should be sanity-checked against lending common sense before anyone relies on them.

## Limits and cautions
* **{LIM_DATA}**
* **Not perfect.** At the chosen cut-off the model still rejects some good customers (about {1-w['precision']:.0%} of its flags are wrong) and misses some defaulters.
* **Fairness.** {SENS} In real lending these are legally sensitive and should be removed or audited (switch `DROP_SENSITIVE`).
* **Costs are assumptions.** The {COST_FN}:{COST_FP} ratio should be replaced with the lender's own numbers; the cut-off will move with it.
* **Use as decision support**, with human review and a route for applicants to contest a rejection.
"""
os.makedirs("reports", exist_ok=True)
open("reports/one_page_summary.md", "w").write(summary)
results.round(4).to_csv("reports/test_results.csv")
json.dump({"winner": winner, "cv": cvres, "thresholds": best_thr}, open("reports/results.json", "w"), indent=2, default=float)
print(summary)
