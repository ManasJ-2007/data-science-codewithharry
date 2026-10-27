# Machine learning with scikit-learn

## Types of ML
| Type | Data | Examples |
|---|---|---|
| Supervised: regression | features → number | house price (project 2) |
| Supervised: classification | features → class | spam vs not spam, churn |
| Unsupervised | features only | clustering (KMeans), PCA, anomaly detection |
| Reinforcement learning | states, actions, rewards | game playing, trading ([FinSearch](https://github.com/ManasJ-2007/finsearch-rl-trading)) |

## The workflow
1. **Frame the problem and metric** (RMSE for prices, F1 for imbalanced classes).
2. **Split first.** Use `train_test_split`, or `StratifiedShuffleSplit` to keep an important variable's distribution. The test set is touched **once**, at the very end.
3. **Explore** the training set only: distributions, correlations, missing values, capped values.
4. **Preprocess in a `Pipeline` / `ColumnTransformer`**: impute → engineer features → scale → encode. The same steps then run at training and prediction time, which prevents data leakage.
5. **Compare models with k-fold cross-validation:** `cross_val_score(model, X, y, cv=10, scoring="neg_root_mean_squared_error")`.
6. **Tune** with `GridSearchCV` / `RandomizedSearchCV`.
7. **Evaluate once on the test set**, with a confidence interval, then save with `joblib.dump`.

## Concepts
- **scikit-learn API:** estimators have `fit`, transformers `transform`, predictors `predict`. Learned attributes end in `_` (`imputer.statistics_`).
- **Scaling:** `StandardScaler` (z-score) for most models; `MinMaxScaler` for bounded ranges. Trees don't need scaling.
- **Encoding:** `OneHotEncoder` for unordered categories; `OrdinalEncoder` only when order is real.
- **Bias–variance:** a deep decision tree has near-zero training error but high CV error (overfitting); linear regression has the opposite problem (underfitting). Ensembles (random forest averages trees, gradient boosting fits residuals) reduce one or the other.
- **Regression metrics:** RMSE (penalises big errors), MAE (robust), R² = 1 − SS_res / SS_tot.
- **Classification metrics:** accuracy, precision = TP/(TP+FP), recall = TP/(TP+FN), F1, ROC-AUC, confusion matrix.

**Applied in:** project 2 covers every step above. It compares 6 models with 10-fold CV, writes two custom transformers, runs a 30-candidate randomised search, and evaluates once on the test set with a 95% CI (R² 0.86).
