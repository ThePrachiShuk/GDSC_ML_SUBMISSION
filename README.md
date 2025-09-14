# GDSC ML Recruitment Submission

## 🔬 Project Overview
This repository contains a comprehensive machine learning solution for predicting `CORRUCYSTIC_DENSITY` from the MiNDAT dataset. This set of solutions of mine showcases multiple approaches, from simple baseline models to advanced ensemble techniques with hyperparameter optimization.

## 📊 Dataset
- **Target Variable**: `CORRUCYSTIC_DENSITY` - A continuous regression target
- **Training Data**: MiNDAT.csv (with both numerical and categorical features)
- **Test Data**: MiNDAT_UNK.csv (for final predictions)
- **Challenge**: Predicting alien biocomputer fragment analysis data with complex feature interactions

## 🧪 Experimental Approaches

### 1. Initial EDA & Baseline (Approach 1)
- **Goal**: Understanding data distribution and establishing baseline performance
- **Features**: Basic preprocessing, simple imputation, standard scaling
- **Models**: Random Forest, Gradient Boosting, Ridge, Lasso
- **Key Insights**: Identified high correlation features and outlier patterns

### 2. Enhanced Feature Engineering (Approach 2)
- **Improvements**: 
  - Better outlier detection using IQR method
  - KNN imputation for missing values
  - Feature interactions and polynomial terms
  - Robust scaling techniques
- **Models**: Extended ensemble with XGBoost and LightGBM
- **Results**: Significant improvement over baseline

### 3. Advanced Preprocessing Pipeline (Approaches 3-4)
- **Advanced Features**:
  - Iterative imputation for complex missing patterns
  - Target encoding for categorical variables with smoothing
  - Advanced statistical features (skewness, kurtosis, energy measures)
  - Domain-specific alien biocomputer features
- **Model Ensemble**: 7-fold cross-validation with multiple algorithms
- **Stacking**: Meta-learner using Gradient Boosting

### 4. Production-Ready Pipeline (Approach 5)
- **Comprehensive Feature Engineering**:
  - Memory sequence features (treating columns as temporal data)
  - Network topology features (distributed processing patterns)
  - Signal processing features (periodicity, complexity)
  - Advanced statistical aggregations
- **Models**: 
  - LightGBM, XGBoost, CatBoost (tree-based)
  - Ridge, Lasso, Elastic Net, Bayesian Ridge (linear)
  - Random Forest, Extra Trees (ensemble)
- **Advanced Techniques**:
  - Stratified K-fold with target binning
  - Multiple meta-learners for stacking
  - Weighted ensemble based on CV performance

### 5. Optuna Hyperparameter Optimization (Approach 6)
- **Automated Tuning**: Used Optuna for hyperparameter optimization
- **TPE Sampler**: Tree-structured Parzen Estimator for efficient search
- **Models Optimized**: LightGBM, XGBoost, CatBoost, Random Forest
- **Configuration**: 30 trials with 3-fold CV for faster optimization

### 6. Ultra-Advanced Pipeline (Approach 7)
- **Enhanced Features**:
  - Interaction features between top predictors
  - Polynomial transformations
  - Binning features for non-linear patterns
  - Clustering-based features
  - Noise injection for regularization
- **Multi-Level Stacking**:
  - Level 1: Base model predictions
  - Level 2: Diverse meta-learners (GBDT, RF, Ridge)
  - Level 3: Final ensemble with adaptive weights
- **Semi-Supervised Learning**: Pseudo-labeling with confidence thresholding
- **Advanced Techniques**: Repeated K-fold, uncertainty-based target encoding

### 7. Neural Network Approach (Approach 8)
- **Architecture**: Simple tabular neural network with dropout
- **Features**: MPS/CUDA acceleration support for Apple Silicon/GPU
- **Training**: Early stopping, batch processing, robust scaling
- **Hybrid Ensemble**: Combined with classical ML models
- **Note**: Didn't achieve best performance but provided model diversity

## 🏆 Key Technical Features

### Data Preprocessing
```python
# Advanced outlier detection
Q1, Q3 = train[target].quantile([0.25, 0.75])
IQR = Q3 - Q1
extreme_outliers = detect_outliers_iqr(data) & detect_outliers_zscore(data)

# Robust target encoding with smoothing
smoothed_means = (target_means * counts + global_mean * smoothing_factor) / (counts + smoothing_factor)
```

### Feature Engineering
```python
# Domain-specific biocomputer features
df["corr_stability"] = 1 / (1 + df["corr_cv"])
df["corr_coherence"] = df["corr_positive_count"] / (df["corr_positive_count"] + df["corr_negative_count"] + 1e-8)
df["signal_noise"] = np.abs(df["mean"]) / (df["std"] + 1e-8)
```

### Advanced Ensemble
```python
# Multi-level stacking with diverse meta-learners
meta_models = {
    'GBDT_meta': GradientBoostingRegressor(...),
    'RF_meta': RandomForestRegressor(...),
    'Ridge_meta': Ridge(...)
}
```

## 📈 Model Performance Evolution

| Approach | Key Innovation | CV RMSE Improvement |
|----------|----------------|-------------------|
| Baseline | Simple preprocessing | Starting point |
| Enhanced FE | Feature interactions | ~15% improvement |
| Advanced Pipeline | Domain features + stacking | ~25% improvement |
| Optuna Tuned | Hyperparameter optimization | ~30% improvement |
| Ultra-Advanced | Multi-level stacking + pseudo-labeling | ~35% improvement |

## 🔧 Technical Stack
- **Core ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Optimization**: Optuna with TPE sampler
- **Deep Learning**: PyTorch with MPS/CUDA support
- **Feature Engineering**: Advanced statistical transforms, domain-specific features
- **Validation**: Stratified K-fold, repeated cross-validation
- **Ensemble**: Multi-level stacking, adaptive weighting

## 🚀 Best Practices Implemented
1. **Robust Cross-Validation**: Stratified K-fold with target binning
2. **Leakage Prevention**: Proper target encoding with out-of-fold methodology
3. **Feature Selection**: Multiple methods (F-regression, mutual information, correlation)
4. **Model Diversity**: Tree-based, linear, and neural network models
5. **Hyperparameter Optimization**: Automated tuning with Optuna
6. **Advanced Ensembling**: Multi-level stacking with meta-learners
7. **Semi-Supervised Learning**: Pseudo-labeling for additional training data

## 📁 Files Structure
- `final_compiled.ipynb`: Complete experimental notebook with all approaches
- `README.md`: This comprehensive documentation
- Generated files (when run):
  - `submission.csv`: Final predictions
  - `oof_predictions.csv`: Out-of-fold validation predictions
  - `feature_importances_*.csv`: Model-specific feature importance analysis

## 🎯 Final Results
The best performing model achieved a significant improvement over baseline through the combination of:
- Advanced feature engineering with domain knowledge
- Hyperparameter optimization using Optuna
- Multi-level ensemble stacking
- Robust cross-validation methodology

This project demonstrates a complete machine learning pipeline from exploratory data analysis to production-ready model deployment, showcasing both classical ML techniques and modern optimization methods.
