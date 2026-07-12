# Statistical Learning — Methods & Applications

Coursework project, Master in Computer Engineering (Statistical Learning course), Universidad Politécnica de Valencia.
Author: Guillermo Gracia

A curated set of implementations covering the core statistical/machine learning toolkit taught across the course: ensemble learning fundamentals, bagging, boosting, clustering, and dimensionality reduction — applied to both synthetic datasets and small real-world case studies (customer segmentation, VRP node clustering).

## Topics covered

**1. Ensemble learning fundamentals** (`01_ensemble_learning_intro/`)
Bias-variance tradeoff, bootstrap aggregating from scratch (fitting many decision trees on bootstrap resamples and averaging predictions), and model stacking (combining a decision tree, k-NN and SVM through a Random Forest meta-learner via `StackingClassifier`).

**2. Bagging techniques** (`02_bagging_techniques/`)
Random Forest classification with `max_features='sqrt'` decision splits and decision-boundary visualization; k-fold stratified cross-validation compared against a single train/test split to show variance in accuracy estimates.

**3. Boosting methods** (`03_boosting_methods/`)
Gradient-boosted trees via XGBoost's native `DMatrix`/`xgb.train` API on the Iris dataset (multi-class softmax objective), with confusion-matrix evaluation. The course also covered AdaBoost, standard Gradient Boosting, LightGBM and CatBoost, and a related case study (**Hurricane Losses**) applying binary and 3-class XGBoost classifiers to insurance loss data.

**4-5. Clustering — K-Means** (`04_clustering_kmeans/`)
K-Means with the Elbow Method (WCSS vs. k) to select the number of clusters, plus centroid visualization on synthetic blob data.

**6. Clustering — DBSCAN applied to routing** (`05_dbscan_vrp_clustering/`)
A more applied piece: DBSCAN density-based clustering used as a **pre-processing/decomposition step for the Vehicle Routing Problem** — partition customers into density-based clusters first, then solve a Clarke-Wright Savings heuristic independently within each cluster, benchmarked on CVRPLIB instances (A/B/E/F/M/P-series) with silhouette-score validation of the clustering.

**7. Dimensionality reduction** (`06_dimensionality_reduction/`)
Customer segmentation pipeline: standardization → correlation analysis → PCA (loadings heatmap, explained variance) → K-Means on the PCA scores → labeled customer segments (e.g. "career focused", "well-off") visualized on the first two principal components.

## Repository contents

```
01_ensemble_learning_intro/
    bootstrap_and_bagging.py     # bagging from scratch: N decision trees on bootstrap resamples
    stacking.py                  # StackingClassifier: tree + kNN + SVM -> RandomForest meta-model
02_bagging_techniques/
    random_forest_classifier.py  # RandomForest decision boundary on make_moons
    cross_validation.py          # StratifiedKFold CV vs. single split, high-dimensional synthetic data
03_boosting_methods/
    xgboost_iris.py               # native XGBoost API, multi-class classification + confusion matrix
04_clustering_kmeans/
    elbow_method.py               # WCSS elbow curve + K-Means cluster visualization
05_dbscan_vrp_clustering/
    dbscan_cws_routing.py         # DBSCAN clustering + per-cluster Clarke-Wright Savings VRP solver
    vrp_objects.py                # Node/Edge/Route/Solution classes used by the routing heuristic
06_dimensionality_reduction/
    customer_segmentation_pca.py  # standardize -> PCA -> K-Means -> labeled customer segments
```

**Not included in this repo:** the remaining weekly exercise scripts (AdaBoost, LightGBM, CatBoost, Hierarchical Clustering, factor analysis rotations, feature-importance/early-stopping/grid-search utilities, and the Hurricane Losses case-study notebook/images) and the course's `Final_Project_Guillermo_Gracia.pdf`, all of which remain in the original coursework folder. This repo is a representative selection of one or two illustrative scripts per topic rather than the full weekly archive.

## Tech stack

Python, scikit-learn, XGBoost, pandas, NumPy, matplotlib, seaborn, NetworkX.
