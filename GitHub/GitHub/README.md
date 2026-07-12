# Guillermo Gracia — Project Portfolio

A selection of academic projects spanning physics/ML (Bachelor's), operations research and applied statistical learning (Master's).

| Project | Description | Stack |
|---|---|---|
| [`tfg-pinn-fluid-dynamics`](./tfg-pinn-fluid-dynamics) | Bachelor's thesis (Physics): Physics-Informed Neural Networks solving the linear transport and Burgers' PDEs, benchmarked against analytical and finite-volume solutions. | Python, TensorFlow, Keras |
| [`tfm-locker-delivery-optimization`](./tfm-locker-delivery-optimization) | Master's thesis: Vehicle Routing Problem with Simultaneous Pickup/Delivery for parcel-locker networks. Phase 1: real NYC locker network with a compartment-feasibility model and KPI framework. Phase 2 (current): an ML-augmented, time-of-day-aware "learnheuristic" that penalizes routes likely to hit saturated lockers, benchmarked on the Rudy (2025) PLBDP instances. | Python, scikit-learn, XGBoost, OSRM |
| [`statistical-learning-methods`](./statistical-learning-methods) | Master's coursework: ensemble learning, bagging, boosting (incl. XGBoost), K-Means/DBSCAN clustering, and PCA-based customer segmentation. Includes an applied piece combining DBSCAN clustering with a Clarke-Wright VRP solver. | Python, scikit-learn, XGBoost |

Each project folder has its own README with problem description, method, and key results.

## Note on this portfolio

These repos were assembled from original coursework folders, each curated to keep the repo focused and runnable:

- **TFG:** full thesis write-up plus all PINN notebooks/scripts and generated result plots.
- **TFM:** raw data files, generated benchmark instances, trained model artifacts and cached distance matrices are left out (all regenerable from the included scripts) — see that project's README for the Phase 1 / Phase 2 split.
- **Statistical Learning:** a curated subset (one or two illustrative scripts per topic) rather than every weekly exercise from the course.
