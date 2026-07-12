# Parcel Locker Delivery Route Optimization

Master's Thesis project (TFM) — Universidad Politécnica de Valencia, Master in Computer Engineering
Author: Guillermo Gracia Rebullida

A Vehicle Routing Problem with Simultaneous Pickup and Delivery (VRPSPD) solver for parcel-locker networks, where each stop may require both **deliveries** and **pickups** subject to a hard constraint that doesn't appear in classical VRP formulations: **locker compartment capacity**.

The project has two phases, run on two different datasets:

- **Phase 1 — pilot study on real data.** A full VRPSPD + locker-feasibility model built on real NYC parcel-locker locations and reservation logs, written up as the formal submitted report ([`Report.md`](./Report.md)).
- **Phase 2 — ML-augmented learnheuristic (current, ongoing).** The routing heuristic is extended with a machine-learning-predicted locker-saturation penalty, developed and rigorously benchmarked on the academic PLBDP instances of Rudy (2025), including a time-of-day-aware dynamic variant and a Monte-Carlo stochastic-release model. This is the more advanced and more recent piece of work, and the one with the most runnable code in this repo.

## Phase 1 — real-data pilot (LockerNYC)

Two real, open datasets: **LockerNYC Locations** (coordinates + S/M/L/XL compartment inventory per station) and **LockerNYC Reservations** (real daily transaction logs used to build realistic demand instances). Real travel distances/times come from OpenStreetMap + OSRM.

- **Depot placement:** geometric median of the 95 locker coordinates (Weiszfeld algorithm), matched to the nearest real logistics facility (Industry City, Sunset Park, Brooklyn).
- **Routing engine:** Clarke & Wright Savings heuristic with Biased Randomization inside a GRASP multi-start loop; distance and time savings normalized to `[0,1]` and blended with a tunable weight `α`.
- **Locker feasibility model:** each node is simulated through its daily visit sequence (pickups first, freeing compartments, then best-fit delivery placement) to flag *structural infeasibility* — a delivery parcel with no compatible compartment, which no routing strategy can fix.
- **KPI framework** (`phase1_lockernyc_report/metrics.py`): three groups of supply-chain KPIs — routing efficiency (travel time, distance, vehicle utilization, load factor), service quality (On-Time Delivery rate, service rate in volume units), and resilience (Performance Impact index `PI = 1 - service_rate`).

**Key results:** on real reservation instances (4 operational days), load factors of 8.6–18.1% against a 6-vehicle fleet, 100% OTD, no infeasibility — the pilot network runs well below fleet capacity. Synthetic stress tests (19 → 90 active lockers) show infeasibility growing with network density (PI: 0% → 8.9%), and — counter-intuitively — a **higher pickup ratio reduces** infeasibility (fewer delivery parcels per node, so fewer compartment-size mismatches). All infeasibility observed is structural (an infrastructure constraint), not a heuristic weakness.

Full derivation, KPI tables and discussion in [`Report.md`](./Report.md) (LaTeX source: [`phase1_lockernyc_report/report_source.tex`](./phase1_lockernyc_report/report_source.tex)). The KPI computation module is kept as runnable reference code; the rest of the Phase 1 pipeline (OSRM matrix building, instance generation from raw reservation logs, the interactive map) is described in the report but not included here, since it depends on raw data files and cached distance matrices that aren't part of this repo.

## Phase 2 — ML-augmented learnheuristic (Rudy 2025 PLBDP benchmark)

The second phase moves to an academic benchmark — the weight-based Parcel Locker Bike/Delivery Problem (PLBDP) instances of Rudy (2025), which don't include GPS coordinates or physical compartments but do include per-locker capacity and realistic time-of-day traffic speed profiles. The goal: can a machine-learning model that predicts *locker saturation risk* be folded into the routing heuristic itself, so routes proactively avoid parcels that are likely to be rejected on arrival?

### Formulation

`s(i,j) = α · s_dist(i,j) + (1-α) · s_time(i,j) − β · P(saturation at j) · d_fallback(j)`

A biased-randomized Clarke-Wright saving is penalized by the probability that a locker will be saturated when the vehicle arrives, weighted by the cost of the resulting fallback (re-routing or failed delivery). Full mathematical formulation in [`code/formulation_plbdp.tex`](./code/formulation_plbdp.tex).

### Evolution of the approach

The design went through several iterations, each addressing a concrete failure of the previous one (see inline docstrings for the full technical detail):

1. **Static saturation model** (`heuristic_learn.py`) — an ML model predicts a single saturation probability per locker from demand features; penalizes CWS savings directly. An early attempt to blend a *pre-route* travel-time estimate into the same formula was reverted after it turned out to be a rescaled copy of the distance term rather than independent signal (a circularity bug caught before it reached results).
2. **Time-aware dynamic heuristic** (`heuristic_dynamic.py`) — recomputes savings live during route construction using the vehicle's actual accumulated arrival time at each candidate node (`Route.tail_time`), since a time-dependent saving is route-state-dependent and different on every GRASP iteration. A genuine distance+time+ML fusion formula (correcting an earlier double-counting bug) only accepts a merge if the fused saving is positive.
3. **Option C — hourly saturation model** (`heuristic_c.py`) — a locker-saturation model conditioned on delivery hour (`FEATURE_COLS_C`), with all 24-hour × all-node predictions pre-batched into a single `predict_proba` call for performance. Labels are generated by solving with the no-ML dynamic heuristic, replaying real arrival times, and simulating real overflow (`generate_labels_C.py`).
4. **Option D — stochastic release model** (`heuristic_d.py`) — models `P(delivered_ok | initial_occupancy_ratio, arrival_hour)` using a Monte-Carlo simulation of background compartment releases (`simulator.simulate_solution_stochastic`) rather than a single deterministic snapshot.

Earlier synthetic-saturation variants (referred to as Model A/B in the design log) were superseded by C/D and are not included here.

### Methodology and results

Train/test split (80/20, seed 42) is computed once over the *combined* pool of low- and high-saturation instance sets and shared between label generation and evaluation, specifically to avoid the contamination of training on the same instances later used for benchmarking. On the resulting dataset (~89,700 labelled rows, ~6.9% positive overflow rate), a gradient-boosted model was the best performer: **AUC ≈ 0.964, F1 ≈ 0.583 ± 0.006**, with `arrival_hour` contributing non-trivial (1.9–5.1%) feature importance alongside the dominant static occupancy/size features.

Benchmarked against the plain distance-based heuristic and a time-aware-but-ML-free variant (`run_experiments_C.py`, `run_experiments_D.py`, reporting distance, makespan `Tmax`, and fallback/effective km), the honest finding is nuanced rather than a clean win: the **time-of-day component is the main driver** of the improvements observed, and the ML saturation penalty adds **marginal robustness — cheaper fallbacks when they do occur — but does not reliably reduce the number of fallbacks below the pure-distance baseline** in the tested sample. This was diagnosed by checking feature importances (static occupancy/size features dominate at 47–64% combined, `arrival_hour` only 1–5%) and by testing whether iterating the label-generation bootstrap with a trained model changed anything (it didn't, to 3 decimal places) — evidence that the effect ceiling is set by the static features, not a training artifact. This is reported here as-is rather than as a more favorable but less accurate conclusion.

A one-factor-at-a-time sensitivity sweep over `β`, `p_bias`, departure hour and GRASP iteration count is in `run_sensitivity.py`.

## Repository contents

```
Report.md                              # Phase 1 formal report (full write-up)
phase1_lockernyc_report/
  report_source.tex                    # LaTeX source of Report.md
  metrics.py                           # Phase 1 KPI computation module (reference)
code/                                  # Phase 2: ML-augmented learnheuristic (runnable)
  model.py                             # Node, Edge, Route, Solution data model
  heuristic.py                         # standard BR-CWS + GRASP baseline (distance only)
  heuristic_learn.py                   # static ML saturation-penalty heuristic
  heuristic_dynamic.py                 # time-of-day-aware dynamic heuristic (real arrival times)
  heuristic_c.py                       # Option C: hourly saturation model + dynamic fusion
  heuristic_d.py                       # Option D: stochastic release-risk model
  instance_reader.py                   # Rudy (2025) .txt instance parser + feature schemas
  instance.h                           # original Rudy (2025) C++ instance generator (reference)
  generate_instances_B.py              # Python port: synthetic high-saturation instance generator
  ml_common.py                         # label helpers, model factories (RF/GBM/LR/XGB), CV scoring
  simulator.py                         # deterministic + stochastic route/overflow simulation
  speed_profile.py                     # 24h traffic speed profile (Rudy 2025, Table 5)
  generate_labels_C.py, generate_labels_D.py  # label generation + shared train/test split
  train_model_C.py, train_model_D.py   # model training pipelines
  run_experiments_C.py, run_experiments_D.py  # std vs. time-only vs. full-ML benchmark
  run_sensitivity.py                   # OFAT sensitivity sweep
  formulation_plbdp.tex                # mathematical formulation of the PLBDP + learnheuristic
```

**Not included in this repo:** raw data files, generated instance sets (`data/instances*`), trained model artifacts (`data/models*`), and cached distance/time matrices — all regenerable from the scripts above.

## Tech stack

Python, NumPy, pandas, scikit-learn, XGBoost, LaTeX. Phase 1 additionally used the OSRM routing API and folium.
