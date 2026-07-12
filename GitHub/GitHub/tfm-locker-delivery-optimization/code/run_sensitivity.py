"""
run_sensitivity.py
-------------------
One-factor-at-a-time (OFAT) parameter sensitivity sweep for a SINGLE
instance: how total distance changes as beta, p_bias, departure_h or n_iter
vary, holding the others at a baseline.

Why OFAT instead of a full grid: matches "comparar para una instancia
concreta diferentes valores de los parametros" directly, keeps runtime
predictable, and produces one clean line-chart per parameter for the thesis
instead of a combinatorial explosion.

Efficiency: build_graph_learn/build_graph_dynamic(+ML) bake beta/departure_h
into the savings list at BUILD time, so sweeping those requires a fresh
build per value. p_bias and n_iter only affect the br_CWS* sampling LOOP, so
this script builds the candidate graph once per (method, beta, departure_h)
combination and reuses it across every p_bias value and across the whole
n_iter convergence curve, rather than going through run_grasp()/
run_grasp_learn()/run_grasp_dynamic() (which rebuild on every call).

Usage
-----
    python run_sensitivity.py <instance.txt>
    python run_sensitivity.py <instance.txt> --sweep beta,p_bias
    python run_sensitivity.py <instance.txt> --betas 0.1,0.3,0.5 --n_iter 20

Output
------
    data/sensitivity_beta.csv, _p_bias.csv, _departure_h.csv, _n_iter.csv
    (only the ones in --sweep are written)
"""

from __future__ import annotations
import os
import sys
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from instance_reader import read_full_instance
from heuristic import INF, N_ITER, P_BIAS, build_graph, br_CWS
from heuristic_learn import build_graph_learn, br_CWS_learn, BETA_DEFAULT
from heuristic_dynamic import build_graph_dynamic, br_CWS_dynamic
from heuristic_c import build_graph_c, br_CWS_c, ALPHA_DEFAULT
from simulator import simulate_solution
from run_comparison import (
    MODEL_A_CANDIDATES, MODEL_B_CANDIDATES, RESULTS_DIR,
    _resolve_model_path, _try_load,
)

DEFAULT_BETAS        = [0.1, 0.2, 0.3, 0.4, 0.5]
DEFAULT_P_BIASES     = [0.10, 0.25, 0.40]
DEFAULT_DEPARTURE_HS = [7.0, 8.0, 12.0, 17.0]   # morning peak, baseline, midday, evening peak
DEFAULT_ALPHAS       = [0.0, 0.25, 0.5, 0.75, 1.0]   # 0=pure time, 1=pure distance

_HERE = os.path.dirname(__file__)
MODEL_C_CANDIDATES = [os.path.join(_HERE, '..', 'data', 'models_C', 'saturation_C_best.pkl')]


# =============================================================================
# GENERIC BUILD-ONCE / LOOP-MANY DRIVER
# =============================================================================

def _grasp_loop(br_fn, active_nodes, savings_list, vehicle_cap, depot, n_iter: int,
                checkpoint_every: int | None = None, **br_kwargs):
    """Run n_iter GRASP iterations of br_fn over an already-built candidate
    list, tracking best-cost-so-far. Returns (best_sol, curve) where curve is
    [(iter, best_cost_km), ...] recorded every checkpoint_every iterations
    (plus the final one); checkpoint_every=None records only the final point."""
    ck = checkpoint_every or n_iter
    best_sol, best_cost = None, INF
    curve = []
    for it in range(1, n_iter + 1):
        sol = br_fn(active_nodes, savings_list, vehicle_cap, depot, **br_kwargs)
        if sol.cost < best_cost:
            best_sol, best_cost = sol, sol.cost
        if it % ck == 0 or it == n_iter:
            curve.append((it, round(best_cost / 1000, 4)))
    return best_sol, curve


def _run_loop(kind: str, active_nodes, savings, vehicle_cap, depot, n_iter: int,
             p_bias: float, dep_h: float, checkpoint_every: int | None = None):
    """kind in {'std', 'learn', 'dynamic'} -- dispatches to the right br_CWS*."""
    if kind in ('std', 'learn'):
        br_fn  = br_CWS if kind == 'std' else br_CWS_learn
        kwargs = {'p': p_bias}
    else:
        br_fn  = br_CWS_dynamic
        kwargs = {'departure_h': dep_h, 'p': p_bias}
    return _grasp_loop(br_fn, active_nodes, savings, vehicle_cap, depot,
                       n_iter, checkpoint_every, **kwargs)


def _available_ml_methods(bundle_a, bundle_b):
    """(name, kind, bundle) for every ML-capable method whose bundle loaded."""
    methods = []
    if bundle_a is not None:
        methods.append(('learn_A', 'learn', bundle_a))
        methods.append(('dynamic_A', 'dynamic', bundle_a))
    if bundle_b is not None:
        methods.append(('learn_B', 'learn', bundle_b))
        methods.append(('dynamic_B', 'dynamic', bundle_b))
    return methods


def _fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km: float) -> dict:
    """Locker-overflow fallback KPIs (simulator.simulate_solution, reused as-is
    -- see run_comparison.py for the same pattern). Empty dict if no capacity
    data is available for this instance."""
    if not locker_cap or best_sol is None:
        return {}
    fb_km, n_fb = simulate_solution(best_sol, orders, locker_cap, dist_matrix)
    return {'fallback_km': round(fb_km, 4), 'n_fallbacks': n_fb,
           'effective_km': round(dist_km + fb_km, 4)}


# =============================================================================
# SWEEPS
# =============================================================================

def sweep_beta(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders, dep_h,
               bundle_a, bundle_b, betas: list[float], p_bias: float, n_iter: int,
               verbose: bool = False) -> list[dict]:
    """beta is build-level for learn/dynamic+ML -- rebuild per value. Reports
    fallback_km/effective_km (simulator.simulate_solution) alongside raw
    dist_km, since beta's whole purpose is trading distance for less
    overflow -- dist_km alone can't show whether it's actually helping."""
    rows = []
    for name, kind, bundle in _available_ml_methods(bundle_a, bundle_b):
        for beta in betas:
            t0 = time.perf_counter()
            if kind == 'learn':
                a_nodes, sav = build_graph_learn(nodes, dist_matrix, bundle, beta,
                                                 dep_h, locker_cap=locker_cap)
            else:
                a_nodes, sav = build_graph_dynamic(nodes, dist_matrix, bundle, beta,
                                                   locker_cap, dep_h)
            build_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            best_sol, _ = _run_loop(kind, a_nodes, sav, vehicle_cap, depot, n_iter,
                                    p_bias, dep_h)
            loop_s = time.perf_counter() - t1

            dist_km = round(best_sol.cost / 1000, 4)
            row = {'method': name, 'beta': beta, 'dist_km': dist_km,
                  'n_routes': len(best_sol.routes),
                  'build_s': round(build_s, 3), 'loop_s': round(loop_s, 3)}
            row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
            rows.append(row)
            if verbose:
                eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
                print(f"    {name:<10} beta={beta:<5} -> {dist_km:.3f} km{eff} "
                      f"(build {build_s:.2f}s, loop {loop_s:.2f}s)")
    return rows


def sweep_p_bias(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders, dep_h,
                 bundle_a, bundle_b, beta: float, p_biases: list[float], n_iter: int,
                 verbose: bool = False) -> list[dict]:
    """p_bias is loop-level for every method -- build once, reuse across values."""
    rows = []
    builds: dict[str, tuple] = {}   # name -> (kind, active_nodes, savings, build_s)

    t0 = time.perf_counter()
    a_nodes, sav = build_graph(nodes, dist_matrix)
    builds['standard'] = ('std', a_nodes, sav, time.perf_counter() - t0)

    t0 = time.perf_counter()
    a_nodes, sav = build_graph_dynamic(nodes, dist_matrix)   # bundle=None
    builds['dynamic'] = ('dynamic', a_nodes, sav, time.perf_counter() - t0)

    for name, kind, bundle in _available_ml_methods(bundle_a, bundle_b):
        t0 = time.perf_counter()
        if kind == 'learn':
            a_nodes, sav = build_graph_learn(nodes, dist_matrix, bundle, beta,
                                             dep_h, locker_cap=locker_cap)
        else:
            a_nodes, sav = build_graph_dynamic(nodes, dist_matrix, bundle, beta,
                                               locker_cap, dep_h)
        builds[name] = (kind, a_nodes, sav, time.perf_counter() - t0)

    for name, (kind, a_nodes, sav, build_s) in builds.items():
        for pb in p_biases:
            t1 = time.perf_counter()
            best_sol, _ = _run_loop(kind, a_nodes, sav, vehicle_cap, depot, n_iter,
                                    pb, dep_h)
            loop_s = time.perf_counter() - t1
            dist_km = round(best_sol.cost / 1000, 4)
            row = {'method': name, 'p_bias': pb, 'dist_km': dist_km,
                  'n_routes': len(best_sol.routes),
                  'build_s': round(build_s, 3), 'loop_s': round(loop_s, 3)}
            row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
            rows.append(row)
            if verbose:
                eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
                print(f"    {name:<10} p_bias={pb:<5} -> {dist_km:.3f} km{eff} "
                      f"(build {build_s:.2f}s, loop {loop_s:.2f}s)")
    return rows


def sweep_departure_h(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                      bundle_a, bundle_b, beta: float, p_bias: float,
                      departure_hs: list[float], n_iter: int,
                      verbose: bool = False) -> list[dict]:
    """departure_h is loop-level for plain 'dynamic' (build_graph is
    time-independent) but build-level for learn/dynamic+ML (bakes into
    t_factor_j / the ML congestion features). 'standard' has no time
    dependency at all, so it's excluded from this sweep."""
    rows = []

    t0 = time.perf_counter()
    dyn_nodes, dyn_sav = build_graph_dynamic(nodes, dist_matrix)   # bundle=None
    dyn_build_s = time.perf_counter() - t0
    for dep_h in departure_hs:
        t1 = time.perf_counter()
        best_sol, _ = _run_loop('dynamic', dyn_nodes, dyn_sav, vehicle_cap, depot,
                                n_iter, p_bias, dep_h)
        loop_s = time.perf_counter() - t1
        dist_km = round(best_sol.cost / 1000, 4)
        row = {'method': 'dynamic', 'departure_h': dep_h,
              'dist_km': dist_km, 'n_routes': len(best_sol.routes),
              'build_s': round(dyn_build_s, 3), 'loop_s': round(loop_s, 3)}
        row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
        rows.append(row)
        if verbose:
            eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
            print(f"    dynamic    dep_h={dep_h:<5} -> {dist_km:.3f} km{eff} "
                  f"(build {dyn_build_s:.2f}s, loop {loop_s:.2f}s)")

    for name, kind, bundle in _available_ml_methods(bundle_a, bundle_b):
        for dep_h in departure_hs:
            t0 = time.perf_counter()
            if kind == 'learn':
                a_nodes, sav = build_graph_learn(nodes, dist_matrix, bundle, beta,
                                                 dep_h, locker_cap=locker_cap)
            else:
                a_nodes, sav = build_graph_dynamic(nodes, dist_matrix, bundle, beta,
                                                   locker_cap, dep_h)
            build_s = time.perf_counter() - t0

            t1 = time.perf_counter()
            best_sol, _ = _run_loop(kind, a_nodes, sav, vehicle_cap, depot, n_iter,
                                    p_bias, dep_h)
            loop_s = time.perf_counter() - t1

            dist_km = round(best_sol.cost / 1000, 4)
            row = {'method': name, 'departure_h': dep_h,
                  'dist_km': dist_km, 'n_routes': len(best_sol.routes),
                  'build_s': round(build_s, 3), 'loop_s': round(loop_s, 3)}
            row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
            rows.append(row)
            if verbose:
                eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
                print(f"    {name:<10} dep_h={dep_h:<5} -> {dist_km:.3f} km{eff} "
                      f"(build {build_s:.2f}s, loop {loop_s:.2f}s)")
    return rows


def sweep_n_iter(nodes, dist_matrix, vehicle_cap, depot, locker_cap, dep_h,
                 bundle_a, bundle_b, beta: float, p_bias: float,
                 n_iter_max: int, checkpoint_every: int,
                 verbose: bool = False) -> list[dict]:
    """Single run per method at baseline params, checkpointing best-cost-so-far
    every checkpoint_every iterations -- one pass yields the full convergence
    curve instead of separate reruns per candidate n_iter value.

    Distance-only (no fallback_km/effective_km): the checkpoint mechanism only
    tracks a scalar best-cost-so-far, not a full Solution snapshot at each
    checkpoint, so there's no solution to run simulate_solution() against per
    point without re-architecting the checkpointing. See sweep_beta/
    sweep_p_bias/sweep_departure_h for effective-distance (with fallback)
    comparisons."""
    rows = []
    methods = [('standard', 'std', build_graph(nodes, dist_matrix)),
              ('dynamic', 'dynamic', build_graph_dynamic(nodes, dist_matrix))]
    for name, kind, bundle in _available_ml_methods(bundle_a, bundle_b):
        if kind == 'learn':
            built = build_graph_learn(nodes, dist_matrix, bundle, beta, dep_h,
                                      locker_cap=locker_cap)
        else:
            built = build_graph_dynamic(nodes, dist_matrix, bundle, beta, locker_cap, dep_h)
        methods.append((name, kind, built))

    for name, kind, (a_nodes, sav) in methods:
        _, curve = _run_loop(kind, a_nodes, sav, vehicle_cap, depot, n_iter_max,
                             p_bias, dep_h, checkpoint_every)
        for it, cost_km in curve:
            rows.append({'method': name, 'iter': it, 'dist_km': cost_km})
        if verbose:
            print(f"    {name:<10} final @ iter {n_iter_max}: {curve[-1][1]:.3f} km")
    return rows


def sweep_alpha_c(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders, dep_h,
                  bundle_c, alphas: list[float], beta: float, p_bias: float, n_iter: int,
                  verbose: bool = False) -> list[dict]:
    """
    Ablation sweep for heuristic_c.py: alpha (distance/time blend) at two
    beta settings -- 0 (no ML penalty, pure distance+time) and the given
    baseline beta (with penalty) -- directly answering "how much better is
    distance+time+penalty vs. just distance+time".

    alpha AND beta are BOTH loop-level for Option C (unlike beta for
    learn/dynamic+ML): build_graph_c's hourly P_j table only depends on
    bundle/locker_cap, not alpha/beta (those only enter _fused_saving_c at
    merge-decision time) -- so build once, reuse across the whole grid.
    """
    rows = []
    t0 = time.perf_counter()
    a_nodes, sav, hourly_probs, d_fb = build_graph_c(nodes, dist_matrix, bundle_c, locker_cap)
    build_s = time.perf_counter() - t0

    for beta_val, penalty_label in ((0.0, 'no_penalty'), (beta, 'with_penalty')):
        for alpha in alphas:
            t1 = time.perf_counter()
            best_sol, _ = _grasp_loop(br_CWS_c, a_nodes, sav, vehicle_cap, depot, n_iter,
                                      None, hourly_probs=hourly_probs, d_fb=d_fb,
                                      alpha=alpha, beta=beta_val, departure_h=dep_h, p=p_bias)
            loop_s = time.perf_counter() - t1

            dist_km = round(best_sol.cost / 1000, 4)
            row = {'alpha': alpha, 'beta': beta_val, 'penalty': penalty_label,
                  'dist_km': dist_km, 'n_routes': len(best_sol.routes),
                  'build_s': round(build_s, 3), 'loop_s': round(loop_s, 3)}
            row.update(_fallback_kpis(best_sol, orders, locker_cap, dist_matrix, dist_km))
            rows.append(row)
            if verbose:
                eff = f"  eff={row['effective_km']:.3f}km" if 'effective_km' in row else ''
                print(f"    alpha={alpha:<5} [{penalty_label:<12}] -> {dist_km:.3f} km{eff} "
                      f"(build {build_s:.2f}s, loop {loop_s:.2f}s)")
    return rows


# =============================================================================
# CSV OUTPUT
# =============================================================================

def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        print(f"  (no rows -- skipping {path})")
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    if 'method' in fieldnames:
        fieldnames = ['method'] + [f for f in fieldnames if f != 'method']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {path}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description='OFAT parameter sensitivity sweep for one concrete instance'
    )
    p.add_argument('instance', help='Instance .txt file')
    p.add_argument('--model-a', default=None)
    p.add_argument('--model-b', default=None)
    p.add_argument('--model-c', default=None)
    p.add_argument('--sweep', default='beta,p_bias,departure_h,n_iter',
                   help='Comma-separated subset of {beta,p_bias,departure_h,n_iter,alpha_c} '
                        '(default: all except alpha_c -- add it explicitly)')
    p.add_argument('--betas',        default=','.join(map(str, DEFAULT_BETAS)))
    p.add_argument('--p_biases',     default=','.join(map(str, DEFAULT_P_BIASES)))
    p.add_argument('--departure_hs', default=','.join(map(str, DEFAULT_DEPARTURE_HS)))
    p.add_argument('--alphas',       default=','.join(map(str, DEFAULT_ALPHAS)),
                   help='alpha_c sweep values (0=pure time, 1=pure distance)')
    p.add_argument('--beta',    type=float, default=BETA_DEFAULT,
                   help='Baseline beta held fixed in non-beta sweeps')
    p.add_argument('--p_bias',  type=float, default=P_BIAS,
                   help='Baseline p_bias held fixed in non-p_bias sweeps')
    p.add_argument('--n_iter',  type=int, default=25,
                   help='GRASP iterations per cell for beta/p_bias/departure_h sweeps '
                        '(default 25 -- lighter than the usual n_iter=100; '
                        're-run the winning config at full n_iter via run_comparison.py)')
    p.add_argument('--n_iter_max', type=int, default=100,
                   help='Max iterations for the n_iter convergence-curve sweep')
    p.add_argument('--checkpoint_every', type=int, default=5)
    p.add_argument('--out_dir', default=None)
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    params, nodes, dist_matrix, orders, locker_cap = read_full_instance(args.instance)
    vehicle_cap    = params.capacity
    depot          = nodes[0]
    dep_h_baseline = params.departure
    out_dir        = args.out_dir or RESULTS_DIR
    sweeps         = {s.strip() for s in args.sweep.split(',') if s.strip()}

    model_a_path = _resolve_model_path(args.model_a, MODEL_A_CANDIDATES, 'A')
    model_b_path = _resolve_model_path(args.model_b, MODEL_B_CANDIDATES, 'B')
    model_c_path = _resolve_model_path(args.model_c, MODEL_C_CANDIDATES, 'C')
    bundle_a = _try_load(model_a_path, 'Model A')
    bundle_b = _try_load(model_b_path, 'Model B')
    bundle_c = _try_load(model_c_path, 'Model C')

    print(f"\n{'='*70}")
    print(f"  SENSITIVITY — {params.name}")
    print(f"  baseline: beta={args.beta}  p_bias={args.p_bias}  "
          f"departure_h={dep_h_baseline}")
    print(f"  sweep n_iter (per cell)={args.n_iter}  n_iter_max (curve)={args.n_iter_max}")
    print(f"{'='*70}\n")

    if 'beta' in sweeps:
        if bundle_a or bundle_b:
            betas = [float(x) for x in args.betas.split(',')]
            print(f"[sweep] beta = {betas}")
            rows = sweep_beta(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                              dep_h_baseline, bundle_a, bundle_b, betas, args.p_bias,
                              args.n_iter, args.verbose)
            _write_csv(rows, os.path.join(out_dir, 'sensitivity_beta.csv'))
        else:
            print('[sweep] beta -- skipped, no ML model available')

    if 'p_bias' in sweeps:
        p_biases = [float(x) for x in args.p_biases.split(',')]
        print(f"[sweep] p_bias = {p_biases}")
        rows = sweep_p_bias(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                            dep_h_baseline, bundle_a, bundle_b, args.beta, p_biases,
                            args.n_iter, args.verbose)
        _write_csv(rows, os.path.join(out_dir, 'sensitivity_p_bias.csv'))

    if 'departure_h' in sweeps:
        departure_hs = [float(x) for x in args.departure_hs.split(',')]
        print(f"[sweep] departure_h = {departure_hs}")
        rows = sweep_departure_h(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                                 bundle_a, bundle_b, args.beta, args.p_bias,
                                 departure_hs, args.n_iter, args.verbose)
        _write_csv(rows, os.path.join(out_dir, 'sensitivity_departure_h.csv'))

    if 'n_iter' in sweeps:
        print(f"[sweep] n_iter convergence curve, max={args.n_iter_max}, "
              f"checkpoint every {args.checkpoint_every}")
        rows = sweep_n_iter(nodes, dist_matrix, vehicle_cap, depot, locker_cap,
                            dep_h_baseline, bundle_a, bundle_b, args.beta, args.p_bias,
                            args.n_iter_max, args.checkpoint_every, args.verbose)
        _write_csv(rows, os.path.join(out_dir, 'sensitivity_n_iter.csv'))

    if 'alpha_c' in sweeps:
        # alpha_c always runs, even without a trained Option-C model (beta=0
        # cells then equal the with-penalty cells, which is itself an
        # informative "no ML available yet" baseline).
        alphas = [float(x) for x in args.alphas.split(',')]
        print(f"[sweep] alpha_c = {alphas}  (Option C: distance/time blend, "
              f"{'with' if bundle_c else 'without'} trained ML penalty)")
        rows = sweep_alpha_c(nodes, dist_matrix, vehicle_cap, depot, locker_cap, orders,
                             dep_h_baseline, bundle_c, alphas, args.beta, args.p_bias,
                             args.n_iter, args.verbose)
        _write_csv(rows, os.path.join(out_dir, 'sensitivity_alpha_c.csv'))

    print(f"\n{'='*70}")
    print("  Done. Re-run the winning configuration at full n_iter=100 for the")
    print("  final thesis comparison table (see run_comparison.py).")
    print('='*70)


if __name__ == '__main__':
    main()
