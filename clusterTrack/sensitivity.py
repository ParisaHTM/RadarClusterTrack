# clusterTrack/sensitivity.py
import os
import sys
import itertools
import argparse
from pathlib import Path
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Ensure repo-relative imports work
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.append(str(THIS_DIR))              # clusterTrack
sys.path.append(str(REPO_ROOT / 'precision'))  # precision

import config as cfg
from dataProcessor import RadarDataProcessor
from utilsPrecision import compute_scene_accuracy  # uses data_dir override



def avg_clusters_per_entry(pkl_path: Path) -> float:
    if not pkl_path.exists():
        return float('nan')
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if not isinstance(data, list) or not data:
        return 0.0
    counts = []
    for entry in data:
        if isinstance(entry, dict):
            if 'num_clusters' in entry and isinstance(entry['num_clusters'], (int, float)):
                try:
                    counts.append(int(entry['num_clusters']))
                    continue
                except Exception:
                    pass
            if 'cluster_ids' in entry and isinstance(entry['cluster_ids'], list):
                counts.append(len(entry['cluster_ids']))
                continue
            if 'clusters' in entry and isinstance(entry['clusters'], list):
                counts.append(len(entry['clusters']))
                continue
        counts.append(0)
    return sum(counts) / max(len(counts), 1)

def run_one_combo(scene: str, data_path: str, v: float, p: float, d: float, Freq: int, sigma: float) -> dict:
    # Override thresholds for this run
    cfg.CLUSTERING_PARAMS['velocity_threshold'] = float(v)
    cfg.CLUSTERING_PARAMS['position_threshold'] = float(p)
    cfg.TRACKING_PARAMS['max_distance_threshold'] = float(d)

    out_dir = REPO_ROOT / f"out_sens_cca_v{v}_p{p}_d{d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process scene with these params, saving pickle per scene in out_dir
    proc = RadarDataProcessor(
        data_path=data_path,
        config={'pickle_save_root': str(out_dir)},
        verbose_class=False
    )
    proc.process_scenes(scene_names=[scene])

    # Evaluate accuracy (MATC-Precision-like) and avg cluster count
    pkl = out_dir / f"{scene}_clustering_data.pkl"
    acc_res = compute_scene_accuracy(scene, assigned_consecutive_frames=Freq, sigma=sigma, data_dir=str(out_dir), verbose=False)
    accuracy = float(acc_res['accuracy']) if acc_res and 'accuracy' in acc_res else 0.0
    avg_clusters = avg_clusters_per_entry(pkl)
    return {
        'v': v, 'p': p, 'd': d,
        'accuracy': accuracy,
        'avg_clusters_per_entry': avg_clusters,
        'output_dir': str(out_dir)
    }

def plot_1d_sensitivity(results, scene, baseline=(0.5, 2.0, 5.0)):
    v_base, p_base, d_base = baseline

    def series(var_name, fixed_pairs):
        vals = sorted({r[var_name] for r in results
                       if all(abs(r[k] - v) < 1e-9 for k, v in fixed_pairs.items())})
        accs, clus = [], []
        for val in vals:
            match = next((r for r in results
                          if abs(r[var_name]-val) < 1e-9 and
                             all(abs(r[k]-v) < 1e-9 for k, v in fixed_pairs.items())), None)
            if match is not None:
                accs.append(match['accuracy'])
                clus.append(match['avg_clusters_per_entry'])
        return vals, accs, clus

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # Vary velocity, fix position & distance
    vals_v, acc_v, clu_v = series('v', {'p': p_base, 'd': d_base})
    ax = axes[0]
    ax.plot(vals_v, acc_v, marker='o', color='#1f77b4')
    ax.set_xlabel(r'$\tau_v$ (m/s)', fontsize=16)
    ax.set_ylabel('Accuracy')
    ax2 = ax.twinx()
    ax2.plot(vals_v, clu_v, marker='x', color='gray', linestyle='--')
    ax2.set_ylabel('Avg clusters ', fontsize=16, fontweight='bold')

    # Vary position, fix velocity & distance
    vals_p, acc_p, clu_p = series('p', {'v': v_base, 'd': d_base})
    ax = axes[1]
    ax.plot(vals_p, acc_p, marker='o', color='#2ca02c')
    ax.set_xlabel(r'$\tau_t$ (m)', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot(vals_p, clu_p, marker='x', color='gray', linestyle='--')

    # Vary distance, fix velocity & position
    vals_d, acc_d, clu_d = series('d', {'v': v_base, 'p': p_base})
    ax = axes[2]
    ax.plot(vals_d, acc_d, marker='o', color='#d62728')
    ax.set_xlabel(r'$\tau_d$ (m)', fontsize=16)
    ax2 = ax.twinx()
    ax2.plot(vals_d, clu_d, marker='x', color='gray', linestyle='--')

    fig.suptitle(f'Sensitivity (scene={scene}, baseline v={v_base}, p={p_base}, d={d_base})', y=1.02)
    out_path = REPO_ROOT / f"sensitivity_{scene}.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved plot -> {out_path}")

def plot_heatmaps_by_v(results, scene, v_grid, p_grid, d_grid,
                       metric='accuracy', cmap='viridis',
                       zlabel='Accuracy', file_suffix='accuracy'):
    ncols = 4
    nrows = int(np.ceil(len(v_grid) / ncols))
    fig_w = 4.8 * ncols
    fig_h = 4.2 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    vals = [r.get(metric) for r in results if r.get(metric) is not None]
    vals = [float(v) for v in vals if not (isinstance(v, float) and np.isnan(v))]
    vmin = np.nanmin(vals) if len(vals) > 0 else 0.0
    vmax = np.nanmax(vals) if len(vals) > 0 else 1.0

    for idx, v in enumerate(v_grid):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        mat = np.full((len(d_grid), len(p_grid)), np.nan, dtype=float)
        for yi, d in enumerate(d_grid):
            for xi, p in enumerate(p_grid):
                m = next((rr for rr in results
                          if np.isclose(rr['v'], v) and np.isclose(rr['p'], p) and np.isclose(rr['d'], d)), None)
                if m is not None and m.get(metric) is not None:
                    mat[yi, xi] = float(m[metric])

        sns.heatmap(mat, ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, cbar=False,
                    xticklabels=[str(x) for x in p_grid],
                    yticklabels=[str(y) for y in d_grid],
                    annot=False)
        ax.set_title(rf'$\tau_v$ = {v}', fontsize=24)
        ax.set_xlabel(r'$\tau_t$ (m)', fontsize=18)
        if c == 0:
            ax.set_ylabel(r'$\tau_d$ (m)', fontsize=18)
        ax.tick_params(axis='x', labelrotation=45, labelsize=15)
        ax.tick_params(axis='y', labelrotation=0, labelsize=15)

    # Hide unused axes
    for j in range(len(v_grid), nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')

    # Shared colorbar on right, with extra margin reserved
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.94])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label(zlabel, fontsize=18, fontweight='bold')
    # fig.suptitle(f'{zlabel} heatmaps by v (scene={scene})', y=0.995, fontsize=16, fontweight='bold')

    out = REPO_ROOT / f'sensitivity_heatmaps_{file_suffix}_{scene}.png'
    fig.savefig(out, dpi=250)
    plt.close(fig)
    print(f'Saved plot -> {out}')

def main():
    parser = argparse.ArgumentParser("Brief sensitivity analysis over v/p/d thresholds")
    parser.add_argument('--scene', type=str, default=cfg.SCENE_CONFIG.get('target_scene', 'scene-0061'))
    parser.add_argument('--data-path', type=str, default=cfg.DEFAULT_PATHS['nuscenes_dataroot'])
    parser.add_argument('--F', type=int, default=3, help='Consecutive-frame requirement for accuracy metric')
    parser.add_argument('--sigma', type=float, default=5.0, help='RCS sigma for accuracy metric')
    parser.add_argument('--plot', action='store_true', help='Save a simple 1D sensitivity plot')
    parser.add_argument('--skip-existing', action='store_true', help='Skip processing if pickle already exists')
    parser.add_argument('--plot2d', action='store_true', help='Save heatmaps by v over p (x) and d (y)')
    parser.add_argument('--plot2d-clusters', action='store_true', help='Save heatmaps by v over p (x) and d (y) for avg clusters')
    args = parser.parse_args()

    # Candidate grids from your config comments
    v_grid = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 10]
    p_grid = [1.0, 2.0, 3.0, 4.0, 5.0, 20]
    d_grid = [2.0, 3.0, 5.0, 7.0, 10.0, 30]

    results = []
    for v, p, d in itertools.product(v_grid, p_grid, d_grid):
        out_dir = REPO_ROOT / f"out_sens_cca_v{v}_p{p}_d{d}"
        pkl = out_dir / f"{args.scene}_clustering_data.pkl"

        if args.skip_existing and pkl.exists():
            # Evaluate only
            acc_res = compute_scene_accuracy(args.scene, assigned_consecutive_frames=args.F, sigma=args.sigma, data_dir=str(out_dir), verbose=False)
            accuracy = float(acc_res['accuracy']) if acc_res and 'accuracy' in acc_res else 0.0
            avg_clusters = avg_clusters_per_entry(pkl)
            results.append({'v': v, 'p': p, 'd': d, 'accuracy': accuracy, 'avg_clusters_per_entry': avg_clusters, 'output_dir': str(out_dir)})
            continue

        res = run_one_combo(args.scene, args.data_path, v, p, d, args.F, args.sigma)
        results.append(res)

        print(f"v={v:>3}, p={p:>3}, d={d:>4} -> acc={res['accuracy']:.4f}, avg_clusters={res['avg_clusters_per_entry']:.2f}")

    # Rank by accuracy (desc), then prefer fewer clusters as tie-breaker
    results.sort(key=lambda r: (-r['accuracy'], r['avg_clusters_per_entry']))

    print("\nTop 5 configs by accuracy:")
    for r in results[:5]:
        print(f"v={r['v']}, p={r['p']}, d={r['d']} | acc={r['accuracy']:.4f} | avg_clusters={r['avg_clusters_per_entry']:.2f} | dir={r['output_dir']}")

    # Also explicitly print your proposed config (0.5, 2.0, 5.0)
    chosen = next((r for r in results if r['v']==0.5 and r['p']==2.0 and r['d']==5.0), None)
    if chosen:
        print(f"\nYour pick (v=0.5, p=2.0, d=5.0): acc={chosen['accuracy']:.4f}, avg_clusters={chosen['avg_clusters_per_entry']:.2f}, dir={chosen['output_dir']}")
    else:
        print("\nYour pick (0.5, 2.0, 5.0) was not evaluated or did not produce output.")
    
    if args.plot:
        plot_1d_sensitivity(results, args.scene, baseline=(0.5, 2.0, 5.0))
    
    if args.plot2d:
        plot_heatmaps_by_v(
            results, args.scene,
            v_grid=[0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 10.0],
            p_grid=[1.0, 2.0, 3.0, 4.0, 5.0, 20.0],
            d_grid=[2.0, 3.0, 5.0, 7.0, 10.0, 30.0],
            metric='accuracy', cmap='viridis', zlabel='Accuracy', file_suffix='accuracy'
        )
    
    
    if args.plot2d_clusters:
        plot_heatmaps_by_v(
            results, args.scene,
            v_grid=[0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 10.0],
            p_grid=[1.0, 2.0, 3.0, 4.0, 5.0, 20.0],
            d_grid=[2.0, 3.0, 5.0, 7.0, 10.0, 30.0],
            metric='avg_clusters_per_entry', cmap='magma', zlabel='Avg clusters', file_suffix='clusters'
        )

if __name__ == "__main__":
    main()