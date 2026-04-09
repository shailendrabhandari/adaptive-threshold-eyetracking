"""
markov.py
---------
Stationarity and first-order Markov adequacy diagnostics for the K-ratio framework.

Implements the two checks from Appendix A of the paper:
  1. Blockwise K-ratio stability (transition probabilities across temporal blocks)
  2. T^k deviation: empirical k-step transition matrix vs. first-order Markov prediction
"""

import numpy as np
import matplotlib.pyplot as plt



def _to_binary(classifier):
    if isinstance(classifier[0], str):
        return np.array([1 if c == "saccade" else 0
                         for c in classifier], dtype=int)
    arr = np.asarray(classifier, dtype=int)
    # EyeLink labels: 1=fixation, 2=saccade -> remap to 0/1
    if arr.max() > 1:
        return (arr == 2).astype(int)
    return arr



def blockwise_kratio_stability(classifier, n_blocks=12):
    labels = _to_binary(classifier)
    L      = len(labels)
    block_size = L // n_blocks

    block_ns, block_pFS, block_pSF, block_kratio = [], [], [], []

    for b in range(n_blocks):
        start = b * block_size
        end   = start + block_size if b < n_blocks - 1 else L
        blk   = labels[start:end]
        Lb    = len(blk)
        if Lb < 2:
            continue

        nS = blk.mean()
        block_ns.append(nS)

        prev = blk[:-1];  nxt = blk[1:]

        # Paper Eq. 10: use sum/L (not np.mean which divides by L-1)
        pFS = np.sum((prev == 0) & (nxt == 1)) / Lb   # n_{F->S}
        pSF = np.sum((prev == 1) & (nxt == 0)) / Lb   # n_{S->F}
        block_pFS.append(pFS)
        block_pSF.append(pSF)

        p_ind = nS * (1 - nS)
        kr    = (pFS / p_ind) if p_ind > 0 else np.nan
        block_kratio.append(kr)

    return dict(
        block_ns    = np.array(block_ns),
        block_pFS   = np.array(block_pFS),
        block_pSF   = np.array(block_pSF),
        block_kratio= np.array(block_kratio),
    )


def markov_tk_deviation(classifier, max_lag=40):
    labels = _to_binary(classifier)

    # Empirical 1-step transition matrix T (row-stochastic)
    T    = np.zeros((2, 2), dtype=float)
    prev = labels[:-1];  nxt = labels[1:]
    for s in range(2):
        mask = prev == s
        if mask.sum() > 0:
            T[s, 0] = np.sum(nxt[mask] == 0) / mask.sum()
            T[s, 1] = np.sum(nxt[mask] == 1) / mask.sum()

    lags       = np.arange(1, max_lag + 1)
    deviations = np.empty(max_lag, dtype=float)

    for i, k in enumerate(lags):
        # Empirical k-step transition matrix
        T_emp_k = np.zeros((2, 2), dtype=float)
        if k < len(labels):
            prev_k = labels[:-k];  nxt_k = labels[k:]
            for s in range(2):
                mask = prev_k == s
                if mask.sum() > 0:
                    T_emp_k[s, 0] = np.sum(nxt_k[mask] == 0) / mask.sum()
                    T_emp_k[s, 1] = np.sum(nxt_k[mask] == 1) / mask.sum()

        # First-order Markov prediction: T^k
        T_pred_k      = np.linalg.matrix_power(T, k)
        deviations[i] = np.mean(np.abs(T_emp_k - T_pred_k))

    return lags, deviations


def plot_markov_diagnostics(classifier, n_blocks=12, max_lag=40,
                             save_path=None, title_suffix=""):
    bw          = blockwise_kratio_stability(classifier, n_blocks=n_blocks)
    lags, devs  = markov_tk_deviation(classifier, max_lag=max_lag)

    fig, axes = plt.subplots(3, 1, figsize=(8, 4))
    block_idx = np.arange(len(bw['block_ns']))

    # ---- (a) Blockwise nS and transitions ----
    ax1  = axes[0]
    ax1r = ax1.twinx()
    ax1.plot(block_idx,  bw['block_ns'],
             'b-o', linewidth=1.5, markersize=5, label=r'$n_S$')
    ax1r.plot(block_idx, bw['block_pFS'],
              color='orange', marker='o', linewidth=1.5, markersize=5,
              label=r'$p(F\to S)$')
    ax1r.plot(block_idx, bw['block_pSF'],
              'g-s', linewidth=1.5, markersize=5, label=r'$p(S\to F)$')
    ax1.set_ylabel(r'$n_S$', fontsize=10, fontweight='bold')
    ax1r.set_ylabel('Transition prob.', fontsize=10, fontweight='bold')
    ax1.set_title(f'(a) Blockwise event fraction and transitions {title_suffix}',
                  fontsize=10, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1r.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax1.tick_params(axis='both', labelsize=8)

    # ---- (b) Blockwise K-ratio ----
    ax2 = axes[1]
    ax2.plot(block_idx, bw['block_kratio'],
             'b-o', linewidth=1.5, markersize=5)
    ax2.set_ylabel('K-ratio', fontsize=10, fontweight='bold')
    ax2.set_title('(b) Blockwise K-ratio stability', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=8)

    # ---- (c) Markov T^k deviation ----
    ax3 = axes[2]
    ax3.plot(lags, devs, 'b-o', linewidth=1.5, markersize=4)
    ax3.set_xlabel('Lag k (steps)', fontsize=10, fontweight='bold')
    ax3.set_ylabel(r'$\|T_{\rm emp}(k) - T^k\|$', fontsize=10, fontweight='bold')
    ax3.set_title(r'(c) First-order Markov adequacy', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', labelsize=8)

    plt.tight_layout(pad=1.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    print(f"\nBlockwise K-ratio : mean={np.nanmean(bw['block_kratio']):.4f}  "
          f"std={np.nanstd(bw['block_kratio']):.4f}  "
          f"range=[{np.nanmin(bw['block_kratio']):.4f}, "
          f"{np.nanmax(bw['block_kratio']):.4f}]")
    print(f"T^k deviation at k=2 : {devs[1]:.2e}")
    print(f"T^k deviation at k=5 : {devs[4]:.2e}")

    return bw, lags, devs