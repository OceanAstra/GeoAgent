#!/usr/bin/env python3
"""GeoAgent -- RQ3 six-class TreeSHAP figures.

Figure A  shap_class_feature_heatmap_latest.{pdf,png}
          6 accident-cause classes x 13 features, mean absolute TreeSHAP,
          averaged over CatBoost seeds 42/123/456, computed from the verified summary CSV.

Figure B  shap_multi_panel_latest.{pdf,png}
          Six panels, one per class: instance-level signed TreeSHAP of that class
          (rows = 13 features, columns = the 6,417 dev-confirm records belonging to
          the class), with the class raw margin f(x) above each panel.
          Recomputed from the three frozen CatBoost models -- no training.

Both figures:
  * read colour stops and plot data from explicit paths (no hard-coded results);
  * label colour semantics from the quantity actually plotted;
  * write what they plotted next to the PDF so the numbers can be audited.

Typical use
-----------
    python figures/shap_reproduction/scripts/generate_shap_figures.py --dpi 200
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
from matplotlib import font_manager

# --------------------------------------------------------------------------- constants
MAIN_13 = ['PIPE_DIAMETER', 'PIPE_WALL_THICKNESS', 'PIPE_SMYS', 'PIPE_SPECIFICATION',
           'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE', 'ACCIDENT_PSIG', 'MOP_PSIG',
           'INTERNAL_INSPECTION_IND', 'PIPE_AGE_AT_ACCIDENT', 'MANUFACTURE_AGE_AT_ACCIDENT',
           'PRESSURE_RATIO']
CATEGORICAL = ['PIPE_SPECIFICATION', 'PIPE_SEAM_TYPE', 'PIPE_COATING_TYPE', 'PIPE_TYPE']
SIX = ['CORROSION', 'EQUIPMENT_FAILURE', 'EXTERNAL_DAMAGE', 'MATERIAL_DEFECT',
       'OPERATIONAL', 'OTHER']
SEEDS = (42, 123, 456)

DISPLAY = {'CORROSION': 'Corrosion', 'EQUIPMENT_FAILURE': 'Equipment Failure',
           'EXTERNAL_DAMAGE': 'External Damage', 'MATERIAL_DEFECT': 'Material Defect',
           'OPERATIONAL': 'Operational', 'OTHER': 'Other Causes'}

# Abbreviations are fixed here, printed on the axis, and exported to
# ABBREVIATION_MAP.csv so every tick maps back to exactly one of the 13 fields.
ABBR = {
    'PIPE_COATING_TYPE': 'CT',
    'PIPE_TYPE': 'PT',
    'PIPE_SEAM_TYPE': 'ST',
    'ACCIDENT_PSIG': 'AP',
    'PRESSURE_RATIO': 'PR',
    'PIPE_SPECIFICATION': 'PS',
    'MOP_PSIG': 'MOP',
    'PIPE_AGE_AT_ACCIDENT': 'PA',
    'PIPE_DIAMETER': 'D',
    'PIPE_WALL_THICKNESS': 'WT',
    'PIPE_SMYS': 'SMYS',
    'INTERNAL_INSPECTION_IND': 'II',
    'MANUFACTURE_AGE_AT_ACCIDENT': 'MA',
}
ABBR_LONG = {
    'CT': 'Coating type', 'PT': 'Pipe type', 'ST': 'Seam type', 'AP': 'Accident pressure',
    'PR': 'Pressure ratio', 'PS': 'Pipe specification', 'MOP': 'Max. operating pressure',
    'PA': 'Pipe age', 'D': 'Diameter', 'WT': 'Wall thickness', 'SMYS': 'Specified min. yield strength',
    'II': 'Internal inspection', 'MA': 'Manufacture age',
}

# Short axis labels for figure A, where the full tick name must stay readable
# under 45-degree rotation at a 190 mm figure width.
SHORT_NAME = {
    'PIPE_COATING_TYPE': 'Pipe coating\ntype',
    'PIPE_TYPE': 'Pipe type',
    'PIPE_SEAM_TYPE': 'Seam type',
    'ACCIDENT_PSIG': 'Accident\npressure',
    'PRESSURE_RATIO': 'Pressure\nratio',
    'PIPE_SPECIFICATION': 'Pipe\nspecification',
    'MOP_PSIG': 'Max. operating\npressure',
    'PIPE_AGE_AT_ACCIDENT': 'Pipe age',
    'PIPE_DIAMETER': 'Diameter',
    'PIPE_WALL_THICKNESS': 'Wall\nthickness',
    'PIPE_SMYS': 'SMYS',
    'INTERNAL_INSPECTION_IND': 'Internal\ninspection',
    'MANUFACTURE_AGE_AT_ACCIDENT': 'Manufacture\nage',
}


# --------------------------------------------------------------------------- style
def set_style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = 'STIXGeneral' if 'STIXGeneral' in have else 'DejaVu Serif'
    plt.rcParams.update({
        'font.family': fam,
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42,          # TrueType, not Type 3 -- required by most publishers
        'ps.fonttype': 42,
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'figure.dpi': 120,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
    })
    return fam


class PowerNorm(Normalize):
    """Monotone, sign-preserving power scaling to [0, 1].

    f(x) = 0.5 * (x/a)**g + 0.5       for x >= 0, symmetric for x < 0
    Pure monotone map: no value is reordered, no value is dropped, and the
    inverse is used to place the colourbar ticks at their true values.
    """

    def __init__(self, a, gamma=0.5, signed=True, clip=True):
        self.a, self.gamma, self.signed, self.clip_to = float(a), float(gamma), signed, clip
        super().__init__(vmin=-a if signed else 0.0, vmax=a, clip=False)

    def __call__(self, value, clip=None):
        v = np.ma.asarray(value, dtype=float)
        if self.clip_to:
            v = np.ma.clip(v, self.vmin, self.vmax)
        if self.signed:
            return 0.5 * np.sign(v) * (np.abs(v) / self.a) ** self.gamma + 0.5
        return (np.abs(v) / self.a) ** self.gamma

    def inverse(self, value):
        v = np.ma.asarray(value, dtype=float)
        if self.signed:
            z = 2.0 * (v - 0.5)
            return self.a * np.sign(z) * np.abs(z) ** (1.0 / self.gamma)
        return self.a * np.asarray(v) ** (1.0 / self.gamma)


def build_cmap(stops):
    """stops: [(value, '#rrggbb'), ...] -> LinearSegmentedColormap on [0,1]."""
    xs = [s['value'] for s in stops]
    cs = [s['hex'] for s in stops]
    return LinearSegmentedColormap.from_list('ref', list(zip(xs, cs)), N=512)


def ink_for(rgba, dark='#101010', light='#FFFFFF'):
    """Pick the higher-contrast text colour for a cell background."""
    r, g, b = rgba[:3]

    def lin(c):
        c = c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        return c
    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    # contrast against white and against near-black
    cw = 1.05 / (lum + 0.05)
    cb = (lum + 0.05) / 0.05
    return dark if cw > cb else light


# --------------------------------------------------------------------------- data
def catboost_frame(df):
    """Same construction as g2_common.feature_frame (frozen protocol).

    Categorical NaN -> the fixed token 'NOT_RECORDED', cast to str.
    Numeric NaN is passed through to CatBoost unchanged: the frozen models were
    fit on native NaN handling, not on median-imputed values. Imputing here
    shifts probabilities by up to 0.84 and is a reproduction error.
    """
    X = df[MAIN_13].copy()
    for c in CATEGORICAL:
        X[c] = X[c].fillna('NOT_RECORDED').astype(str)
    for c in MAIN_13:
        if c not in CATEGORICAL:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    return X


def load_or_compute_shap(args):
    """Return (shap_mean, base_mean, y, prob, row_id, X) for the 6,417 dev-confirm rows."""
    man = pd.read_csv(os.path.join(
        args.repro, 'models_and_configs/geoad_materials/data/data_row_manifest_development_only.csv'),
        low_memory=False)
    dc = man[man.split == 'dev_confirm'].reset_index(drop=True)
    X = catboost_frame(dc)
    y = dc.y.to_numpy()
    row_id = dc.row_id.to_numpy()

    if args.shap_cache and os.path.exists(args.shap_cache):
        z = np.load(args.shap_cache, allow_pickle=False)
        assert np.array_equal(z['y'], y), 'cached SHAP array does not match the manifest order'
        return (z['shap_mean'].astype(np.float64), z['base_mean'].astype(np.float64),
                y, z['prob_mean'].astype(np.float64), row_id, X)

    from catboost import CatBoostClassifier, Pool
    import shap

    cat_idx = [MAIN_13.index(c) for c in CATEGORICAL]
    pool = Pool(X, cat_features=cat_idx)
    W = os.path.join(args.repro, 'models_and_configs/geoad_materials/weights')

    shaps, bases, probs = [], [], []
    for s in SEEDS:
        m = CatBoostClassifier()
        m.load_model(os.path.join(W, f'cb_seed{s}.cbm'))
        ex = shap.TreeExplainer(m)
        sv = np.asarray(ex.shap_values(pool), dtype=np.float64)      # (n, 13, 6)
        bases.append(np.ravel(np.asarray(ex.expected_value, dtype=np.float64)))
        raw = np.asarray(m.predict(pool, prediction_type='RawFormulaVal'), dtype=np.float64)
        resid = np.abs(sv.sum(axis=1) + bases[-1] - raw).max()
        assert resid < 1e-6, f'additivity broken for seed {s}: {resid}'
        shaps.append(sv)
        probs.append(m.predict_proba(pool))

    shap_mean = np.mean(shaps, axis=0)
    base_mean = np.mean(bases, axis=0)
    prob_mean = np.mean(probs, axis=0)     # == A_ensemble.csv, verified separately

    if args.shap_cache:
        np.savez_compressed(args.shap_cache, shap_mean=shap_mean.astype(np.float32),
                            shap_seed42=shaps[0].astype(np.float32),
                            shap_seed123=shaps[1].astype(np.float32),
                            shap_seed456=shaps[2].astype(np.float32),
                            base_mean=base_mean, y=y.astype(np.int16),
                            prob_mean=prob_mean.astype(np.float32), row_id=row_id.astype(str),
                            features=np.array(MAIN_13), classes=np.array(SIX))
    return shap_mean, base_mean, y, prob_mean, row_id, X


def resolve_abbr_order(features):
    """Column order = global importance rank of the shipped global CSV, so that
    the heatmap reads left-to-right by decreasing overall importance, matching
    the ordering already used by the manuscript's global-importance figure."""
    return list(features)


# --------------------------------------------------------------------------- figure A
def figure_a(df_cls, cls_order, feat_order, cmap, outbase, gamma, dpi,
             explain_text):
    """df_cls: DataFrame index=class, columns=feature, values=mean |TreeSHAP| (3-seed mean)."""
    M = df_cls.loc[cls_order, feat_order].to_numpy()
    vmax = float(M.max())
    norm = PowerNorm(vmax, gamma=gamma, signed=False)

    # canvas: 190 mm text width of an Elsevier double-column page
    fig = plt.figure(figsize=(7.48, 4.02))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.026], wspace=0.020,
                          left=0.135, right=0.905, top=0.972, bottom=0.412)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    nR, nC = M.shape
    ax.imshow(M, cmap=cmap, norm=norm, aspect='auto',
              extent=(-0.5, nC - 0.5, nR - 0.5, -0.5), interpolation='nearest')
    ax.set_xlim(-0.5, nC - 0.5)
    ax.set_ylim(nR - 0.5, -0.5)

    # cell values: 3 decimals, real mean absolute TreeSHAP in raw-score units
    for i in range(nR):
        for j in range(nC):
            bg = cmap(norm(M[i, j]))
            ax.text(j, i, f'{M[i, j]:.3f}', ha='center', va='center',
                    fontsize=7.2, color=ink_for(bg))

    ax.set_xticks(range(nC))
    ax.set_xticklabels([SHORT_NAME[f] for f in feat_order], rotation=42, ha='right',
                       rotation_mode='anchor', fontsize=7.0, linespacing=0.95)
    ax.set_yticks(range(nR))
    ax.set_yticklabels([DISPLAY[c] for c in cls_order], fontsize=7.8)
    ax.tick_params(axis='both', length=0, pad=3)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)

    # colourbar
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = [t for t in (0.0, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00, vmax) if t <= vmax + 1e-9]
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([f'{t:.2f}' for t in ticks], fontsize=6.6)
    cb.ax.tick_params(length=2, width=0.5, pad=2)
    cb.outline.set_linewidth(0.6)
    cb.ax.set_title('High', fontsize=7.2, pad=4)
    cb.ax.text(0.5, -0.035, 'Low', transform=cb.ax.transAxes, ha='center', va='top',
               fontsize=7.2)
    cb.set_label(f'Mean $|$TreeSHAP$|$ (raw-score units)\n'
                 f'colour scaled by $x^{{{gamma:g}}}$', fontsize=7.2, labelpad=7)

    # No abbreviation key is needed here: the x-axis carries the full field name
    # (wrapped), so every column is unambiguous. ABBREVIATION_MAP.csv is still
    # exported for the paneled figure, which does use abbreviations.
    for i, line in enumerate(explain_text.split('\n')):
        fig.text(0.5, 0.138 - i * 0.048, line, fontsize=7.0, ha='center', va='top')

    for ext in ('pdf', 'png'):
        fig.savefig(f'{outbase}.{ext}', dpi=dpi if ext == 'png' else None,
                    facecolor='white')
    plt.close(fig)
    return vmax


# --------------------------------------------------------------------------- figure B
def figure_b(shap_mean, base_mean, y, outbase, cmap, gamma, dpi, masks_order,
             instance_order_note):
    """Six panels of within-class instance-level signed SHAP, plus f(x) strips."""
    nC = len(SIX)
    # symmetric colour limit: the largest |phi| actually plotted
    vals = np.concatenate([shap_mean[y == i][:, :, i].ravel() for i in range(nC)])
    a = float(np.abs(vals).max())
    a = float(np.ceil(a * 10) / 10)          # readable round limit >= data range
    norm = PowerNorm(a, gamma=gamma, signed=True)

    feat_order = list(masks_order)           # global-importance order
    fidx = [MAIN_13.index(f) for f in feat_order]

    fig = plt.figure(figsize=(7.48, 5.55))
    outer = fig.add_gridspec(2, 3, left=0.075, right=0.935, top=0.925, bottom=0.235,
                             wspace=0.14, hspace=0.30)

    panels = []
    for k, ci in enumerate(range(nC)):
        r, c = divmod(k, 3)
        gsub = outer[r, c].subgridspec(2, 1, height_ratios=[0.20, 0.80], hspace=0.045)
        axf = fig.add_subplot(gsub[0])
        ax = fig.add_subplot(gsub[1], sharex=axf)
        panels.append((ax, axf))

        m = np.where(y == ci)[0]
        phi = shap_mean[m][:, fidx, ci]      # (n_i, 13)
        f_x = base_mean[ci] + phi.sum(axis=1)

        ax.imshow(phi.T, cmap=cmap, norm=norm, aspect='auto',
                  extent=(-0.5, len(m) - 0.5, len(feat_order) - 0.5, -0.5),
                  interpolation='nearest', rasterized=True)
        ax.set_xlim(-0.5, len(m) - 0.5)
        ax.set_ylim(len(feat_order) - 0.5, -0.5)
        ax.set_yticks(range(len(feat_order)))
        if c == 0:
            ax.set_yticklabels([ABBR[f] for f in feat_order], fontsize=6.6)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0, pad=2)
        ax.set_xlabel(f'Instances  ($n$ = {len(m):,})', fontsize=6.6, labelpad=2)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4, integer=True))
        ax.tick_params(axis='x', labelsize=6.0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.55)

        axf.plot(np.arange(len(m)), f_x, color='#222222', lw=0.45)
        axf.set_xlim(-0.5, len(m) - 0.5)
        lo, hi = float(f_x.min()), float(f_x.max())
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        axf.set_ylim(lo - pad, hi + pad)
        axf.set_yticks([])
        axf.tick_params(axis='x', bottom=False, labelbottom=False, length=0)
        axf.text(-0.012, 0.5, r'$f(x)$', transform=axf.transAxes, ha='right', va='center',
                 fontsize=6.4)
        for sp in ('top', 'right', 'left'):
            axf.spines[sp].set_visible(False)
        axf.spines['bottom'].set_linewidth(0.55)

        ax.text(0.5, -0.30, f'({chr(97 + k)}) {DISPLAY[SIX[ci]]}', transform=ax.transAxes,
                ha='center', va='top', fontsize=7.6)

    # shared colourbar, right side, High top / Low bottom
    cax = fig.add_axes([0.950, 0.235, 0.0115, 0.655])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    ticks = [-a, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, a]
    ticks = sorted({round(t, 2) for t in ticks})
    cb.set_ticks(ticks)
    cb.ax.set_yticklabels([f'{t:+.1f}' if t != 0 else '0' for t in ticks], fontsize=6.4)
    cb.ax.tick_params(length=2, width=0.5, pad=2)
    cb.outline.set_linewidth(0.6)
    cb.ax.set_title('High', fontsize=7.2, pad=4)
    cb.ax.text(0.5, -0.028, 'Low', transform=cb.ax.transAxes, ha='center', va='top',
               fontsize=7.2)
    cb.set_label('Signed TreeSHAP for the panel\'s class (raw-score units)\n'
                 f'colour scaled by $\\mathrm{{sign}}(x)\\,|x|^{{{gamma:g}}}$',
                 fontsize=6.9, labelpad=7)

    # abbreviation box, two-block layout like the reference
    left_keys = ['CT', 'PT', 'ST', 'AP', 'PR', 'PS', 'MOP']
    right_keys = ['PA', 'D', 'WT', 'SMYS', 'II', 'MA']
    box = fig.add_axes([0.075, 0.012, 0.94 - 0.075, 0.175])
    box.set_xticks([]); box.set_yticks([])
    box.set_facecolor('white')
    for sp in box.spines.values():
        sp.set_linewidth(0.6)
    for i, k in enumerate(left_keys):
        box.text(0.012, 0.90 - i * 0.125, f'$\\bf{{{k}}}$ = {ABBR_LONG[k]}',
                 fontsize=6.1, ha='left', va='top', transform=box.transAxes)
    for i, k in enumerate(right_keys):
        box.text(0.255, 0.90 - i * 0.125, f'$\\bf{{{k}}}$ = {ABBR_LONG[k]}',
                 fontsize=6.1, ha='left', va='top', transform=box.transAxes)
    box.text(0.50, 0.90,
             'Rows: the 13 model inputs in descending global importance.',
             fontsize=6.1, ha='left', va='top', transform=box.transAxes)
    box.text(0.50, 0.775,
             'Columns: every dev-confirm record of that class (by recorded cause,',
             fontsize=6.1, ha='left', va='top', transform=box.transAxes)
    box.text(0.50, 0.650,
             'not by prediction), archived row order. Colour = signed contribution;',
             fontsize=6.1, ha='left', va='top', transform=box.transAxes)
    box.text(0.50, 0.525,
             r'$f(x)$ = base value + $\sum_j \phi_j$ = that class raw margin,',
             fontsize=6.1, ha='left', va='top', transform=box.transAxes)
    box.text(0.50, 0.400,
             'mean over CatBoost seeds 42/123/456; additivity checked per record.',
             fontsize=6.1, ha='left', va='top', transform=box.transAxes)

    fig.text(0.075, 0.960, 'Instance-level TreeSHAP of the CatBoost diagnostic backbone '
                           'on 6,417 development-confirmation records',
             fontsize=8.2, ha='left', va='top')

    for ext in ('pdf', 'png'):
        fig.savefig(f'{outbase}.{ext}', dpi=dpi if ext == 'png' else None,
                    facecolor='white')
    plt.close(fig)
    return a


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    REPO_ROOT = os.path.dirname(os.path.dirname(PKG))
    ap.add_argument('--handoff', default=os.path.join(PKG, 'inputs'))
    ap.add_argument('--repro', default=REPO_ROOT)
    ap.add_argument('--outdir', default=os.path.join(PKG, 'figures'))
    ap.add_argument('--shap-cache',
                    default=os.path.join(PKG, 'data', 'per_sample_shap_6417.npz'))
    ap.add_argument('--colours', default='')
    ap.add_argument('--gamma', type=float, default=0.5,
                    help='exponent of the documented monotone colour scaling')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    fam = set_style()
    colfile = args.colours or os.path.join(
        args.handoff, 'style_reference', 'color_palette.json')
    with open(colfile) as f:
        cref = json.load(f)

    stopA = cref['reference_A_aggregated_heatmap']['color_stops_low_to_high']
    stopB = cref['reference_B_sample_panels']['color_stops_low_to_high']
    cmapA = build_cmap(stopA)
    cmapB = build_cmap(stopB)

    # ---- figure A source numbers -------------------------------------------
    pcf = pd.read_csv(os.path.join(
        args.handoff, '02_VERIFIED_LATEST_SHAP_SUMMARIES/shap_per_class_feature_importance.csv'))
    glo = pd.read_csv(os.path.join(
        args.handoff, '02_VERIFIED_LATEST_SHAP_SUMMARIES/shap_global_importance.csv'))
    assert pcf.shape[0] == 234 and set(pcf.n) == {6417} and set(pcf.seed) == set(SEEDS)

    # 3-seed arithmetic mean per (class, feature); the shipped per-class CSV holds
    # one row per seed, so this is mean over the 3 rows.
    A = (pcf.groupby(['cls', 'feature'])['mean_abs_shap'].mean()
            .unstack().loc[SIX, MAIN_13])
    feat_order = list(glo.sort_values('rank_cross_seed')['feature'])
    cls_order = list(SIX)
    A.to_csv(os.path.join(args.outdir, 'figureA_class_feature_matrix.csv'),
             float_format='%.9f')

    vmaxA = figure_a(
        A, cls_order, feat_order, cmapA,
        os.path.join(args.outdir, 'shap_class_feature_heatmap_latest'), args.gamma, args.dpi,
        explain_text=('Mean absolute TreeSHAP of the frozen CatBoost backbone '
                      '(seeds 42/123/456, $n$ = 6,417).\n'
                      'Values are absolute contributions in raw-score units -- not feature '
                      'values, and not an explanation of the Agent repair gate.'))

    # ---- figure B source numbers -------------------------------------------
    shap_mean, base_mean, y, prob, row_id, X = load_or_compute_shap(args)
    a_lim = figure_b(shap_mean, base_mean, y,
                     os.path.join(args.outdir, 'shap_multi_panel_latest'),
                     cmapB, args.gamma, args.dpi, feat_order, '')

    # ---- exported audit tables --------------------------------------------
    pd.DataFrame({'abbreviation': [ABBR[f] for f in MAIN_13],
                  'feature': MAIN_13,
                  'expansion': [ABBR_LONG[ABBR[f]] for f in MAIN_13],
                  'role': ['categorical' if f in CATEGORICAL else 'numeric' for f in MAIN_13],
                  'column_index_in_model': list(range(13))}
                 ).to_csv(os.path.join(args.outdir, 'ABBREVIATION_MAP.csv'), index=False)

    rows = []
    for ci, c in enumerate(SIX):
        m = np.where(y == ci)[0]
        f_x = base_mean[ci] + shap_mean[m][:, :, ci].sum(axis=1)
        rows.append(dict(cls=c, display=DISPLAY[c], n_instances=len(m),
                         base_value=base_mean[ci],
                         fx_min=float(f_x.min()), fx_median=float(np.median(f_x)),
                         fx_max=float(f_x.max()),
                         mean_abs_shap_own_class=float(np.abs(shap_mean[m][:, :, ci]).mean()),
                         max_abs_shap_own_class=float(np.abs(shap_mean[m][:, :, ci]).max())))
    pd.DataFrame(rows).to_csv(os.path.join(args.outdir, 'figureB_panel_stats.csv'),
                              index=False, float_format='%.9f')

    meta = dict(
        figure_a=dict(source='handoff shap_per_class_feature_importance.csv (mean of 3 seeds)',
                      vmax=vmaxA, gamma=args.gamma, classes=SIX, features=feat_order),
        figure_b=dict(source='frozen cb_seed42/123/456.cbm + dev_confirm manifest',
                      symmetric_limit=a_lim, gamma=args.gamma,
                      shap_shape=list(shap_mean.shape),
                      base_values=dict(zip(SIX, base_mean.tolist())),
                      instance_counts=dict(zip(SIX, [(y == i).sum().item() for i in range(6)]))),
        font=fam, dpi=args.dpi,
    )
    with open(os.path.join(args.outdir, 'figures_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
