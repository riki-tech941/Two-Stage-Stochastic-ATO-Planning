import numpy as np
import copy
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

## Parametri
c = [95, 310, 35, 160, 580, 65]          # costi acquisto componenti
v = [round(x * 0.33) for x in c]         # valori recupero 33%

DISCOUNT = {
    'standard':      [1.00, 1.00, 1.00],
    'back_to_school':[0.90, 1.00, 1.00],
    'black_friday':  [0.85, 0.85, 0.85],
}

a = [2+0.5, 2+1.0, 2+1.5]     # tempi assemblaggio [ore/unità]
C_cap = 800
COSTO_ASSEMBLAGGIO_ORA = 15
r_base = [
    1.25*(a[0]*COSTO_ASSEMBLAGGIO_ORA + c[0]+c[2]+c[5]),
    1.35*(a[1]*COSTO_ASSEMBLAGGIO_ORA + c[1]+2*c[2]+c[3]+c[5]),
    1.45*(a[2]*COSTO_ASSEMBLAGGIO_ORA + c[1]+2*c[2]+c[4]+c[5]),
]
COSTO_FISSO = 2500 
LOST_SALE_PENALTY = [round(x * 0.13) for x in r_base]

# Styling grafici
STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.facecolor": "#FAFAFA",
    "figure.facecolor": "white",
}

C_BLUE = "#1565C0"
C_ORANGE = "#E65100"
C_GREEN = "#2E7D32"
C_PINK = "#AD1457"
C_RED = "#C62828"
C_AMBER = "#F57F17"
C_GREY = "#546E7A"
ALPHA_BAND = 0.18

## Macro-scenari e campionamento

MACRO_SCENARIOS = {
    'standard': {
        'weight': 0.55,
        'means':  [100, 50, 30],
        'stds':   [15,  10,  8],
        'corr': np.array([
            [1.00, 0.40, 0.25],
            [0.40, 1.00, 0.30],
            [0.25, 0.30, 1.00],
        ]),
        'discounts': DISCOUNT['standard'],
    },
    'back_to_school': {
        'weight': 0.25,
        'means':  [155, 95, 42],
        'stds':   [30,  28, 18],
        'corr': np.array([
            [1.00, 0.75, 0.15],
            [0.75, 1.00, 0.20],
            [0.15, 0.20, 1.00],
        ]),
        'discounts': DISCOUNT['back_to_school'],
    },
    'black_friday': {
        'weight': 0.20,
        'means':  [165, 125, 105],
        'stds':   [35,  35,  40],
        'corr': np.array([
            [1.00, 0.30, 0.20],
            [0.30, 1.00, 0.70],
            [0.20, 0.70, 1.00],
        ]),
        'discounts': DISCOUNT['black_friday'],
    },
}

# Genera n campioni casuali interi non negativi da una normale multivariata Cholesky

def _cholesky_sample(rng, means, stds, corr, n):
    std_vec = np.array(stds)
    cov = corr * np.outer(std_vec, std_vec)
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(3) * 1e-6
        L = np.linalg.cholesky(cov)
    Z = rng.standard_normal((3, n))
    X = np.array(means).reshape(3, 1) + L @ Z
    return np.maximum(0, np.round(X)).astype(int)

# Genera scenari di domanda campionando il macro-scenario poi applicando Cholesky

def sample_demand(n_scenarios, seed=42):
    rng = np.random.default_rng(seed)
    weights = [MACRO_SCENARIOS[k]['weight'] for k in MACRO_SCENARIOS]
    macro_keys = list(MACRO_SCENARIOS.keys())
    chosen = rng.choice(len(macro_keys), size=n_scenarios, p=weights)
    labels = np.array([macro_keys[c] for c in chosen])
    D = np.zeros((3, n_scenarios), dtype=int)
    for key in macro_keys:
        mask = (labels == key)
        n_k = mask.sum()
        if n_k == 0:
            continue
        params = MACRO_SCENARIOS[key]
        D[:, mask] = _cholesky_sample(rng, params['means'], params['stds'], params['corr'], n_k)
    return D, labels

# Domanda attesa

def expected_demand():
    E = np.zeros(3)
    for key, params in MACRO_SCENARIOS.items():
        E += params['weight'] * np.array(params['means'])
    return E

# Ricavo atteso

def expected_revenues():
    """
    Ricavi medi ponderati sugli sconti di ogni macro-scenario.
    r_ev[j] = sum_k p_k * discount_k[j] * r_base[j]
    Usati nel modello EV per coerenza con la distribuzione stocastica.
    """
    r_ev = np.zeros(3)
    for key, params in MACRO_SCENARIOS.items():
        w = params['weight']
        for j in range(3):
            r_ev[j] += w * params['discounts'][j] * r_base[j]
    return r_ev

# Ricavi effettivi

def effective_revenues(scenario_label):
    discounts = MACRO_SCENARIOS[scenario_label]['discounts']
    return [r_base[j] * discounts[j] for j in range(3)]


## Modello ATO a due stadi
# D domanda: array (3,S), labels: etichette del macroscenario (per gli sconti)

def solve_ato(D, labels=None, r_fixed=None, use_capacity=True, verbose=False):
    n_scenarios = D.shape[1]
    S = range(n_scenarios)

    m = gp.Model("ATO_Stocastico")
    m.setParam("OutputFlag", 1 if verbose else 0)

    x = m.addVars(6, lb=0.0, name="x")
    y = m.addVars(3, n_scenarios, lb=0.0, name="y")
    w = m.addVars(6, n_scenarios, lb=0.0, name="w")
    u = m.addVars(3, n_scenarios, lb=0.0, name="u")
    h = m.addVars(n_scenarios,    lb=0.0, name="h")

    first_stage_cost = gp.quicksum(c[i] * x[i] for i in range(6)) + COSTO_FISSO

    recourse_terms = []
    for s in S:
        if r_fixed is not None:
            r_s = list(r_fixed)
        elif labels is not None:
            r_s = effective_revenues(labels[s])
        else:
            r_s = r_base

        recourse_terms.append(
            - r_s[0] * y[0, s]
            - r_s[1] * y[1, s]
            - r_s[2] * y[2, s]
            - gp.quicksum(v[i] * w[i, s] for i in range(6))
            + COSTO_ASSEMBLAGGIO_ORA * h[s]
            + LOST_SALE_PENALTY[0] * u[0, s]
            + LOST_SALE_PENALTY[1] * u[1, s]
            + LOST_SALE_PENALTY[2] * u[2, s]
        )

    m.setObjective(
        first_stage_cost + (1.0 / n_scenarios) * gp.quicksum(recourse_terms),
        GRB.MINIMIZE
    )

    for s in S:
        m.addConstr(y[0, s] + u[0, s] == D[0, s], f"dom_office_{s}")
        m.addConstr(y[1, s] + u[1, s] == D[1, s], f"dom_creator_{s}")
        m.addConstr(y[2, s] + u[2, s] == D[2, s], f"dom_gaming_{s}")

        m.addConstr(y[0, s] + w[0, s] == x[0])
        m.addConstr(y[1, s] + y[2, s] + w[1, s] == x[1])
        m.addConstr(y[0, s] + 2*y[1, s] + 2*y[2, s] + w[2, s] == x[2])
        m.addConstr(y[1, s] + w[3, s] == x[3])
        m.addConstr(y[2, s] + w[4, s] == x[4])
        m.addConstr(y[0, s] + y[1, s] + y[2, s] + w[5, s] == x[5])

        m.addConstr(h[s] == a[0]*y[0, s] + a[1]*y[1, s] + a[2]*y[2, s])
        if use_capacity:
            m.addConstr(h[s] <= C_cap)

    m.optimize()

    if m.Status == GRB.OPTIMAL:
        x_opt = np.array([x[i].X for i in range(6)])
        obj = m.ObjVal
        status = "Optimal"
    else:
        x_opt = np.zeros(6)
        obj = np.nan
        status = f"GRB_status_{m.Status}"

    return x_opt, obj, status

# ANALISI

def evaluate_out_of_sample(x_fixed, n_eval=2000, seed=9999, use_capacity=True):
    D_eval, labels_eval = sample_demand(n_eval, seed=seed)
    first_stage = np.dot(c, x_fixed) + COSTO_FISSO
    revenues = [
        _second_stage_value(x_fixed, D_eval[:, s], effective_revenues(labels_eval[s]), use_capacity)
        for s in range(n_eval)
    ]
    return first_stage - np.mean(revenues)


def _second_stage_value(x_fixed, d_s, r_s, use_capacity=True):
    m = gp.Model("second_stage")
    m.setParam("OutputFlag", 0)

    y = m.addVars(3, lb=0.0, name="y")
    w = m.addVars(6, lb=0.0, name="w")
    u = m.addVars(3, lb=0.0, name="u")

    m.setObjective(
        r_s[0]*y[0] + r_s[1]*y[1] + r_s[2]*y[2]
        + gp.quicksum(v[i]*w[i] for i in range(6))
        - COSTO_ASSEMBLAGGIO_ORA * (a[0]*y[0] + a[1]*y[1] + a[2]*y[2])
        - LOST_SALE_PENALTY[0] * u[0]
        - LOST_SALE_PENALTY[1] * u[1]
        - LOST_SALE_PENALTY[2] * u[2],
        GRB.MAXIMIZE
    )

    m.addConstr(y[0] + u[0] == d_s[0])
    m.addConstr(y[1] + u[1] == d_s[1])
    m.addConstr(y[2] + u[2] == d_s[2])

    m.addConstr(y[0] + w[0] == x_fixed[0])
    m.addConstr(y[1] + y[2] + w[1] == x_fixed[1])
    m.addConstr(y[0] + 2*y[1] + 2*y[2] + w[2] == x_fixed[2])
    m.addConstr(y[1] + w[3] == x_fixed[3])
    m.addConstr(y[2] + w[4] == x_fixed[4])
    m.addConstr(y[0] + y[1] + y[2] + w[5] == x_fixed[5])

    if use_capacity:
        m.addConstr(a[0]*y[0] + a[1]*y[1] + a[2]*y[2] <= C_cap)

    m.optimize()
    return m.ObjVal if m.Status == GRB.OPTIMAL else 0.0


# VSS

def compute_vss(n_eval=2000, seed_eval=9999):

    D_rp, labels_rp = sample_demand(500, seed=42)
    x_rp, _, _ = solve_ato(D_rp, labels_rp)
    rp = evaluate_out_of_sample(x_rp, n_eval=n_eval, seed=seed_eval)
    print(f"  RP  (stocastica, out-of-sample): {rp:,.2f} €")

    E_D = expected_demand()
    r_ev = expected_revenues()
    D_ev = E_D.reshape(3, 1).astype(int)
    x_ev, _, _ = solve_ato(D_ev, labels=None, r_fixed=r_ev)
    eev = evaluate_out_of_sample(x_ev, n_eval=n_eval, seed=seed_eval)
    print(f"  EEV (deterministica, out-of-sample): {eev:,.2f} €")

    vss = eev - rp
    print(f"  VSS = EEV - RP = {vss:,.2f} €")
    print(f"  La soluzione stocastica vale {vss:,.2f} € in più rispetto")
    print(f"  alla soluzione deterministica sui valori attesi.")
    return rp, eev, vss, x_rp, x_ev



# EVPI

def compute_evpi(n_eval=200, seed_eval=9999):
    D_rp, labels_rp = sample_demand(500, seed=42)
    x_rp, _, _ = solve_ato(D_rp, labels_rp)

    D_ws, labels_ws = sample_demand(n_eval, seed=seed_eval)

    first_stage_rp = np.dot(c, x_rp) + COSTO_FISSO
    revenues_rp = [
        _second_stage_value(x_rp, D_ws[:, s], effective_revenues(labels_ws[s]))
        for s in range(n_eval)
    ]
    rp = first_stage_rp - np.mean(revenues_rp)

    ws_values = []
    for s in range(n_eval):
        d_s = D_ws[:, s].reshape(3, 1)
        labels_s = np.array([labels_ws[s]])
        _, obj_ws_s, _ = solve_ato(d_s, labels=labels_s)
        ws_values.append(obj_ws_s)

    ws = np.mean(ws_values)
    evpi = max(rp - ws, 0.0)

    print(f"  WS (wait-and-see): {ws:,.2f} €")
    print(f"  RP (stocastica, stesso campione): {rp:,.2f} €")
    print(f"  EVPI = RP - WS = {evpi:,.2f} €")
    print(f"  -> L'informazione perfetta vale al massimo {evpi:,.2f} €.")
    return ws, rp, evpi



# stabilità in-sample/out-sample

def stability_analysis(
    scenario_sizes=[10, 25, 50, 100, 250, 500],
    n_replicas=10,
    n_eval=2000,
    seed_eval=9999
):
    results = {
        'N': scenario_sizes,
        'in_mean': [], 'in_std': [],
        'out_mean': [], 'out_std': [],
    }

    for N in scenario_sizes:
        in_vals, out_vals = [], []
        for rep in range(n_replicas):
            # Il seed viene modificato per ogni replica
            seed = rep * 100 + N
            D, labels = sample_demand(N, seed=seed)
            x_opt, obj_in, status = solve_ato(D, labels)
            obj_out = evaluate_out_of_sample(x_opt, n_eval=n_eval, seed=seed_eval)
            in_vals.append(obj_in)
            out_vals.append(obj_out)

        results['in_mean'].append(np.mean(in_vals))
        results['in_std'].append(np.std(in_vals))
        results['out_mean'].append(np.mean(out_vals))
        results['out_std'].append(np.std(out_vals))

        print(f"  N={N:4d} | "
              f"in-sample:  {np.mean(in_vals):10,.1f} ± {np.std(in_vals):6,.1f} | "
              f"out-of-sample: {np.mean(out_vals):10,.1f} ± {np.std(out_vals):6,.1f}")

    return results



# Analisi di Robustezza

def robustness_analysis(n_eval=2000, seed_eval=9999):
    D_ref, labels_ref = sample_demand(500, seed=42)
    x_ref, _, _ = solve_ato(D_ref, labels_ref)
    obj_ref = evaluate_out_of_sample(x_ref, n_eval=n_eval, seed=seed_eval)
    print(f"  Riferimento (distribuzione corretta): {obj_ref:,.2f} €")
    print(f"  x_ref = {np.round(x_ref).astype(int)}")

    scenarios_rob = {'Riferimento': obj_ref}

    def solve_perturbed(macro_perturbed, label, verbose_x=True):
        rng = np.random.default_rng(42)
        n_s = 500
        macro_keys = list(macro_perturbed.keys())
        weights = [macro_perturbed[k]['weight'] for k in macro_keys]
        chosen = rng.choice(len(macro_keys), size=n_s, p=weights)
        lbl = np.array([macro_keys[cc] for cc in chosen])

        D_pert = np.zeros((3, n_s), dtype=int)
        for key in macro_keys:
            mask = (lbl == key)
            n_k = mask.sum()
            if n_k == 0:
                continue
            p = macro_perturbed[key]
            D_pert[:, mask] = _cholesky_sample(rng, p['means'], p['stds'], p['corr'], n_k)

        x_pert, _, _ = solve_ato(D_pert, lbl)
        obj_pert = evaluate_out_of_sample(x_pert, n_eval=n_eval, seed=seed_eval)
        deg = obj_pert - obj_ref
        print(f"  {label}: {obj_pert:,.2f} € (degradazione: {deg:+,.2f} €)")
        if verbose_x:
            print(f"    x_pert = {np.round(x_pert).astype(int)}")
        scenarios_rob[label] = obj_pert
        return obj_pert

    # Errori sulle medie

    for factor, label in [(1.20, "Medie +20%"), (0.80, "Medie -20%"),
                          (1.30, "Medie +30%"), (0.70, "Medie -30%")]:
        m = copy.deepcopy(MACRO_SCENARIOS)
        for k in m:
            m[k]['means'] = [mu * factor for mu in m[k]['means']]
        solve_perturbed(m, label)

    # Errori sulla varianza

    for factor, label in [(0.5, "Std ×0.5 (sottostima varianza)"),
                           (2.0, "Std ×2.0 (sovrastima varianza)")]:
        m = copy.deepcopy(MACRO_SCENARIOS)
        for k in m:
            m[k]['stds'] = [s * factor for s in m[k]['stds']]
        solve_perturbed(m, label)

    # Errori sui pesi dei macro-scenari

    weight_configs = [
        ([0.6875, 0.3125, 0.00], "Black Friday ignorato (0.69, 0.31, 0.00)"),
        ([0.50, 0.30, 0.20],     "Black Friday sovrastimato (0.50, 0.30, 0.20)"),
        ([1/3,  1/3,  1/3],      "Pesi uniformi (1/3, 1/3, 1/3)"),
    ]
    for new_weights, label in weight_configs:
        m = copy.deepcopy(MACRO_SCENARIOS)
        for i, k in enumerate(m.keys()):
            m[k]['weight'] = new_weights[i]
        solve_perturbed(m, label)

    # Correlazione ignorata (indipendenza)

    m_ind = copy.deepcopy(MACRO_SCENARIOS)
    for k in m_ind:
        m_ind[k]['corr'] = np.eye(3)
    solve_perturbed(m_ind, "Correlazione ignorata")

    # Correlazione negativa (anti-correlazione)

    neg_corr = np.array([
        [1.00, -0.40, -0.30],
        [-0.40,  1.00, -0.35],
        [-0.30, -0.35,  1.00],
    ])
    m_neg = copy.deepcopy(MACRO_SCENARIOS)
    for k in m_neg:
        m_neg[k]['corr'] = neg_corr.copy()
    solve_perturbed(m_neg, "Correlazione negativa")

    return scenarios_rob

# Analisi estesa (capacità)

def capacity_analysis(
        capacities=None,
        n_scenarios=500,
        n_eval=2000,
        seed=42,
        seed_eval=9999,
):

    global C_cap

    if capacities is None:
        capacities = [160, 240, 320, 400, 480, 560, 640, 720, 800]

    results = {
        'caps': capacities,
        'rp_cost': [],
        'eev_cost': [],
        'vss': [],
    }

    D_train, labels_train = sample_demand(n_scenarios, seed=seed)
    r_ev = expected_revenues()
    E_D = expected_demand()
    D_ev = E_D.reshape(3, 1).astype(int)

    for cap in capacities:
        C_cap = cap

        x_rp_cap, _, _ = solve_ato(D_train, labels_train, use_capacity=True)
        rp_cost = evaluate_out_of_sample(x_rp_cap, n_eval=n_eval, seed=seed_eval, use_capacity=True)

        x_ev_cap, _, _ = solve_ato(D_ev, labels=None, r_fixed=r_ev, use_capacity=True)
        eev_cost = evaluate_out_of_sample(x_ev_cap, n_eval=n_eval, seed=seed_eval, use_capacity=True)

        vss = eev_cost - rp_cost

        results['rp_cost'].append(rp_cost)
        results['eev_cost'].append(eev_cost)
        results['vss'].append(vss)

        print(f"  C_cap={cap:4d}h | RP={rp_cost:>10,.0f}€ | EEV={eev_cost:>10,.0f}€ | VSS={vss:>8,.0f}€")

    C_cap = 800  # ripristina valore originale
    return results



def plot_stability_ac(results):
    
    with plt.rc_context(STYLE):
        N = np.array(results['N'])
        in_mean = np.array(results['in_mean'])
        in_std = np.array(results['in_std'])
        out_mean = np.array(results['out_mean'])
        out_std = np.array(results['out_std'])

        fig, ax = plt.subplots(figsize=(8.5, 5))

        ax.fill_between(N, in_mean - in_std, in_mean + in_std,
                        alpha=ALPHA_BAND, color=C_BLUE,
                        label='In-sample  ±1 std')
        ax.fill_between(N, out_mean - out_std, out_mean + out_std,
                        alpha=ALPHA_BAND, color=C_PINK,
                        label='Out-of-sample  ±1 std')

        ax.plot(N, in_mean,  'o-', color=C_BLUE, linewidth=2.0,
                markersize=6, label='In-sample  (media)')
        ax.plot(N, out_mean, 's-', color=C_PINK, linewidth=2.0,
                markersize=6, label='Out-of-sample  (media)')

        conv_lo, conv_hi = 250, 500
        ax.axvspan(conv_lo, conv_hi, alpha=0.06, color=C_GREEN, zorder=0)
    

        for n_val, im, is_, om, os_ in zip(N, in_mean, in_std, out_mean, out_std):
            ax.vlines(n_val, im - is_, im + is_,
                      color=C_BLUE, alpha=0.35, linewidth=1.8)
            ax.vlines(n_val, om - os_, om + os_,
                      color=C_PINK, alpha=0.35, linewidth=1.8)

        ax.set_xscale('log')
        ax.set_xticks(N)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.set_xlabel("Numero di scenari $N$ (scala logaritmica)", labelpad=6)
        ax.set_ylabel("Valore obiettivo [€]", labelpad=6)
        ax.set_title(
            "Stabilità della soluzione stocastica: In-Sample vs Out-of-Sample",
            fontsize=11, fontweight='bold', pad=10,
        )

        leg = ax.legend(loc='upper right', framealpha=0.92,
                        edgecolor='#CCCCCC', ncol=2)
        leg.get_frame().set_linewidth(0.8)

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{x:,.0f} €'))

        plt.tight_layout()
        plt.savefig("stability.png")
        print("\n  Grafico salvato: stability.png")
        plt.show()


def plot_vss_evpi_ac(rp, eev, ws):

    with plt.rc_context(STYLE):
        labels_disp = ["WS\n(Info Perfetta)", "RP\n(Stocastica)", "EEV\n(Deterministica)"]
        vals = [ws, rp, eev]
        disp = [-ws, -rp, -eev]
        colors = [C_GREEN, C_BLUE, C_ORANGE]

        fig, ax = plt.subplots(figsize=(7.5, 5.8))
        x_pos = np.arange(3)
        bars = ax.bar(x_pos, disp, width=0.50,
                      color=colors, alpha=0.88,
                      edgecolor='white', linewidth=1.5, zorder=3)

        # Etichette dentro le barre
        y_min_disp = min(disp)
        for bar, val_cost, val_profit in zip(bars, vals, disp):
            offset = abs(y_min_disp) * 0.015
            y_txt = val_profit - offset
            ax.text(
                bar.get_x() + bar.get_width() / 2, y_txt,
                f'{-val_cost:+,.0f} €\n',
                ha='center', va='top',
                fontsize=8.5, color='white', fontweight='bold', zorder=5,
            )

        evpi_abs = abs(rp - ws)
        evpi_pct = evpi_abs / abs(ws) * 100 if ws != 0 else 0

        x_evpi = x_pos[0] + 0.30
        y_top = disp[0]                   #
        y_bot = disp[1]

        ax.annotate(
            '', xy=(x_evpi, y_bot), xytext=(x_evpi, y_top),
            arrowprops=dict(arrowstyle='<->', color=C_GREEN, lw=2.0), zorder=6,
        )
        ax.text(
            x_evpi + 0.07, (y_top + y_bot) / 2,
            f'EVPI = {evpi_abs:,.0f} €\n({evpi_pct:.1f}% profitto perso\nvs info perfetta)',
            ha='left', va='center', fontsize=8.0, color=C_GREEN, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=C_GREEN, alpha=0.90, lw=0.9),
            zorder=7,
        )

        vss_abs = abs(eev - rp)
        vss_pct = vss_abs / abs(rp) * 100 if rp != 0 else 0

        x_vss = x_pos[1] + 0.30             
        y_top2 = disp[1]                 
        y_bot2 = disp[2]                     

        ax.annotate(
            '', xy=(x_vss, y_bot2), xytext=(x_vss, y_top2),
            arrowprops=dict(arrowstyle='<->', color=C_ORANGE, lw=2.0), zorder=6,
        )
        ax.text(
            x_vss + 0.07, (y_top2 + y_bot2) / 2,
            f'VSS = {vss_abs:,.0f} €\n({vss_pct:.1f}% recuperato\ncon approccio stocastico)',
            ha='left', va='center', fontsize=8.0, color=C_ORANGE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec=C_ORANGE, alpha=0.90, lw=0.9),
            zorder=7,
        )

        # Linea tratteggiata riferimento RP
        ax.axhline(-rp, color=C_BLUE, linestyle=':', linewidth=1.2, alpha=0.6, zorder=2)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels_disp, fontsize=10)
        ax.set_ylabel(
            "Profitto netto atteso out-of-sample [€]\n"
            "(barra più bassa = profitto maggiore)", labelpad=6,
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f} €'))
        ax.set_title("Valore dell'Informazione: VSS e EVPI", fontsize=11, fontweight='bold', pad=10)

        ax.set_xlim(-0.5, x_pos[-1] + 1.1)

        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='x', bottom=False)

        plt.tight_layout()
        plt.savefig("vss_evpi.png", dpi=150, bbox_inches='tight')
        print("  Grafico salvato: vss_evpi.png")
        plt.show()


def plot_robustness_ac(scenarios_rob):

    with plt.rc_context(STYLE):
        ref = scenarios_rob['Riferimento']
        items = {k: val for k, val in scenarios_rob.items() if k != 'Riferimento'}

        labels_raw = list(items.keys())
        pct_deg = [(val - ref) / abs(ref) * 100 for val in items.values()]

        order = np.argsort(pct_deg)[::-1]
        labels_ord = [labels_raw[i] for i in order]
        pct_ord = [pct_deg[i] for i in order]

        def _color(label, pct):
            lbl = label.lower()
            if ('mean' in lbl or 'med' in lbl or '+' in lbl or
                    '-20' in lbl or '-30' in lbl or '20%' in lbl or '30%' in lbl):
                return C_RED if abs(pct) > 5 else C_AMBER
            return C_GREY

        colors_bar = [_color(l, p) for l, p in zip(labels_ord, pct_ord)]

        fig, ax = plt.subplots(figsize=(10, max(5.5, len(labels_ord) * 0.62 + 1.5)))
        y_pos = np.arange(len(labels_ord))
        bars = ax.barh(y_pos, pct_ord, color=colors_bar, alpha=0.87,
                       edgecolor='white', linewidth=1.0, height=0.62, zorder=3)

        ax.axvline(0, color=C_GREY, linewidth=1.8, linestyle='--',
                   zorder=4, label='Riferimento (0% degrado)')

        for bar, pct in zip(bars, pct_ord):
            sign = '+' if pct >= 0 else ''
            offset = 0.15
            x_txt = pct + offset if pct >= 0 else pct - offset
            ha = 'left' if pct >= 0 else 'right'
            weight = 'bold' if abs(pct) > 5 else 'normal'
            ax.text(x_txt, bar.get_y() + bar.get_height() / 2,
                    f'{sign}{pct:.1f}%',
                    va='center', ha=ha, fontsize=8.5, fontweight=weight, zorder=5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels_ord, fontsize=9)

        x_max = max(abs(p) for p in pct_ord) * 1.35
        ax.set_xlim(-0.2*x_max, x_max)
        ax.set_xlabel("Variazione del costo netto atteso rispetto al Riferimento [%]", labelpad=6)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:+.0f}%'))
        ax.invert_yaxis()

        ax.set_title(
            "Analisi di Robustezza Distribuzionale\n"
            "Degrado del profitto al variare delle ipotesi sul modello di domanda",
            fontsize=11, fontweight='bold', pad=10,
        )
        
        plt.tight_layout()
        plt.savefig("robustness.png")
        print("  Grafico salvato: robustness.png")
        plt.show()


def plot_first_stage_enhanced_ac(x_rp, x_ev):

    with plt.rc_context(STYLE):
        components = ['CPU\nBase', 'CPU\nHigh-End', 'RAM\n8GB',
                      'GPU\nMedia', 'GPU\nTop', 'Chassis\n& PSU']
        x_pos = np.arange(len(components))
        width = 0.35

        pct_diff = np.where(
            x_ev > 0,
            100 * (x_rp - x_ev) / x_ev,
            np.where(x_rp > 0, 100.0, 0.0)
        )
        bar_colors_rp = ['#D32F2F' if abs(p) > 10 else C_BLUE for p in pct_diff]

        fig, ax = plt.subplots(figsize=(11, 5))
        bars_rp = ax.bar(x_pos - width / 2, x_rp, width,
                         color=bar_colors_rp, alpha=0.88,
                         label='Stocastica (RP)', edgecolor='white')
        bars_ev = ax.bar(x_pos + width / 2, x_ev, width,
                         color=C_ORANGE, alpha=0.75,
                         label='Deterministica (EV)', edgecolor='white')

        for i, (bar_rp, bar_ev, pct) in enumerate(zip(bars_rp, bars_ev, pct_diff)):
            y_top = max(bar_rp.get_height(), bar_ev.get_height()) + 2
            color_ann = '#D32F2F' if abs(pct) > 10 else '#555555'
            sign = '+' if pct >= 0 else ''
            ax.text(x_pos[i], y_top, f'{sign}{pct:.0f}%',
                    ha='center', va='bottom', fontsize=9,
                    color=color_ann, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(components, fontsize=10)
        ax.set_ylabel("Quantità acquistata [unità]", fontsize=11)
        ax.set_title(
            "Confronto primo stadio: Stocastica (RP) vs Deterministica (EV)",
            fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig("first_stage_enhanced.png")
        print(" Grafico salvato: first_stage_enhanced.png")
        plt.show()


def plot_capacity_analysis_ac(results):

    with plt.rc_context(STYLE):
        caps = np.array(results['caps'])
        rp_cost = np.array(results['rp_cost'])
        eev_cost = np.array(results['eev_cost'])
        vss = np.array(results['vss'])

        def _fte_label(cap):
            fte = cap / 160
            if fte == int(fte):
                return f"{cap}h\n({int(fte)} FTE)"
            return f"{cap}h\n({fte:.1f} FTE)"

        fte_labels = [_fte_label(c) for c in caps]

        rp_range = rp_cost.max() - rp_cost.min()
        plateau_thresh = rp_range * 0.01
        delta_rp = np.abs(np.diff(rp_cost))
        plateau_indices = np.where(delta_rp < plateau_thresh)[0]
        plateau_cap = caps[plateau_indices[0] + 1] if len(plateau_indices) > 0 else caps[-2]

        fig, ax1 = plt.subplots(figsize=(11, 5.5))

        ax1.axvspan(plateau_cap - 10, caps[-1] + 10,
                    alpha=0.07, color=C_AMBER, zorder=0,
                    label=f'Plateau (cap ≥ {plateau_cap}h)')
        ax1.axvline(plateau_cap, color=C_AMBER, linestyle='--',
                    linewidth=1.8, alpha=0.85, zorder=2)
        
        ax1.plot(caps, rp_cost, 'o-', color=C_BLUE, linewidth=2.2,
                 markersize=7, label='Costo RP (stocastico)', zorder=4)

        idx_plat = caps.tolist().index(plateau_cap)
        ax1.plot(caps[idx_plat:], rp_cost[idx_plat:],
                 'o-', color=C_BLUE, linewidth=4.5,
                 markersize=7, alpha=0.25, zorder=3)

        ax1.plot(caps, eev_cost, 's--', color=C_ORANGE, linewidth=2.0,
                 markersize=6, label='Costo EEV (deterministico)', zorder=4)

        ax1.fill_between(caps, rp_cost, eev_cost,
                         alpha=0.07, color=C_GREY, zorder=1, label='Gap RP–EEV')

        ax1.axvline(800, color=C_GREY, linestyle=':', linewidth=1.2,
                    alpha=0.7, label='Baseline (800h)')

        ax1.set_xlabel("Capacità mensile [ore / FTE equivalenti]", labelpad=6)
        ax1.set_ylabel("Costo netto atteso out-of-sample [€]",
                       color='black', labelpad=6)
        ax1.set_xticks(caps)
        ax1.set_xticklabels(fte_labels, fontsize=8.5)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{x:,.0f} €'))
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()
        ax2.plot(caps, vss, 'D:', color=C_PINK, linewidth=2.0,
                 markersize=7, label='VSS', zorder=4)
        ax2.axhline(0, color=C_PINK, linewidth=0.7, linestyle=':', alpha=0.6)
        ax2.set_ylabel("VSS [€]", color=C_PINK, labelpad=6)
        ax2.tick_params(axis='y', labelcolor=C_PINK)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{x:,.0f} €'))
        ax2.spines['right'].set_edgecolor(C_PINK)
        ax2.spines['right'].set_linewidth(1.2)

        lines1, lbels1 = ax1.get_legend_handles_labels()
        lines2, lbels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, lbels1 + lbels2,
                   loc='upper right', framealpha=0.92,
                   edgecolor='#CCCCCC', fontsize=8.5, ncol=2)

        ax1.set_title(
            "Analisi di Sensitività sulla Capacità Produttiva\n"
            "Costo ottimo e valore dell'informazione stocastica (VSS) al variare delle ore disponibili",
            fontsize=11, fontweight='bold', pad=10,
        )

        plt.tight_layout()
        plt.savefig("capacity_analysis.png")
        print("\n  Grafico salvato: capacity_analysis.png")
        plt.show()

def plot_marginal_capacity_value_ac(results):

    with plt.rc_context(STYLE):
        COSTO_LAVORO_ORA = 15
        SOGLIA_STOP = 10

        caps = np.array(results['caps'])
        rp_cost = np.array(results['rp_cost'])

        delta_cost = np.diff(rp_cost)
        delta_cap = np.diff(caps)
        marg_val = -delta_cost / delta_cap
        mid_caps = (caps[:-1] + caps[1:]) / 2

        bar_colors = [
            C_GREEN if val > COSTO_LAVORO_ORA else
            (C_AMBER if val > SOGLIA_STOP else C_GREY)
            for val in marg_val
        ]

        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        bars = ax.bar(mid_caps, marg_val, width=delta_cap[0] * 0.72,
                      color=bar_colors, alpha=0.88,
                      edgecolor='white', linewidth=1.2, zorder=3)

        y_max = max(marg_val) * 1.02 if marg_val.max() > 0 else 1
        for bar, val in zip(bars, marg_val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + y_max * 0.02,
                f'{val:.1f} €/h',
                ha='center', va='bottom', fontsize=8.5, zorder=5,
            )

        ax.axhline(COSTO_LAVORO_ORA, color=C_PINK, linestyle='--',
                   linewidth=2.0, zorder=4,
                   label=f'Costo manodopera  ({COSTO_LAVORO_ORA} €/h)')
        ax.text(
            mid_caps[-1] + delta_cap[0] * 0.4, COSTO_LAVORO_ORA,
            f'{COSTO_LAVORO_ORA} €/h',
            va='center', ha='left', fontsize=8.5,
            color=C_PINK, fontweight='bold',
        )

        ax.set_xticks(mid_caps)
        ax.set_xticklabels([f"{int(c)}h" for c in mid_caps], fontsize=9)
        ax.set_xlabel("Capacità mensile [ore]", labelpad=6)
        ax.set_ylabel("Riduzione costo marginale [€/h aggiuntiva]", labelpad=6)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f'{x:.0f} €/h'))
        ax.set_ylim(bottom=0, top=y_max * 1.35)

        ax.set_title(
            "Valore Marginale della Capacità Produttiva\n"
            "Convenienza economica nell'assumere personale aggiuntivo (step +80h)",
            fontsize=11, fontweight='bold', pad=10,
        )

        plt.tight_layout()
        plt.savefig("marginal_capacity.png")
        print(" Grafico salvato: marginal_capacity.png")
        plt.show()


def main():

    print(f"\n  Parametri economici:")
    print(f"  Costi fissi mensili:  {COSTO_FISSO:>8,.0f} €")
    print(f"  Costo assemblaggio/ora:  {COSTO_ASSEMBLAGGIO_ORA:>8,.0f} €/h")
    print(f"  Capacità produttiva:  {C_cap:>8,} ore/mese")
    print(f"  Ricavi base: Office={r_base[0]:.0f}€  Creator={r_base[1]:.0f}€  Gaming={r_base[2]:.0f}€")

    r_ev = expected_revenues()
    print(f"  Ricavi medi ponderati (EV): Office={r_ev[0]:.1f}€  "
          f"Creator={r_ev[1]:.1f}€  Gaming={r_ev[2]:.1f}€")

    E_D = expected_demand()
    print(f"  Domanda attesa: Office={E_D[0]:.0f}  Creator={E_D[1]:.0f}  Gaming={E_D[2]:.0f}")

    ore_attese = E_D[0]*a[0] + E_D[1]*a[1] + E_D[2]*a[2]
    print(f"  Ore assemblaggio attese: {ore_attese:.1f} / {C_cap} (capacità)")

    
    stability_results = stability_analysis(
        scenario_sizes=[25, 50, 100, 250, 500, 2000],
        n_replicas=15,
        n_eval=2000,
        seed_eval=9999,
    )

    #stabilità is/os
    plot_stability_ac(stability_results)

    #vss
    rp, eev, vss, x_rp, x_ev = compute_vss(n_eval=2000, seed_eval=9999)

    #evpi
    ws, rp_evpi, evpi = compute_evpi(n_eval=2000, seed_eval=9999)

    plot_vss_evpi_ac(rp_evpi, eev, ws)
    
    #robustezza
    scenarios_rob = robustness_analysis(n_eval=2000, seed_eval=9999)
    plot_robustness_ac(scenarios_rob)

    plot_first_stage_enhanced_ac(x_rp, x_ev)

    cap_results = capacity_analysis(
        capacities=[160, 240, 320, 400, 480, 560, 640, 720, 800],
        n_scenarios=500,
        n_eval=2000,
        seed=42,
        seed_eval=9999,
    )
    plot_capacity_analysis_ac(cap_results)

    plot_marginal_capacity_value_ac(cap_results)


if __name__ == "__main__":
    main()