from numpy.linalg import norm
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import tensorly as tl
from copy import deepcopy


def plot_subjects(factors):

    C1 = factors[2].copy()  # metabolites x R1
    n_metabolites, R1 = C1.shape
    
    # normalize each column of C1
    for i in range(R1):
        C1[:, i] /= (norm(C1[:, i]))
    
    fig, axes = plt.subplots(R1, 1, figsize=(8, 1.5*R1), sharex=False)
    if R1 == 1:
        axes = [axes]
    
    for r in range(R1):
        axes[r].bar(range(n_metabolites), C1[:, r], label=f"Comp {r+1}")
        
        axes[r].set_ylabel(f"Comp {r+1}")
        # axes[r].set_xticks(range(n_metabolites))
        # axes[r].set_xticklabels(metabolites, rotation=45, ha="right", fontsize=7)
    
    axes[-1].set_xlabel('Subjects')
    plt.tight_layout()
    plt.show()

def plot_allergens(factors,allergens):
    A1 = factors[0].copy()  # metabolites x R1
    n_allergens, R1 = A1.shape

    # normalize each column of C1
    for i in range(R1):
        A1[:, i] /= (norm(A1[:, i]))
    
    fig, axes = plt.subplots(R1, 1, figsize=(8, 1.5*R1), sharex=False)
    if R1 == 1:
        axes = [axes]
    
    for r in range(R1):
        x = np.arange(n_allergens)
        axes[r].bar(x, A1[:, r], tick_label=allergens)
        axes[r].set_ylabel(f"Comp {r+1}")
        axes[r].set_xticklabels(allergens, rotation=45, 
                                ha="right", fontsize=7)
    
    axes[-1].set_xlabel('Allergens')
    plt.tight_layout()
    plt.show()

def plot_mean_profile(factors):
    A, B, D = factors
    B2plot = deepcopy(B)

    R = D.shape[1]      # components
    K = D.shape[0]      # subjects
    J = B[0].shape[0]   # time points

    if np.allclose(D,np.ones_like(D)) == False: #parafac2

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    else: # cmf

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm
                D[k,r] *= normm

            normm2 = tl.norm(A[:,r])
            D[:,r] *= normm2

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    fig, axs = plt.subplots(R, 1, figsize=(8, 4.5*R))
    if R == 1:
        axs = [axs]

    x = [*range(J)]

    for r in range(R):
        ax = axs[r]
        stack = np.stack([B2plot[k][:, r] for k in range(K)], axis=0).astype(float)
        with np.errstate(invalid='ignore'):
            center = np.nanmean(stack, axis=0)  

            # SEM of the mean per time point
            counts = np.sum(np.isfinite(stack), axis=0)
            sd = np.nanstd(stack, axis=0, ddof=1)
            sem = np.where(counts > 1, sd / np.sqrt(counts), 0.0)

            # Shaded band and curve
            ax.fill_between(x, center - sem, center + sem, alpha=0.25, color='tab:gray', linewidth=0)
            ax.plot(x, center, alpha=0.9, color='tab:gray', linewidth=3)

        # ax.set_xticks(xticks)
        ax.grid(True)
        ax.set_title(f"r={r+1} (mean ± SEM)")
        # ax.legend()

    plt.tight_layout()

def plot_stratified_mean_profile(factors, meta):

    A, B, D = factors

    B2plot = deepcopy(B)

    R = D.shape[1]          # number of components
    K = D.shape[0]          # number of subjects/sessions
    J = B[0].shape[0]       # number of time points

    if np.allclose(D,np.ones_like(D)) == False:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    else:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm
                D[k,r] *= normm

            normm2 = tl.norm(A[:,r])
            D[:,r] *= normm2

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]
        

    # Precompute indices per BMI category (sorted for deterministic coloring)
    bmi_vals = np.unique(meta)
    group_idx = {bv: np.where(meta == bv)[0] for bv in bmi_vals}

    # Color/label mapping
    def bmi_info(bv):
        if bv == 1:
            return "Natural(1)", 'tab:red'
        elif bv == 2:
            return "C-section(2)", 'tab:blue'
        elif bv == 3:
            return "Vacuum(3)", 'tab:green'

    fig, axs = plt.subplots(R, 1, figsize=(1.5*R, 3*R))
    if R == 1:
        axs = [axs]

    for r in range(R):
        ax = axs[r]
        for bv in bmi_vals:
            # if bv == 3: continue
            idx = group_idx[bv]
            if idx.size == 0:
                continue

            center = np.empty(J, dtype=float)
            sem = np.zeros(J, dtype=float)

            for j in range(J):
                # Collect values across subjects in this BMI group for time j, component r
                vals = np.array([B2plot[k][j, r] for k in idx], dtype=float)
                # Drop non-finite just in case
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    center[j] = np.nan
                    sem[j] = np.nan
                    continue
                
                center[j] = np.mean(vals)

                # SEM around the mean (as requested), even if center is the median
                if vals.size > 1:
                    sem[j] = np.std(vals, ddof=1) / np.sqrt(vals.size)
                else:
                    sem[j] = 0.0

            bmi_label, bmi_color = bmi_info(bv)

            # Shaded SEM band (no label to avoid legend dupes)
            ax.fill_between(np.arange(J), center - sem, center + sem, alpha=0.25, color=bmi_color, linewidth=0)

            # Central curve
            ax.plot(np.arange(J), center, label=bmi_label, alpha=0.9, color=bmi_color, linewidth=3)

        ax.grid(True)
        ax.set_title(f"r={r+1} (mean ± SEM)")
        ax.legend()

    plt.tight_layout()

def get_profiles_factors(factors):
    A, B, D = factors
    B2plot = deepcopy(B)

    R = D.shape[1]      # components
    K = D.shape[0]      # subjects
    J = B[0].shape[0]   # time points

    if np.allclose(D,np.ones_like(D)) == False: #parafac2

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    else: # cmf
        
        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm
                D[k,r] *= normm

            normm2 = tl.norm(A[:,r])
            D[:,r] *= normm2

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    return A, B2plot, D

def plot_metabolite_component(factors,metabolites,comp,comp_i,axes,plot_legend=False,plot_title=False,plot_xticks=False):

    from matplotlib.lines import Line2D

    C1 = factors[0].copy()  # metabolites x R
    n_metabolites, R1 = C1.shape

    # normalize each column of C1
    for i in range(R1):
        nrm = tl.norm(C1[:, i])
        if nrm != 0:
            C1[:, i] /= nrm

    # --- NEW: explicit metabolite sets ---
    ketone_set = {'bOHbutyrate', 'Acetate', 'Acetoacetate', 'Acetone'}
    glycolysis_set = {'Glucose', 'Lactate', 'Pyruvate', 'Citrate'}

    # ---- style map: label -> (color, marker) ----
    styles = {
        # biomarkers
        "Insulin":   dict(color="tab:orange", marker="s"),
        "Cpeptid":   dict(color="tab:brown",  marker="D"),

        # --- NEW groups ---
        "Ketone bodies": dict(color="tab:olive", marker="P"),
        "Glycolysis":    dict(color="tab:pink",  marker="X"),

        # lipoproteins
        "HDL-":      dict(color="tab:red",    marker="o"),
        "XL-HDL":    dict(color="tab:red",    marker="^"),
        "L-HDL":     dict(color="tab:red",    marker="*"),
        "M-HDL":     dict(color="tab:red",    marker="+"),
        "S-HDL":     dict(color="tab:red",    marker="x"),

        "IDL-":      dict(color="tab:blue",   marker="o"),

        "LDL-":      dict(color="tab:green",  marker="o"),
        "L-LDL":     dict(color="tab:green",  marker="*"),
        "M-LDL":     dict(color="tab:green",  marker="+"),
        "S-LDL":     dict(color="tab:green",  marker="x"),

        "Rest":      dict(color="0.45",       marker="o"),
        "Total":     dict(color="tab:cyan",   marker="o"),

        "VLDL-":     dict(color="tab:purple", marker="o"),
        "XXL-VLDL":  dict(color="tab:purple", marker=">"),
        "XL-VLDL":   dict(color="tab:purple", marker="^"),
        "L-VLDL":    dict(color="tab:purple", marker="*"),
        "M-VLDL":    dict(color="tab:purple", marker="+"),
        "S-VLDL":    dict(color="tab:purple", marker="x"),
        "XS-VLDL":   dict(color="tab:purple", marker="v"),
    }

    # classify metabolite name -> legend key
    # IMPORTANT: match longer prefixes first so XL-VLDL doesn't get caught by VLDL-
    # Exclude non-prefix groups from prefix matching.
    non_prefix_groups = {"Rest", "Total", "Ketone bodies", "Glycolysis"}
    match_order = sorted(
        [k for k in styles.keys() if k not in non_prefix_groups],
        key=len, reverse=True
    )

    def metab_key(name: str) -> str:
        s = str(name).strip()

        # exact-name groups first
        if s == "Total":
            return "Total"
        if s in ketone_set:
            return "Ketone bodies"
        if s in glycolysis_set:
            return "Glycolysis"

        # then prefix-based groups
        for key in match_order:
            if s.startswith(key):
                return key

        return "Rest"

    # group indices by class
    groups = {k: [] for k in styles.keys()}
    for i, m in enumerate(metabolites):
        groups[metab_key(m)].append(i)

    x = np.arange(n_metabolites)

    lw = 0.7  # helps '+' and 'x' show up

    if plot_xticks is True:
        axes.set_xticks([0, 49, 99, 149], labels=["1", "50", "100", "150"])
        axes.set_xlabel("Metabolites",fontsize=5)

    axes.set_axisbelow(True)
    axes.grid(True, zorder=0)
    axes.set_ylabel(f"r={comp_i+1}")

    for key, idxs in groups.items():
        if not idxs:
            continue
        st = styles[key]
        idxs = np.asarray(idxs, dtype=int)

        axes.scatter(
            x[idxs], C1[idxs, comp],
            s=3,
            marker=st["marker"],
            c=st["color"],
            linewidths=lw
        )

    if plot_title is True:
        axes.set_title(r"Metabolites ($\mathbf{a}_r$)")

    axes.set_ylim([0, 0.8])

    # ----- one legend for the whole figure -----
    legend_order = [
        "HDL-", "XXL-VLDL", "XL-HDL", "L-HDL", "M-HDL", "S-HDL",
        "IDL-",
        "LDL-", "L-LDL", "M-LDL", "S-LDL",
        "Rest", "Total",
        "VLDL-", "XL-VLDL", "L-VLDL", "M-VLDL", "S-VLDL", "XS-VLDL",
        "Ketone bodies", "Glycolysis",
        "Insulin", "Cpeptid",
    ]

    handles = [
        Line2D([], [], linestyle="None",
               marker=styles[k]["marker"],
               color=styles[k]["color"],
               label=k,
               markersize=4)
        for k in legend_order
        if k in styles
    ]

    axes.tick_params(labelsize=5)

    if plot_legend is True:
        axes.legend(handles=handles, loc=[-0.1, -1.24], ncols=7, fontsize=5)


def plot_metabolomics_profile_component(factors,comp,comp_i,axes,plot_legend=True,plot_xticks=False,plot_title=False,time_points=None):

    A, B, D = factors

    B2plot = deepcopy(B)

    R = D.shape[1]          # number of components
    K = D.shape[0]          # number of subjects/sessions
    J = B[0].shape[0]       # number of time points

    if np.allclose(D,np.ones_like(D)) == False:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    else:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm
                D[k,r] *= normm

            normm2 = tl.norm(A[:,r])
            D[:,r] *= normm2

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    if J == 8:
        x = time_points
        xticks = [0,1,2,3,4]
    else:
        x = time_points
        xticks = np.arange(0,len(time_points))
    
    for subj in range(K):
        axes.grid(True)
        axes.plot(x,B2plot[subj][:, comp], alpha=0.3, color='gray', linewidth=0.7)
        axes.set_ylim([0,0.28])

    if plot_title is True:
        axes.set_title(
            r"Individual profiles $\left\{c_{k,r} \; [\boldsymbol{b}_k]_r\right\}_{k=1}^{K}$"
        )
    
    axes.tick_params(labelsize=5)

    if plot_xticks is True:
        if R == 6:
            axes.set_xlabel("Time (hours)",fontsize=5)
        else:
            axes.set_xlabel("Time (years)",fontsize=5)
        if time_points is not None:
            axes.set_xticks(xticks,xticks,fontsize=5)
    else:
        axes.set_xticks(xticks,[],fontsize=5)

def plot_metabolomics_stratified_profiles_component(factors,group,axes,comp,plot_title=False,plot_legend=False,plot_xticks=False,time_points=None):

    A, B, D = factors
    B2plot = deepcopy(B)

    R = D.shape[1]      # components
    K = D.shape[0]      # subjects
    J = B[0].shape[0]   # time points

    if np.allclose(D,np.ones_like(D)) == False:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    else:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm
                D[k,r] *= normm

            normm2 = tl.norm(A[:,r])
            D[:,r] *= normm2

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    # x-axis
    if J == 8:
        x = np.array(time_points, dtype=float)
        xticks = [0, 1, 2, 3, 4]
    else:
        x = np.array(time_points, dtype=float)
        xticks = [0, 1, 2, 3, 4]

    g = pd.Series(group, dtype="string").fillna("").to_numpy()

    groups = [
        ("noIR-lower BMI",   g == "noIR-lower BMI",   "NoIR, lower BMI",  "tab:blue"),
        ("IR-lower BMI",     g == "IR-lower BMI",     "IR, lower BMI",    "tab:orange"),
        ("IR-higher BMI",    g == "IR-higher BMI",    "IR, higher BMI",   "tab:purple"),  # <-- THIS label
        ("noIR-higher BMI",  g == "noIR-higher BMI",  "NoIR, higher BMI", "tab:cyan"),
    ]

    for key, mask, label, color in groups:
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue

        # Stack subjects in group: shape (n_group, J)
        stack = np.stack([B2plot[k][:, comp] for k in idx], axis=0).astype(float)

        # Center curve
        with np.errstate(invalid='ignore'):
            center = np.nanmean(stack, axis=0)

            # SEM of the mean per time point
            counts = np.sum(np.isfinite(stack), axis=0)
            sd = np.nanstd(stack, axis=0, ddof=1)
            sem = np.where(counts > 1, sd / np.sqrt(counts), 0.0)

        # Shaded band and curve
        axes.fill_between(x, center - sem, center + sem, alpha=0.25, color=color, linewidth=0)
        axes.plot(x, center, label=label, alpha=0.9, color=color, linewidth=1.5)
        axes.set_ylim([0,0.078])
        axes.grid(True)
        axes.tick_params(labelsize=5)

        if plot_xticks is True:
            axes.set_xticks(xticks)
            axes.set_xlabel("Time (hours)",fontsize=5)
        
        if plot_legend is True:
            axes.legend(loc=[0,-0.9],ncols=2,fontsize=5)

        if plot_title is True:
            axes.set_title("Stratified profiles (mean ± SEM)")

def plot_allergens_component(factors,axes,comp,comp_i,allergens,plot_title=False,plot_xticks=False,plot_legend=False):
    A1 = factors[0].copy()  # metabolites x R1
    n_allergens, R1 = A1.shape

    # normalize each column of C1
    for i in range(R1):
        A1[:, i] /= (tl.norm(A1[:, i]))
    
    x = np.arange(n_allergens)
    axes.set_axisbelow(True)

    # draw bars ABOVE the grid
    axes.bar(x, A1[:, comp], tick_label=allergens, zorder=2)

    # draw grid (behind)
    axes.grid(True, zorder=0)
    axes.set_ylabel(f"Comp {comp_i+1}")
    if plot_legend is True and plot_xticks is True:
        axes.set_xticklabels(allergens, rotation=35, 
                                ha="right", fontsize=5)
    else:
        axes.set_xticklabels([])

    axes.tick_params(labelsize=5)

    axes.set_ylabel(f"r={comp_i+1}")
    axes.set_ylim([0,1.0])

    if plot_title is True:
        axes.set_title(r"Allergens ($\mathbf{a}_r$)")

def plot_sensitization_profiles_component(factors,comp,comp_i,axes,plot_legend=True,plot_xticks=False,plot_title=False,time_points=None):

    A, B, D = factors

    B2plot = deepcopy(B)

    R = D.shape[1]          # number of components
    K = D.shape[0]          # number of subjects/sessions
    J = B[0].shape[0]       # number of time points

    if np.allclose(D,np.ones_like(D)) == False:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    else:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm
                D[k,r] *= normm

            normm2 = tl.norm(A[:,r])
            D[:,r] *= normm2

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    x = [*range(len(time_points))]
    
    for subj in range(K):
        axes.grid(True)
        axes.plot(x,B2plot[subj][:, comp], alpha=0.3, color='gray', linewidth=0.7)
        axes.set_ylim([0,0.28])

    if plot_title is True:
        axes.set_title(
            r"Individual profiles $\left\{c_{k,r} \; [\boldsymbol{b}_k]_r\right\}_{k=1}^{K}$"
        )

    if plot_xticks is True:
        if R == 6:
            axes.set_xlabel("Time (hours)",fontsize=5)
        else:
            axes.set_xlabel("Time (years)",fontsize=5)
        if time_points is not None:
            axes.set_xticks(x,time_points,fontsize=5)
    else:
        axes.set_xticks(x,[],fontsize=5)

    axes.tick_params(labelsize=5)

def plot_sensitization_profiles_stratified_component(factors,axes,meta,comp,plot_title=False,plot_legend=False,plot_xticks=False,time_points=None):

    A, B, D = factors

    B2plot = deepcopy(B)

    R = D.shape[1]          # number of components
    K = D.shape[0]          # number of subjects/sessions
    J = B[0].shape[0]       # number of time points

    if np.allclose(D,np.ones_like(D)) == False:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    else:

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm
                D[k,r] *= normm

            normm2 = tl.norm(A[:,r])
            D[:,r] *= normm2

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]
        

    # Precompute indices per BMI category (sorted for deterministic coloring)
    bmi_vals = np.unique(meta)
    group_idx = {bv: np.where(meta == bv)[0] for bv in bmi_vals}

    # Color/label mapping
    def bmi_info(bv):
        if bv == 1:
            return "Natural", 'tab:red'
        elif bv == 2:
            return "C-section", 'tab:blue'
        elif bv == 3:
            return "Vacuum", 'tab:green'

    for bv in bmi_vals:
        # if bv == 3: continue
        idx = group_idx[bv]
        if idx.size == 0:
            continue

        center = np.empty(J, dtype=float)
        sem = np.zeros(J, dtype=float)

        for j in range(J):
            # Collect values across subjects in this BMI group for time j, component r
            vals = np.array([B2plot[k][j, comp] for k in idx], dtype=float)
            # Drop non-finite just in case
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                center[j] = np.nan
                sem[j] = np.nan
                continue
            
            center[j] = np.mean(vals)

            # SEM around the mean (as requested), even if center is the median
            if vals.size > 1:
                sem[j] = np.std(vals, ddof=1) / np.sqrt(vals.size)
            else:
                sem[j] = 0.0

        bmi_label, bmi_color = bmi_info(bv)

        axes.set_axisbelow(True)

        # Shaded SEM band (no label to avoid legend dupes)
        axes.fill_between(np.arange(J), center - sem, center + sem, alpha=0.25, color=bmi_color, linewidth=0)

        # Central curve
        axes.plot(np.arange(J), center, label=bmi_label, alpha=0.9, color=bmi_color, linewidth=1.5)
        axes.tick_params(labelsize=5)
        axes.grid(True,zorder=0)
        axes.set_xticks(np.arange(J),[])
        axes.set_ylim([0,0.06])
        if plot_title is True:
            axes.set_title("Stratified profiles (mean ± SEM)")
        if plot_xticks is True:
            axes.set_xticks(np.arange(J),time_points)
        if plot_legend is True:
            axes.legend(loc=[-0.05,-0.85],ncols=2,fontsize=5)
            axes.set_xlabel("Time (years)",fontsize=5)

def plot_sensitization_mean_profile_component(factors,axes,comp,plot_title=False,plot_legend=False,plot_xticks=False,time_points=None):
    A, B, D = factors
    B2plot = deepcopy(B)

    R = D.shape[1]      # components
    K = D.shape[0]      # subjects
    J = B[0].shape[0]   # time points

    if np.allclose(D,np.ones_like(D)) == False: #parafac2

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    else: # cmf

        for r in range(R):
            for k in range(K):
                normm = tl.norm(B2plot[k][:,r])
                B2plot[k][:,r] /= normm
                D[k,r] *= normm

            normm2 = tl.norm(A[:,r])
            D[:,r] *= normm2

        for r in range(R):
            D[:,r] /= tl.norm(D[:,r])

        for r in range(R):
            for k in range(K):
                B2plot[k][:, r] *= D[k, r]

    x = [*range(J)]

    stack = np.stack([B2plot[k][:, comp] for k in range(K)], axis=0).astype(float)
    with np.errstate(invalid='ignore'):
        center = np.nanmean(stack, axis=0)  
        axes.plot(x, center, alpha=0.9, color='black', linewidth=1.5)

    axes.grid(True)
    axes.set_ylim([0,0.03])

    if plot_xticks is True and plot_legend is True:
        axes.set_xticks(x,time_points)
        axes.set_xlabel("Time (years)",fontsize=5)
    else:
        axes.set_xticks(x,[])
    
    if plot_title is True:
        axes.set_title(f"Mean profile")
