"""
Step4_Corr_ana.py
Reproduce correlation plots between estimated parameters (alpha, sigma) and
sleepiness scales (Stanford SSS, Karolinska KSS) for the Pre session.

Inputs
------
- EEG directory containing files named like:
    sub01_nap_lixin.mat
    sub15 exp1_noon_maosiqi.mat
- SleepScale.xlsx (second column = subject 'name';
  columns 5:7 = SSS Pre/Post/After, columns 8:10 = KSS Pre/Post/After)
- ./est_param/param_real.npy produced by your estimation step

Output
------
- ../figures/fig8_corr.png   (1×2 panel: SSS alpha/sigma)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# ----------------------- USER-PATHS (edit if needed) ------------------------ #
LS_DATA_DIR = r"../data/EEG"
SCALE_FILE = r"../data/Scale/SleepScale.xlsx"
PARAM_NPY = r"../est_param/param_real.npy"
OUT_FIG   = r"../figures/fig8_corr.png"

# ---------------------------------------------------------------------------- #

# Fonts / style (matches your previous script)
plt.rcParams["font.family"] = "Arial"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Arial"
plt.rcParams.update({'font.size': 14})


# --- Helpers ---------------------------------------------------------------- #
PAT = re.compile(
    r"sub\d+[ _](?:exp[12]|sleep|nap|rest)(?:_[^_]+)*_(\w+)\.mat",
    re.IGNORECASE,
)

def _find_data_files(base_dir: str, recursive: bool = True):
    """Return sorted list of .mat filenames in base_dir that match PAT."""
    found = []
    if recursive:
        for root, _, files in os.walk(base_dir):
            for f in files:
                if f.lower().endswith(".mat") and PAT.search(f):
                    found.append(os.path.join(root, f))
    else:
        for f in os.listdir(base_dir):
            if f.lower().endswith(".mat") and PAT.search(f):
                found.append(os.path.join(base_dir, f))
    found = sorted(found, key=lambda p: os.path.basename(p).lower())
    return found


def build_scale_df(ls_data_dir: str, stage_file: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Scan EEG dir, extract subject names from filenames, and build a tidy DF of
    both Stanford (SSS) and Karolinska (KSS) scales across sessions.
    Returns:
      scale_df: columns = [sub_id, session, type_scale, scale_value]
      subjects: list[str] subject names in row order
    """
    files = _find_data_files(ls_data_dir, recursive=True)
    if not files:
        raise RuntimeError(f"No matching .mat files found under: {ls_data_dir}")

    # Subject names in the order we’ll use everywhere else
    subj_names = []
    for f in files:
        m = PAT.search(os.path.basename(f))
        if not m:
            raise ValueError(f"Could not parse subject name from {f}")
        subj_names.append(m.group(1))

    # Load spreadsheet
    result = pd.read_excel(stage_file)
    name_col = result.iloc[:, 1].astype(str).str.strip()  # second column (MATLAB {:,2})

    # Build tidy scale dataframe
    sessions = ["Pre", "Post", "After"]
    scale_labels = ["Stanford", "Karolinska"]  # SSS, KSS
    rows = []

    for i, sname in enumerate(subj_names):
        # Find row in spreadsheet (case-insensitive)
        idxs = name_col.str.lower().eq(sname.lower())
        if not idxs.any():
            raise ValueError(f'Name "{sname}" not found in {stage_file}')
        row_idx = idxs.idxmax()  # first match

        # SSS: MATLAB [5:7] => pandas iloc columns [4:7)
        sss_vals = result.iloc[row_idx, 4:7].to_numpy(dtype=float)
        # KSS: MATLAB [8:10] => pandas iloc columns [7:10)
        kss_vals = result.iloc[row_idx, 7:10].to_numpy(dtype=float)

        for ses, val in zip(sessions, sss_vals):
            rows.append((i, ses, scale_labels[0], float(val)))
        for ses, val in zip(sessions, kss_vals):
            rows.append((i, ses, scale_labels[1], float(val)))

    scale_df = pd.DataFrame(rows, columns=["sub_id", "session", "type_scale", "scale_value"])
    return scale_df, subj_names


def load_params(param_npy: str,
                target_subjects: list[str]) -> pd.DataFrame:
    """
    Load parameter matrix from ./est_param/param_real.npy and return tidy DF:
      columns = [sub_id, param, est_param]
    Accepts either:
      - plain ndarray (n_params × n_subj) or (n_subj × n_params)
      - object npy with keys like {'param_vals', 'files', 'param_names'}

    Aligns row order to `target_subjects` by matching final token in filenames
    if a 'files' list is present; otherwise assumes same order/length.
    """
    data = np.load(param_npy, allow_pickle=True)

    file_list = []
    param_names = None
    if isinstance(data, np.ndarray) and data.dtype == object:
        obj = data.item()
        # common key names used in projects
        param_vals = np.asarray(
            obj.get("param_vals", obj.get("params", obj.get("values")))
        )
        file_list = [os.path.basename(p) for p in obj.get("files", [])]
        param_names = obj.get("param_names", None)
        if param_names is not None:
            param_names = list(param_names)
    else:
        param_vals = np.asarray(data)

    # Normalize shape -> est.shape = (n_subj, n_params)
    if param_vals.ndim != 2:
        raise ValueError(f"param_real.npy must be 2D; got shape {param_vals.shape}")
    if param_vals.shape[0] <= param_vals.shape[1]:
        # rows are params, cols are subjects  -> transpose
        est = param_vals.T
    else:
        est = param_vals

    n_subj, n_params = est.shape
    # Default parameter names if none provided
    if not param_names:
        default = ["alpha", "t0", "sigma", "alpha_prior", "beta", "gamma"]
        param_names = default[:n_params]

    # If we have a file list, try to align subjects via the same regex
    if file_list:
        if len(file_list) != n_subj:
            raise ValueError("Length of 'files' in .npy does not match param rows.")
        # Extract subject tokens from file list
        file_subjects = []
        for f in file_list:
            m = PAT.search(os.path.basename(f))
            if not m:
                # fall back to last underscore token (robust-ish)
                file_subjects.append(os.path.basename(f).split("_")[-1].split(".")[0])
            else:
                file_subjects.append(m.group(1))

        # build reorder index so that file_subjects -> target_subjects
        pos = {s.lower(): i for i, s in enumerate(file_subjects)}
        reorder = []
        for s in target_subjects:
            key = s.lower()
            if key not in pos:
                raise ValueError(f'Subject "{s}" not found among parameter files.')
            reorder.append(pos[key])
        est = est[np.array(reorder)]

    # final sanity check
    if len(target_subjects) != est.shape[0]:
        raise ValueError("Subject count mismatch between scales and parameters.")

    # Tidy dataframe
    records = []
    for sid in range(est.shape[0]):
        for p_idx, pname in enumerate(param_names):
            records.append((sid, pname, float(est[sid, p_idx])))
    est_df = pd.DataFrame(records, columns=["sub_id", "param", "est_param"])
    return est_df


def corr_plot(est_df: pd.DataFrame,
                  scale_df: pd.DataFrame,
                  session: str = "Pre",
                  out_png: str = OUT_FIG):
    """
    Make a 1×2 correlation figure for SSS only (Pre session):
      Left  : alpha vs SSS
      Right : sigma vs SSS
    """
    params_to_plot = ["alpha", "sigma"]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.1)

    for c, param in enumerate(params_to_plot):
        ax = axs[c]

        # select parameter rows and merge with SSS only
        p_df = est_df[est_df["param"] == param]
        dfm = p_df.merge(
            scale_df[scale_df["type_scale"] == "Stanford"],  # SSS
            on="sub_id",
        )

        # pivot SSS to sessions and join the unique parameter value per subject
        pivot = dfm.pivot_table(
            index="sub_id", columns="session", values="scale_value", aggfunc="mean"
        ).reset_index()
        p_unique = p_df[["sub_id", "est_param"]].drop_duplicates().rename(columns={"est_param": param})
        model_df = pivot.merge(p_unique, on="sub_id")

        sns.regplot(
            data=model_df,
            x=param, y=session,
            x_jitter=0.01, y_jitter=0.1,
            ci=95, ax=ax
        )

        # stats (works across SciPy versions)
        corr = spearmanr(model_df[session], model_df[param], nan_policy="omit")
        r = getattr(corr, "correlation", getattr(corr, "statistic", np.nan))
        p = corr.pvalue

        # labels/titles
        sym = r"\alpha" if param == "alpha" else r"\sigma"
        ax.set_xlabel(rf"$\hat{{{sym}}}$")
        ax.set_ylabel(f"Sleepiness ({session})")
        ax.set_title(rf"Correlation between $\hat{{{sym}}}$ and SSS scale")
        ax.text(0.55, 0.95, f"Spearman r = {r:.2f}\nP-value = {p:.3f}",
                transform=ax.transAxes, fontsize=12, va="top")
        ax.set_ylim(0.5, 7)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.show()

# --- Main ------------------------------------------------------------------- #
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    scale_df, subject_order = build_scale_df(LS_DATA_DIR, SCALE_FILE)
    est_df = load_params(PARAM_NPY, subject_order)
    corr_plot(est_df, scale_df, session="Pre", out_png=OUT_FIG)
