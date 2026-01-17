import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import pickle
import matcouply
import tlviz

import scipy.io as sio
import numpy as np
import tensorly as tl
import pandas as pd
from tqdm import tqdm

from matcouply.penalties import NonNegativity, Parafac2, GeneralizedL2Penalty#1,NonNegativeGL
from matcouply.decomposition import cmf_aoadmm
from matcouply.decomposition import initialize_cmf
from copy import deepcopy
from multiprocessing import Pool
from tensorly.decomposition import non_negative_parafac
from tlviz.factor_tools import factor_match_score, degeneracy_score
from tensorly.metrics import congruence_coefficient

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold


def preprocess_centerscale(X, flag_scale=True, flag_center=True):
    """
    Preprocess a 3D array (subjects × time × metabolites):
    - Center across subjects (mode-1)
    - Scale within each metabolite (mode-3) using RMS
    - Handles NaNs

    Parameters:
    - X: np.ndarray, shape (subjects, time, metabolites)
    - flag_scale: bool, whether to apply RMS scaling
    - flag_center: bool, whether to apply centering

    Returns:
    - Xpre: np.ndarray, preprocessed version of X
    """
    Xpre = np.copy(X)

    if flag_center:
        mean_subjects = np.nanmean(Xpre, axis=0)
        for i in range(Xpre.shape[0]):
            Xpre[i, :, :] = Xpre[i, :, :] - mean_subjects

    if flag_scale:
        for k in range(Xpre.shape[2]):
            rms = np.sqrt(np.nanmean(Xpre[:, :, k] ** 2))
            if rms != 0:
                Xpre[:, :, k] /= rms
            else:
                Xpre[:, :, k] = 0

    return Xpre


def worktask_cp(args):
    r = args["r"]
    s = args["s"]
    data = args["data"]
    l_B = args["l_B"]
    no_of_components = args["no_of_components"]
    mask = args["mask"]
    tol = args["tol"]
    feasibility_penalty_scale = args["feasibility_penalty_scale"]

    best_error = np.inf
    best_factors = None

    for init in range(40):

        # print(data.shape)

        factors, errors = non_negative_parafac(
            tensor=data,
            tol=tol,
            rank=no_of_components,
            return_errors=True,
            n_iter_max=28000,
            mask=mask,
            random_state=init + r * 100 + s * 1000000
        )

        if degeneracy_score(factors) > -0.85 and len(errors) < 8000:
            if errors[-1] < best_error:
                best_error = errors[-1]
                best_factors = factors[1]

        # print(data.shape,factors[1][0].shape,factors[1][1].shape,factors[1][2].shape)

    return {
        "r": r,
        "s": s,
        "factors": best_factors,
        "sorted_train_index": args["sorted_train_index"],
    }


def worktask_cmf(args):
    r = args["r"]
    s = args["s"]
    data = args["data"]
    l_B = args["l_B"]
    no_of_components = args["no_of_components"]
    mask = args["mask"]
    tol = args["tol"]
    feasibility_penalty_scale = args["feasibility_penalty_scale"]

    best_error = np.inf
    best_factors = None

    for init in range(40):

        weights, (A_init, B_init, D_init) = initialize_cmf(
            matrices=data,
            rank=no_of_components,
            init="random",
            svd_fun="truncated_svd",
        )

        A_init = np.ones_like(A_init)
        cmf_init = weights, (A_init, B_init, D_init)

        (weights, (D, B, A)), run_diagnostics = cmf_aoadmm(
            matrices=data,
            rank=no_of_components,
            return_errors=True,
            n_iter_max=8000,
            regs=[[NonNegativity()], [NonNegativity()], [NonNegativity()]],
            l2_penalty=[l_B, l_B, l_B],
            tol=tol,
            init=cmf_init,
            update_A=False,
            inner_n_iter_max=20,
            absolute_tol=1e-6,
            feasibility_tol=1e-5,
            inner_tol=1e-5,
            mask=mask,
            random_state=init + r * 100 + s * 1000000,
        )

        if (
            len(run_diagnostics.regularized_loss) > 7990
        ):
            continue
        else:
            if run_diagnostics.regularized_loss[-1] < best_error:
                best_error = run_diagnostics.regularized_loss[-1]
                best_factors = (D, B, A)

    return {
        "r": r,
        "s": s,
        "factors": best_factors,
        "sorted_train_index": args["sorted_train_index"],
    }


def worktask_parafac2(args):
    r = args["r"]
    s = args["s"]
    data = args["data"]
    l_B = args["l_B"]
    no_of_components = args["no_of_components"]
    mask = args["mask"]
    tol = args["tol"]
    feasibility_penalty_scale = args["feasibility_penalty_scale"]

    best_error = np.inf
    best_factors = None

    for init in range(40):

        try:

            # print(data.shape)

            (weights, (D, B, A)), run_diagnostics = cmf_aoadmm(
                matrices=data,
                rank=no_of_components,
                return_errors=True,
                n_iter_max=8000,
                regs=[
                    [NonNegativity()],
                    [NonNegativity(), Parafac2()],
                    [NonNegativity()],
                ],
                l2_penalty=[l_B, l_B, l_B],
                tol=tol,
                feasibility_penalty_scale=feasibility_penalty_scale,
                inner_n_iter_max=20,
                absolute_tol=1e-6,
                feasibility_tol=1e-5,
                inner_tol=1e-5,
                mask=mask,
                random_state=init + r * 100 + s * 1000000,
            )

            # print(A.shape,len(B),B[0].shape,D.shape)

            if (
                len(run_diagnostics.regularized_loss) > 7990
            ):
                continue
            else:
                if run_diagnostics.regularized_loss[-1] < best_error:
                    best_error = run_diagnostics.regularized_loss[-1]
                    best_factors = (D, B, A)

        except Exception as e:
            print(f"Exception occurred: {e}")
            pass

    if best_factors is None:
        print(f"Failed to converge for r={r}, s={s}")

    return {
        "r": r,
        "s": s,
        "factors": best_factors,
        "sorted_train_index": args["sorted_train_index"],
    }


if __name__ == "__main__":

    model = sys.argv[1]
    no_of_components = int(sys.argv[2])  # 10
    l_B = float(sys.argv[3])  # 0.01
    dataset = str(sys.argv[4])  # -m: Metabolomics, -s: Sensitization
    splits = int(sys.argv[5])
    repeats = 10

    if dataset == "-m":

        if len(sys.argv) < 6:
            raise ValueError("Missing gender argument after -m. Use: -m males|females")

        gender = sys.argv[5].lower()

        if gender in ("m", "male", "males"):

            data_object = sio.loadmat("Metabolomics/Z_males_notprocessed.mat", simplify_cells=True)["Z"]
            meta = sio.loadmat("Metabolomics/Z_males_notprocessed.mat", simplify_cells=True)["Xfinal"][
                "class"
            ][0][0][-1]

        elif gender in ("f", "female", "females"):

            data_object = sio.loadmat("Metabolomics/Z_females_notprocessed.mat", simplify_cells=True)["Z"]
            meta = sio.loadmat("Metabolomics/Z_females_notprocessed.mat", simplify_cells=True)["Xfinal"][
                "class"
            ][0][0][-1]

        K = len(data_object["object"])

        data = np.empty(
            (data_object["object"][0].shape[0],
             data_object["object"][0].shape[1],
             K)
        )

        mask = np.empty_like(data)

        for k in range(K):
            data[:,:,k] = data_object["object"][k]
            mask[:,:,k] = data_object["miss"][k]

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(K):
                    if mask[i,j,k] == 0:
                        data[i,j,k] = np.nan

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(K):
                    if mask[i,j,k] == 0:
                        data[i,j,k] = np.nanmean(data[i,j,:])

        data = data.T

        args = []

        rskf = RepeatedStratifiedKFold(
            n_splits=splits, n_repeats=repeats, random_state=42
        )

        for r, (train_index, test_index) in enumerate(
            rskf.split(deepcopy(data), deepcopy(meta))
        ):

            sorted_train_index = sorted(train_index)
            train = data[sorted_train_index]

            mask2use = tl.ones(train.shape)

            train = preprocess_centerscale(train, flag_scale=True, flag_center=False)

            for i in range(train.shape[0]):
                for j in range(train.shape[1]):
                    for k in range(train.shape[2]):
                        if np.isnan(train[i, j, k]):
                            mask2use[i, j, k] = 0
                            train[i, j, k] = np.nanmean(train[:, j, k])

            train = tl.transpose(train, (2, 1, 0))
            mask2use = tl.transpose(mask2use, (2, 1, 0))

            # print(train.T.shape,mask2use.T.shape,data.shape)

            train = train / tl.norm(train)

            args.append(
                {
                    "r": r // splits,
                    "s": r % splits,
                    "data": deepcopy(train.T),
                    "l_B": l_B,
                    "no_of_components": no_of_components,
                    "sorted_train_index": sorted_train_index,
                    "mask": mask2use.T,
                    "tol": 1e-6,
                    "feasibility_penalty_scale": 1,
                }
            )

    elif dataset == "-s":

        data = sio.loadmat("Sensitization/sensitization_data.mat", simplify_cells=True)[
            "data"
        ]

        df = pd.read_csv('Sensitization/metadata.csv')

        meta = df['Delivery(Natural(1)/C-section(2)/Vacuum(3))'].to_numpy()

        args = []

        splitter_cls = RepeatedStratifiedKFold()
        rskf = splitter_cls(n_splits=splits, n_repeats=repeats, random_state=42)

        split_args = (deepcopy(data), deepcopy(meta))

        for r, (train_index, test_index) in enumerate(rskf.split(*split_args)):

            sorted_train_index = sorted(train_index)
            train = data[sorted_train_index]

            # print(train.shape, data.shape)

            for k in range(train.shape[2]):
                temp = train[:, :, k]
                denom = np.sqrt(np.nanmean(temp**2))
                if denom != 0:
                    train[:, :, k] = temp / denom

            mask2use = None

            train = train / tl.norm(train)

            args.append(
                {
                    "r": r // splits,
                    "s": r % splits,
                    "data": deepcopy(train),
                    "l_B": l_B,
                    "no_of_components": no_of_components,
                    "sorted_train_index": sorted_train_index,
                    "mask": mask2use,
                    "tol": 1e-8,
                    "feasibility_penalty_scale": 10,
                }
            )

    print(
        f"{model}-{no_of_components}-{l_B}:{data.shape} ({dataset}) (len(args):{len(args)})"
    )

    if len(args) > 128:
        pool = Pool(65)
    else:
        pool = Pool(65)

    total_jobs = len(args)

    if model == "parafac2":
        results_iter = pool.imap_unordered(worktask_parafac2, args)
    elif model == "cmf":
        results_iter = pool.imap_unordered(worktask_cmf, args)
    elif model == "cp":
        results_iter = pool.imap_unordered(worktask_cp, args)
        results_iter = []

    results = list(
        tqdm(results_iter, total=total_jobs, desc="Running jobs", unit="job")
    )

    pool.close()
    pool.join()

    print(f"{len(results)} runs finished.")

    # save results in a pickle file
    if dataset == "-m":
        with open(
            f"Metabolomics/results/replicability/factors_{model}_{no_of_components}_components_l_B_{l_B}_splits_{splits}_{gender}.pkl",
            "wb",
        ) as f:
            pickle.dump(results, f)            
    elif dataset == "-s":
        with open(
            f"Sensitization/results/replicability/factors_{model}_{no_of_components}_components_l_B_{l_B}_splits_{splits}.pkl",
            "wb",
        ) as f:
            pickle.dump(results, f)
