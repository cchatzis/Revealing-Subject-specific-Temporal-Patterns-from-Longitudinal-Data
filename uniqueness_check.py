import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import matcouply
import pickle
import tlviz

import numpy as np
import tensorly as tl
import scipy.io as sio


from copy import deepcopy
from tqdm import tqdm
from multiprocessing import Pool
from tensorly.decomposition import non_negative_parafac
from matcouply.penalties import (
    NonNegativity,
    Parafac2,
)
from matcouply.decomposition import cmf_aoadmm
from matcouply.decomposition import initialize_cmf
from tlviz.factor_tools import degeneracy_score


def worktask_cp(args):

    data = args["data"]
    l_B = args["l_B"]
    init_no = args["init_no"]
    no_of_components = args["no_of_components"]
    mask = args["mask"]
    scale = args["feasibility_penalty_scale"]
    tol = args["tol"]
    state = args["state"]

    factors, errors = non_negative_parafac(
        tensor=data,
        tol=tol,
        rank=no_of_components,
        return_errors=True,
        n_iter_max=8000,
        mask=mask,
        random_state=init_no * state,
    )

    if len(errors) > 7990:
        factors = None
        loss = None
    else:
        factors = (factors[1][2], factors[1][1], factors[1][0])
        loss = errors[-1]

    return (factors, loss)


def worktask_cmf(args):
    data = args["data"]
    l_B = args["l_B"]
    init_no = args["init_no"]
    no_of_components = args["no_of_components"]
    mask = args["mask"]
    scale = args["feasibility_penalty_scale"]
    tol = args["tol"]
    state = args["state"]

    weights, (A_init, B_init, D_init) = initialize_cmf(
        matrices=data,
        rank=no_of_components,
        init="random",
        svd_fun="truncated_svd",
        random_state=init_no,
    )

    A_init = np.ones_like(A_init)
    cmf_init = weights, (A_init, B_init, D_init)

    (weights, (D, B, A)), run_diagnostics = cmf_aoadmm(
        matrices=data,
        rank=no_of_components,
        return_errors=True,
        n_iter_max=15000,
        regs=[[NonNegativity()], [NonNegativity()], [NonNegativity()]],
        l2_penalty=[l_B, l_B, l_B],
        tol=tol,
        init=cmf_init,
        update_A=False,
        feasibility_penalty_scale=scale,
        inner_n_iter_max=20,
        absolute_tol=1e-6,
        feasibility_tol=1e-5,
        inner_tol=1e-5,
        mask=mask,
        random_state=init_no * state,
    )

    # print(len(run_diagnostics.rec_errors),run_diagnostics.regularized_loss[-1],run_diagnostics.message)

    if len(run_diagnostics.regularized_loss) > 14990:
        factors = None
        loss = None
    else:
        factors = (D, B, A)
        loss = run_diagnostics.regularized_loss[-1]

    return (factors, loss)


def worktask_parafac2(args):
    data = args["data"]
    l_B = args["l_B"]
    init_no = args["init_no"]
    no_of_components = args["no_of_components"]
    mask = args["mask"]
    scale = args["feasibility_penalty_scale"]
    tol = args["tol"]
    state = args["state"]

    (weights, (D, B, A)), run_diagnostics = cmf_aoadmm(
        matrices=data,
        rank=no_of_components,
        return_errors=True,
        n_iter_max=15000,
        regs=[[NonNegativity()], [Parafac2(), NonNegativity()], [NonNegativity()]],
        l2_penalty=[l_B, l_B, l_B],
        tol=tol,
        feasibility_penalty_scale=scale,
        inner_n_iter_max=20,
        absolute_tol=1e-6,
        feasibility_tol=1e-5,
        inner_tol=1e-5,
        mask=mask,
        random_state=init_no * state,
    )

    # print(len(run_diagnostics.rec_errors),run_diagnostics.regularized_loss[-1])

    if len(run_diagnostics.regularized_loss) > 14990:
        factors = None
        loss = None
    else:
        factors = (D, B, A)
        loss = run_diagnostics.regularized_loss[-1]

    return (factors, loss)


if __name__ == "__main__":

    model = sys.argv[1]
    no_of_components = int(sys.argv[2])  # 10
    l_B = float(sys.argv[3])  # 0.01
    dataset = str(sys.argv[4])  # -m: Metabolomics, -s: Sensitization

    if dataset == "-m":

        if len(sys.argv) < 6:
            raise ValueError("Missing gender argument after -m. Use: -m males|females")

        gender = sys.argv[5].lower()

        if gender in ("m", "male", "males"):
            data_object = sio.loadmat("Z_males.mat", simplify_cells=True)["Z"]
        elif gender in ("f", "female", "females"):
            data_object = sio.loadmat("Z_females.mat", simplify_cells=True)["Z"]

        K = len(data_object["object"])

        data = np.empty(
            (data_object["object"][0].shape[0], data_object["object"][0].shape[1], K)
        )

        mask = np.empty_like(data)

        for k in range(K):
            data[:, :, k] = data_object["object"][k]
            mask[:, :, k] = data_object["miss"][k]

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(K):
                    if mask[i, j, k] == 0:
                        data[i, j, k] = np.nan

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(K):
                    if mask[i, j, k] == 0:
                        data[i, j, k] = np.nanmean(data[i, j, :])

        args = []

        for init_no in range(50):

            args.append(
                {
                    "init_no": init_no,
                    "data": deepcopy(data.T),
                    "l_B": l_B,
                    "no_of_components": no_of_components,
                    "mask": mask.T,
                    "tol": 1e-6,
                    "feasibility_penalty_scale": 1,
                    "state": 17,
                }
            )

    elif dataset == "-s":

        data = sio.loadmat("Sensitization/sensitization_data.mat", simplify_cells=True)[
            "data"
        ]

        data = tl.transpose(data, (2, 1, 0))

        data = data / tl.norm(data)

        args = []

        for init_no in range(50):

            args.append(
                {
                    "init_no": init_no,
                    "data": deepcopy(data.T),
                    "l_B": l_B,
                    "no_of_components": no_of_components,
                    "mask": None,
                    "tol": 1e-8,
                    "feasibility_penalty_scale": 10,
                    "state": 133,
                }
            )

    print(f"{model}-{no_of_components}-{l_B}:{data.shape} ({dataset})")

    pool = Pool(50)

    if model == "parafac2":
        results_iter = pool.imap_unordered(worktask_parafac2, args)
    elif model == "cmf":
        results_iter = pool.imap_unordered(worktask_cmf, args)
    elif model == "cp":
        results_iter = pool.imap_unordered(worktask_cp, args)
    else:
        raise ValueError(
            f"Unknown model: {model!r}. Expected one of: 'parafac2', 'cmf', 'cp'."
        )

    results = list(tqdm(results_iter, total=50, desc="Running jobs", unit="job"))

    pool.close()
    pool.join()

    results = [r for r in results if r[1] is not None]  # remove Nones
    results = sorted(results, key=lambda x: x[1])
    print(f"{len(results)} runs finished.")

    # save results in a pickle file
    if dataset == "-m":
        if gender in ("m", "male", "males"):
            with open(
                f"Metabolomics/results/uniqueness/factors_{model}_{no_of_components}_components_l_B_{l_B}_males.pkl",
                "wb",
            ) as f:
                pickle.dump(results, f)
        elif gender in ("f", "female", "females"):
            with open(
                f"Metabolomics/results/uniqueness/factors_{model}_{no_of_components}_components_l_B_{l_B}_females.pkl",
                "wb",
            ) as f:
                pickle.dump(results, f)
    elif dataset == "-s":
        with open(
            f"Sensitization/results/uniqueness/factors_{model}_{no_of_components}_components_l_B_{l_B}.pkl",
            "wb",
        ) as f:
            pickle.dump(results, f)
