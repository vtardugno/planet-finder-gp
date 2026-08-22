# stellar_activity_gp

Modelling stellar activity (RV + RHK) with Gaussian Processes to recover injected/real planetary
signals. Several modelling approaches live here; `planet_finder_model/main.py` + `functions.py`
is the current working model.

## Layout

```
planet_finder_model/     current working model (stationary multi-series GP kernel)
  main.py                 CLI entry point: load data -> fit cycle -> fit GP (+ planet) -> MCMC
  functions.py             data loading, cycle fit, likelihoods, optimisation, plotting
  CV.py                    12-fold cross-validation + full-dataset AIC/BIC model comparison
  data/                    Solar_Data/, HD4628/ input time series
  results/                 saved fits (xbest, sampler, map_params .npy)
  notebooks/               exploratory notebook

  multigp_cycle_variant/   alternative model: stationary kernel + a warped-sine cyclic
                           term added to the *mean* (not the covariance) -- a parallel
                           approach to functions.py's cycle-subtraction, not a non-
                           stationary kernel
    multiGP_functions.py
    notebooks/             multiGP_run.ipynb, try_stuff.ipynb, try_stuff_HD4628.ipynb

  rot_cyc_kernel_variant/  alternative model: additive sum of two kernel terms (rotation
                           SHOKernel + a separate cycle SHOKernel), not a non-stationary
                           kernel either
    notebooks/             rot_cyc_gp.ipynb (also contains an unrelated scratch section
                           exploring a custom G+Gdot kernel)

non_stationary_gp/        the true non-stationary GP kernel: covariance is
                           SimpleProductKernel(NonStationaryKernel x rotation kernel),
                           i.e. the process's variance itself is modulated by activity-
                           cycle phase -- what this folder is actually named for
  nskernel/                the non-stationary kernel implementation
    nskernel.py            NonStationaryKernel class (spleaf term.Kernel subclass)
    test_nsgp.py, test_nsgp_qp.py
    data/                  Solar_Data/, HD4628/, saved kernel/sample arrays, synthetic
                           realization .sav files (also used by rot_cyc_kernel_variant/)
  notebooks/               research progression: data generation (gendata_from_nsk) ->
                           kernel dev (nonstat_kernel, nonstat_kerne_GOODl) -> applied to
                           real data (sun_data_nonstat, hd4628_fit, sun_data_inject,
                           sun_finalGP_model) -> validation (nonstat_kern_val,
                           sun_nonstat_kern_val)

archive/pyaneti_stuff/    old pyaneti MCMC test runs, kept for reference only
```

## Workflow

1. **Data**: RV + RHK (activity indicator) time series per star, stored as
   `Analyse_summary.csv` / `Analyse_ccf.p` under a `data/<Star>/` folder.
2. **Cycle removal**: `functions.fit_cycle` fits and subtracts a shared long-term
   activity cycle from RV and RHK before GP fitting (`--cycle-fit`, on by default).
3. **GP fit**: a multi-series GP kernel (`spleaf`) is fit jointly to RV and RHK to model
   correlated stellar activity, optionally with an injected/candidate planet signal
   (`--fit-planet`, `--planet-*` args).
4. **Optimisation + MCMC**: `main.py` optimises the GP hyperparameters (multiple planet
   amplitude/phase initial guesses, best log-likelihood kept), then runs `emcee` for
   posterior sampling, saving corner plots and MAP parameters to `results/`.
5. **Cross-validation + model comparison**: `CV.py` runs a 12-fold CV comparing
   cycle-removed vs. non-cycle-removed fits (per-fold results in `results/cv_results.csv`),
   and separately reports full-dataset AIC/BIC for the same comparison.

Run the current model with, e.g.:

```bash
cd planet_finder_model
python main.py --path "data/Solar_Data" --star-name Sun --fit-planet --run-mcmc
```

`non_stationary_gp/` explores replacing the stationary kernel with a genuinely
**non-stationary** one (`nskernel/nskernel.py`'s `NonStationaryKernel`, multiplied into the
covariance) to better capture time-varying stellar activity; it is not yet the production
pipeline. `planet_finder_model/multigp_cycle_variant/` and `rot_cyc_kernel_variant/` explore
two *other* ways of handling the activity cycle (additive mean modulation, and a separate
additive cycle kernel term, respectively) -- despite the folder history, neither of these
uses a non-stationary kernel, so they live alongside the current working model rather than
in `non_stationary_gp/`.

## History

This project was reorganised from two separate working directories (`planet_finder_model/`
and `non_stationary_GP/`) into one repo. `non_stationary_gp/` was originally its own git
repo (`nonstat_gp`, tracking the non-stationary kernel work); that history is preserved here.
`multigp_cycle_variant/` and `rot_cyc_kernel_variant/` were later moved out of
`non_stationary_gp/` into `planet_finder_model/` once it became clear they don't actually
use the non-stationary kernel technique.
