# stellar_activity_gp

Modelling stellar activity (RV + RHK) with Gaussian Processes to recover injected/real planetary
signals. Two modelling approaches live here; `planet_finder_model/` is the current working model.

## Layout

```
planet_finder_model/     current working model (stationary multi-series GP kernel)
  main.py                 CLI entry point: load data -> fit cycle -> fit GP (+ planet) -> MCMC
  functions.py             data loading, cycle fit, likelihoods, optimisation, plotting
  CV.py, cross_validate_*.py   cross-validation variants of the pipeline
  data/                    Solar_Data/, HD4628/ input time series
  results/                 saved fits (xbest, sampler, map_params .npy)
  notebooks/               exploratory notebook

non_stationary_gp/        experimental non-stationary GP kernel (nskernel)
  multiGP_functions.py     shared multi-series GP fit/plot helpers
  nskernel/                the non-stationary kernel implementation
    nskernel.py
    test_nsgp.py, test_nsgp_qp.py
    data/                  Solar_Data/, HD4628/, saved kernel/sample arrays
  notebooks/               research progression: data generation -> kernel dev ->
                            validation on synthetic/solar/HD4628 data -> final model

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
5. **Cross-validation**: `CV.py` / `cross_validate_12fold_random*.py` run the same
   pipeline in a 12-fold CV setup to assess planet-detection robustness.

Run the current model with, e.g.:

```bash
cd planet_finder_model
python main.py --path "data/Solar_Data" --star-name Sun --fit-planet --run-mcmc
```

`non_stationary_gp/` explores replacing the stationary kernel with a non-stationary one
(`nskernel/nskernel.py`) to better capture time-varying stellar activity; it is not yet
the production pipeline. Its `notebooks/` trace that work from synthetic data generation
through kernel validation on solar and HD4628 data.

## History

This project was reorganised from two separate working directories (`planet_finder_model/`
and `non_stationary_GP/`) into one repo. `non_stationary_gp/` was originally its own git
repo (`nonstat_gp`, tracking the non-stationary kernel work); that history is preserved here.
