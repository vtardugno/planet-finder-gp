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
  notebooks/               exploratory notebook (functions.py-based)

  multigp_cycle_variant/   alternative model: stationary kernel + a warped-sine cyclic
                           term added to the *mean* (not the covariance) -- a parallel
                           approach to functions.py's cycle-subtraction, not a non-
                           stationary kernel
    multiGP_functions.py    alpha/negloglike/plot_fit/load_and_norm_data helpers
    notebooks/
      run_model.ipynb        the one notebook to run this model and see plots -- set
                             STAR ("Sun"/"HD4628") and FIT_PLANET at the top. The Sun
                             preset is verified to reproduce multiGP_run.ipynb's and
                             sun_data_inject.ipynb's results (both since removed); the
                             HD4628 preset is NOT a verified match for hd4628_fit.ipynb
                             (see below) but produces qualitatively similar fits.
      hd4628_fit.ipynb        kept separately, not consolidated into run_model.ipynb:
                             it loads data normalised with unshifted absolute time and
                             fits a planet with its own hand-rolled K/phi negloglike
                             (not multiGP_functions.py's A/B convention), so
                             run_model.ipynb's HD4628 preset -- built on the shared
                             mf.negloglike -- can't exactly reproduce it. Both show the
                             same qualitative behaviour (RHK fits reasonably, RV fit is
                             weak), but that hasn't been confirmed to be a quantitative
                             match, so this notebook stays as the reference instead of
                             being deleted.

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
    nskernel.py             NonStationaryKernel class (spleaf term.Kernel subclass)
    test_nsgp.py, test_nsgp_qp.py
    data/                   Solar_Data/, HD4628/, saved kernel/sample arrays, synthetic
                            realization .sav files (also used by rot_cyc_kernel_variant/)
  notebooks/
    sun_data_nonstat.ipynb   the one notebook applying the true non-stationary kernel to
                             real data -- currently Sun only (star_name/data_path are
                             variables at the top, ready to point at another star's data
                             when/if that's fit)
    gendata_from_nsk.ipynb   generates synthetic data from the model (parameter recovery)
    nonstat_kernel.ipynb, nonstat_kerne_GOODl.ipynb
                             kernel development on synthetic data
    nonstat_kern_val.ipynb, sun_nonstat_kern_val.ipynb
                             validation: compares the non-stationary kernel against
                             simpler baselines (deliberately uses more than one
                             technique -- that's the point of a validation notebook)
    sun_finalGP_model.ipynb  a 3-series (rv/tom_f/tom_v) variant, structurally different
                             from everything else here -- left as-is, unreviewed

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
use the non-stationary kernel technique. `multigp_cycle_variant/run_model.ipynb`
consolidates near-duplicate per-star notebooks that hand-rolled the same alpha-in-mean
technique into one notebook parameterized by star name. Consolidating exposed a real bug:
an early version fixed the kernel's `sig` parameter at `np.var(y_full)` for every star,
but the original Sun notebook actually fixed it at `1.0` throughout optimisation/MCMC
(only substituting the empirical variance for the final plot) -- using the wrong value
collapsed the RHK fit. After matching the original recipe exactly (sig, bounds, and a
non-zero parameter warm-start), the Sun preset was confirmed to reproduce the original
notebooks' results, and `multiGP_run.ipynb`/`sun_data_inject.ipynb` were removed.
`hd4628_fit.ipynb` was kept rather than removed: it turned out to differ from the shared
`multiGP_functions.py` pipeline more deeply (normalised data, unshifted time, a bespoke
K/phi planet fit), so `run_model.ipynb`'s HD4628 preset -- while qualitatively similar --
isn't a verified match for it the way the Sun preset is.
