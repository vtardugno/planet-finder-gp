# stellar_activity_gp

Gaussian-process models of stellar activity, fit jointly to RV and log R'HK time series,
for detecting planets in RV data.

## Layout

```
planet_finder_model/
  cyc/          cyc and no_cycle models: functions.py, main_cycle_likelihood.py (single fit)
  nonstat2/     nonstat2 model: functions_nonstat.py, main_nonstat.py (single fit)
  CV/           CV.py (12-fold CV), CV_block.py (contiguous-block CV), mgic.py (MGIC per model)
  inj_rec/      injection-recovery sweeps, MGIC calibration, pooling and plotting
  data/         input data, one folder per star (Solar_Data/, HD4628/)
  results/      saved fits and MGIC thresholds (results/mgic_calibration/)
archive/        old models and experiments, kept for reference
```

## Models

- `cyc`: stationary multi-series GP + activity cycle (linear trend + sinusoid) in the mean.
- `no_cycle`: the same GP with no cycle term (`cyc/main_cycle_likelihood.py --no-fit-cycle`).
- `nonstat2`: the cycle scales both the mean and the GP variance.

## Requirements

```bash
pip install numpy scipy pandas matplotlib astropy scikit-learn spleaf emcee corner
```

## Data

Each star has a folder `data/<Star>/` containing `Analyse_summary.csv` and `Analyse_ccf.p`.
Select it with `--path data/<Star> --star-name <Star>` (default: the Sun).

## Usage

Every script takes `--help`. Output folders are not tracked, so create them once after cloning:

```bash
cd planet_finder_model
mkdir -p cyc/results_cyc cyc/results_nocyc nonstat2/results_nonstat results/mgic_calibration
```

Single fits are run from inside their model folder and write to its `results_*/` folders.
By default they inject a test planet and fit GP + planet; use `--no-inject-planet --no-fit-planet`
for a planet-free fit and `--run-mcmc` for MCMC.

```bash
cd cyc
python main_cycle_likelihood.py --path ../data/Solar_Data --star-name Sun   # cyc
python main_cycle_likelihood.py --path ../data/Solar_Data --no-fit-cycle    # no_cycle
cd ../nonstat2
python main_nonstat.py --path ../data/Solar_Data                            # nonstat2
```

Everything else is run from `planet_finder_model/`:

```bash
# Cross-validation of all three models
python CV/CV.py
python CV/CV_block.py

# Injection-recovery sweep, parameter match only (no MGIC): recovered = P and K match the
# injected values. Resumable: re-run the same command to continue.
python inj_rec/injection_recovery.py --models cyc,nonstat2,no_cycle --period-search gp \
    --output-csv inj_rec/ir.csv

# Same sweep, but recovery also requires Delta MGIC > threshold (a separate script).
# Calibrate the per-model thresholds on the uninjected Sun first.
python inj_rec/mgic_calibration.py --period-search gp --threshold-path results/mgic_calibration/mgic_threshold.json
python inj_rec/injection_recovery_mgic.py --models cyc,nonstat2,no_cycle --period-search gp \
    --mgic-threshold-path results/mgic_calibration/mgic_threshold.json

# Re-apply MGIC thresholds to an existing CSV without refitting
python inj_rec/rescore_mgic.py --input-csv inj_rec/injection_recovery_mgic.csv \
    --threshold-path results/mgic_calibration/mgic_threshold.json --output-csv inj_rec/rescored.csv

# Merge CSVs from several runs, then plot recovery fractions
python inj_rec/pool_results.py "inj_rec/ir_*.csv" --output inj_rec/pooled.csv
python inj_rec/plot_injection_recovery.py --input-csv inj_rec/pooled.csv --output inj_rec/grid.png
# (for an MGIC CSV, --param-match-only plots the parameter-match recovery and ignores the MGIC check)
```

A planet counts as recovered when the fitted P is within 10% and K within 15% of the injected
values. `injection_recovery_mgic.py` also requires the RV ΔMGIC (GP-only minus GP+planet) to
exceed the threshold.
