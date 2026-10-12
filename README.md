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

Run everything from `planet_finder_model/`. Every script takes `--help`.

```bash
# Single fit (GP + planet; --no-fit-planet for GP only, --run-mcmc for MCMC)
python cyc/main_cycle_likelihood.py --path data/Solar_Data --star-name Sun
python cyc/main_cycle_likelihood.py --no-fit-cycle          # no_cycle model
python nonstat2/main_nonstat.py --period-search gp

# Cross-validation of all three models
python CV/CV.py
python CV/CV_block.py

# Injection-recovery sweep (resumable: re-run the same command to continue)
python inj_rec/injection_recovery.py --models cyc,nonstat2,no_cycle --period-search gp \
    --output-csv inj_rec/ir.csv

# Same, but also requiring a Delta MGIC detection; calibrate thresholds on the uninjected Sun first
python inj_rec/mgic_calibration.py --period-search gp --threshold-path results/mgic_calibration/mgic_threshold.json
python inj_rec/injection_recovery_mgic.py --models cyc,nonstat2,no_cycle --period-search gp \
    --mgic-threshold-path results/mgic_calibration/mgic_threshold.json

# Re-apply MGIC thresholds to an existing CSV without refitting
python inj_rec/rescore_mgic.py --input-csv inj_rec/injection_recovery_mgic.csv \
    --threshold-path results/mgic_calibration/mgic_threshold.json --output-csv inj_rec/rescored.csv

# Merge CSVs from several runs, then plot recovery fractions
python inj_rec/pool_results.py "inj_rec/ir_*.csv" --output inj_rec/pooled.csv
python inj_rec/plot_injection_recovery.py --input-csv inj_rec/pooled.csv --output inj_rec/grid.png
```

A planet counts as recovered when the fitted P is within 10% and K within 15% of the injected
values. `injection_recovery_mgic.py` also requires the RV ΔMGIC (GP-only minus GP+planet) to
exceed the threshold.
