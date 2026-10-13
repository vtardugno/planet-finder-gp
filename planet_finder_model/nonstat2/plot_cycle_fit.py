import functions_nonstat as mf

t, y, yerr, si = mf.load_and_norm_data("data/Solar_Data", "Sun", normalise=False, inject_planet=False)
mf.fit_cycle(t, y, si, b0=0.0, P0=4000.0, phi0=0.0, print_results=True, plot=True, output_name="results_nonstat/cycle_fit_sun_fit_nonstat2_paper.png")
