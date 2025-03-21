import pandas as pd
from astroquery.mast import Observations

# Load the crossmatch results produced by your pipeline
results_file = "../PRIMVS/cv_results/tess_crossmatch_results/tess_crossmatch_results.csv"
results = pd.read_csv(results_file)

# Filter to those objects that have a TIC match (in_tic == True)
tic_ids = results[results['in_tic'] == True]['tic_id']

# Iterate over each TIC ID and query for TESS lightcurve products
for tic_id in tic_ids:
    tic_id_int = int(tic_id)  # ensure it's an integer
    try:
        # Query TESS observations using the TIC id as target_name
        obs = Observations.query_criteria(target_name=f"TIC {tic_id_int}", obs_collection='TESS')
        # If no products are returned, print a statement
        if len(obs) == 0:
            print(f"TIC {tic_id_int}: No TESS lightcurve data found.")
        else:
            print(f"TIC {tic_id_int}: TESS data exists. Check manually: {len(obs)} observations returned.")
    except Exception as e:
        print(f"TIC {tic_id_int}: Error encountered - {e}")
