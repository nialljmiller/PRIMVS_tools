import pandas as pd
from astropy.coordinates import Angle
import astropy.units as u  # <<<< This was missing in the last script

# Load the OCVC catalog
input_file = "cv_list.csv"  # Replace with actual file path
df = pd.read_csv(input_file)

# Function to convert sexagesimal to decimal degrees
def convert_sexagesimal_to_decimal(ra_str, dec_str):
    try:
        ra = Angle(ra_str, unit=u.hourangle).degree  # Convert RA from H:M:S to degrees
        dec = Angle(dec_str, unit=u.deg).degree  # Convert Dec from D:M:S to degrees
        return ra, dec
    except Exception as e:
        print(f"Error converting: {ra_str}, {dec_str} -> {e}")
        return None, None

# Apply conversion to the DataFrame
df[["RA (deg)", "Dec (deg)"]] = df.apply(lambda row: convert_sexagesimal_to_decimal(row["RA"], row["Dec"]), axis=1, result_type="expand")

# Save the corrected CSV
output_file = "cv_list_with_decimal_coords.csv"
df.to_csv(output_file, index=False)
print(f"Formatted CSV saved to {output_file}")
