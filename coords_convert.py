import pandas as pd

def sexagesimal_to_decimal(coord_str, is_ra=True):
    """
    Converts a sexagesimal coordinate (as a string "hh:mm:ss.s" or "dd:mm:ss.s")
    to decimal degrees. For RA the result is multiplied by 15.
    """
    try:
        # Split on colon and check that we have exactly three parts
        parts = coord_str.strip().split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid coordinate format: {coord_str}")
        if is_ra:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return (hours + minutes/60 + seconds/3600) * 15
        else:
            # Dec: handle potential negative sign in degrees
            sign = -1 if parts[0].strip().startswith('-') else 1
            degrees = abs(float(parts[0]))
            minutes = float(parts[1])
            seconds = float(parts[2])
            return sign * (degrees + minutes/60 + seconds/3600)
    except Exception as e:
        # If conversion fails, return the original string
        return coord_str

# Read the original CSV file (update the path if needed)
df = pd.read_csv('ASCII_Report_comma_separated.csv')

# Assuming the RA and Dec columns are the 2nd and 3rd columns (or have headers "RA(2000)" and "Dec(2000)")
# Adjust the column names below to match your file’s actual header names.
ra_col = 'RA(2000)'  # change as needed
dec_col = 'Dec(2000)'  # change as needed

# Apply the conversion only to rows with a colon (':') in the coordinate
df[ra_col] = df[ra_col].apply(lambda x: sexagesimal_to_decimal(x, is_ra=True) if isinstance(x, str) and ':' in x else x)
df[dec_col] = df[dec_col].apply(lambda x: sexagesimal_to_decimal(x, is_ra=False) if isinstance(x, str) and ':' in x else x)

# Write back to the same file
df.to_csv('ASCII_Report_comma_separated.csv', index=False)
