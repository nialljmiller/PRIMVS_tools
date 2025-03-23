import pandas as pd

def sexigesimal_to_decimal(coord_str, is_ra=True):
    """
    Converts a sexigesimal coordinate (string format "hh:mm:ss.ss" or "dd:mm:ss.ss")
    into decimal degrees. For RA, multiplies by 15.
    """
    # Split on colon
    parts = coord_str.split(':')
    if len(parts) != 3:
        raise ValueError(f"Unexpected coordinate format: {coord_str}")
    
    # For RA, parts are hours, minutes, seconds
    if is_ra:
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        decimal = (hours + minutes / 60 + seconds / 3600) * 15
    else:
        # For Dec, account for the sign on degrees
        deg_str = parts[0].strip()
        sign = -1 if deg_str.startswith('-') else 1
        degrees = abs(float(deg_str))
        minutes = float(parts[1])
        seconds = float(parts[2])
        decimal = sign * (degrees + minutes / 60 + seconds / 3600)
    return decimal

# Load your CSV file
df = pd.read_csv('../PRIMVS/cv_data.csv')

# Convert RA and Dec columns. Adjust the column names if needed.
df['RA_decimal'] = df['RA(2000)'].apply(lambda x: sexigesimal_to_decimal(x, is_ra=True))
df['Dec_decimal'] = df['Dec(2000)'].apply(lambda x: sexigesimal_to_decimal(x, is_ra=False))

# Save the new CSV with the added decimal coordinate columns
df.to_csv('your_output_file.csv', index=False)

print("Conversion complete! Your decimal coords are in 'your_output_file.csv'")
