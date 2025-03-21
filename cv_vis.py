import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import patheffects
from astropy.coordinates import SkyCoord
import astropy.units as u

# -------------------------
# TESSCycle8Overlay class (as provided)
# -------------------------
class TESSCycle8Overlay:
    """TESS Cycle 8 camera footprint visualization for CV candidates"""
    
    def __init__(self):
        # TESS Year 8 camera positions (RA, Dec, Roll) in degrees
        self.camera_positions = {
            97: [
                (24.13, -9.32, 292.44),
                (34.60, -31.26, 296.14),
                (51.45, -51.83, 127.55),
                (90.00, -66.56, 161.25)
            ],
            98: [
                (76.26, 4.75, 275.70),
                (78.71, -19.13, 276.02),
                (82.05, -42.96, 97.78),
                (90.00, -66.56, 104.42)
            ],
            99: [
                (136.70, -10.03, 296.66),
                (149.01, -31.13, 301.08),
                (168.24, -50.40, 133.88),
                (206.58, -62.97, 166.47)
            ],
            100: [
                (160.77, -19.40, 290.79),
                (171.88, -41.46, 296.53),
                (194.15, -61.36, 134.29),
                (251.44, -70.27, 187.40)
            ],
            101: [
                (184.53, -29.77, 289.32),
                (197.12, -51.89, 297.73),
                (231.12, -70.22, 148.05),
                (303.95, -68.82, 217.35)
            ],
            102: [
                (210.05, -39.39, 292.95),
                (228.76, -60.37, 307.55),
                (284.10, -72.46, 179.13),
                (340.57, -60.80, 231.85)
            ],
            103: [
                (239.66, -46.44, 302.50),
                (269.50, -63.95, 327.47),
                (328.67, -66.46, 202.01),
                (5.45, -50.96, 234.00)
            ],
            104: [
                (273.80, -48.76, 317.02),
                (310.89, -60.43, 347.75),
                (357.50, -56.84, 208.16),
                (26.00, -41.28, 230.08)
            ],
            105: [
                (308.41, -45.00, 331.21),
                (343.49, -51.67, 357.83),
                (19.73, -46.57, 205.65),
                (45.61, -32.84, 222.47)
            ],
            106: [
                (338.92, -36.33, 339.59),
                (9.24, -40.96, 358.81),
                (39.96, -37.24, 198.48),
                (65.56, -26.76, 212.26)
            ],
            107: [
                (5.23, -25.55, 341.34),
                (31.87, -30.75, 354.10),
                (59.79, -30.20, 188.47),
                (85.93, -24.07, 200.58)
            ]
        }
        
        # Convert to galactic coordinates
        self.galactic_positions = self._convert_to_galactic()
    
    def _convert_to_galactic(self):
        """Convert equatorial to galactic coordinates"""
        from astropy.coordinates import SkyCoord
        galactic_positions = {}
        
        for sector, cameras in self.camera_positions.items():
            galactic_positions[sector] = []
            for ra, dec, roll in cameras:
                coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                gal = coords.galactic
                l_deg = gal.l.degree
                if l_deg > 180:
                    l_deg -= 360
                galactic_positions[sector].append((l_deg, gal.b.degree, roll))
        return galactic_positions
    
    def add_to_plot(self, ax, focus_region=None, alpha=0.2):
        """
        Add TESS Cycle 8 camera footprints to a Galactic plot.
        """
        camera_colors = ['red', 'purple', 'blue', 'green']
        
        if focus_region == 'bulge':
            sectors = [103, 104, 105]
            camera_indices = [0, 1]
        elif focus_region == 'disk':
            sectors = [102, 103] 
            camera_indices = [1, 2]
        else:
            sectors = list(self.galactic_positions.keys())
            camera_indices = [0, 1, 2, 3]
        
        for sector in sectors:
            for i in camera_indices:
                if i >= len(self.galactic_positions[sector]):
                    continue
                l, b, roll = self.galactic_positions[sector][i]
                self._add_camera_footprint(ax, l, b, roll, 
                                           color=camera_colors[i],
                                           alpha=alpha,
                                           label=f"Camera {i+1}" if f"Camera {i+1}" not in [child.get_label() for child in ax.get_children() if hasattr(child, 'get_label')] else "",
                                           sector=sector)
        # Add legend for cameras
        handles = []
        labels = []
        for i, color in enumerate(camera_colors):
            if i in camera_indices:
                handles.append(plt.Line2D([], [], color=color, marker='s', 
                                          linestyle='None', markersize=10, alpha=0.6))
                labels.append(f'Camera {i+1}')
        if handles and labels:
            ax.legend(handles, labels, loc='upper right', title="TESS Cycle 8")
        return ax
    
    def _add_camera_footprint(self, ax, l, b, roll, color='red', alpha=0.2, label="", sector=None, size=12):
        """Add a single camera footprint on a Galactic plot"""
        vertices_l = np.array([-size, size, size, -size, -size])
        vertices_b = np.array([-size, -size, size, size, -size])
        roll_rad = np.radians(roll-90)
        rotated_l = vertices_l * np.cos(roll_rad) - vertices_b * np.sin(roll_rad)
        rotated_b = vertices_l * np.sin(roll_rad) + vertices_b * np.cos(roll_rad)
        vertices_l = l + rotated_l
        vertices_b = b + rotated_b
        polygon = Polygon(np.column_stack([vertices_l, vertices_b]),
                          alpha=alpha, color=color, closed=True, label=label)
        ax.add_patch(polygon)
        if sector:
            text = ax.text(l, b, str(sector), fontsize=8, ha='center', va='center',
                           color='white', fontweight='bold')
            text.set_path_effects([
                patheffects.Stroke(linewidth=2, foreground='black'),
                patheffects.Normal()
            ])

# -------------------------
# Helper function to overlay TESS footprints in Equatorial coordinates
# -------------------------
def add_tess_overlay_equatorial(ax, alpha=0.2, size=12):
    """
    Overlay TESS Cycle 8 camera footprints on an Equatorial plot.
    Uses the original camera_positions from TESSCycle8Overlay.
    """
    camera_colors = ['red', 'purple', 'blue', 'green']
    # Create an instance to access the original camera_positions
    tess = TESSCycle8Overlay()
    
    for sector, cameras in tess.camera_positions.items():
        for i, (ra, dec, roll) in enumerate(cameras):
            # Create square vertices (before rotation)
            vertices_ra = np.array([-size, size, size, -size, -size])
            vertices_dec = np.array([-size, -size, size, size, -size])
            roll_rad = np.radians(roll-90)
            rotated_ra = vertices_ra * np.cos(roll_rad) - vertices_dec * np.sin(roll_rad)
            rotated_dec = vertices_ra * np.sin(roll_rad) + vertices_dec * np.cos(roll_rad)
            vertices_ra = ra + rotated_ra
            vertices_dec = dec + rotated_dec
            polygon = Polygon(np.column_stack([vertices_ra, vertices_dec]),
                              alpha=alpha, color=camera_colors[i % len(camera_colors)],
                              closed=True)
            ax.add_patch(polygon)
            # Add sector label at the camera center
            ax.text(ra, dec, str(sector), fontsize=8, ha='center', va='center',
                    color='white', fontweight='bold',
                    path_effects=[patheffects.Stroke(linewidth=2, foreground='black'),
                                  patheffects.Normal()])
    # Add a simple legend manually
    handles = [plt.Line2D([], [], color=color, marker='s', linestyle='None', markersize=10, alpha=0.6) for color in camera_colors]
    labels = [f'Camera {i+1}' for i in range(len(camera_colors))]
    ax.legend(handles, labels, loc='upper right', title="TESS Cycle 8")
    return ax







# Load your CV candidates data (update the file path as needed)
df = pd.read_csv('../PRIMVS/cv_results/cv_candidates.csv')

# Adjust your data from [0, 360) to [-180, 180)
df['l'] = df['l'].apply(lambda x: x - 360 if x > 180 else x)

# Ensure 'cv_prob' exists (output from your classifier)
if 'cv_prob' not in df.columns:
    raise ValueError("cv_prob column not found. Run your CV pipeline first.")

# Make sure known CVs are flagged; if not, mark all as unknown (this is your fault, so fix it!)
if 'is_known_cv' not in df.columns:
    df['is_known_cv'] = False

# For ROC curve, we need binary labels: 1 for known CV, 0 for others
df['true_label'] = df['is_known_cv'].astype(int)

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(df['true_label'], df['cv_prob'])
roc_auc = auc(fpr, tpr)

# Determine optimal threshold using Youden's J statistic (maximizes TPR - FPR)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal ROC threshold: {optimal_threshold:.2f}")

# -------------------------
# Plot 1: ROC Curve
# -------------------------
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--', color='grey')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f'Optimal threshold = {optimal_threshold:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for CV Classifier')
plt.legend()
plt.grid(True)
plt.savefig("../PRIMVS/cv_results/roc_curve.png", dpi=300)
#plt.show()

# -------------------------
# Plot 2: Bailey Diagram (Period vs Amplitude)
# -------------------------
plt.figure(figsize=(8,6))
# All candidates
plt.scatter(df['true_period'], df['true_amplitude'], label='All Candidates', alpha=0.5)
# Best candidates: those with probability >= optimal threshold
best = df[df['cv_prob'] >= optimal_threshold]
plt.scatter(best['true_period'], best['true_amplitude'], label='Best Candidates', alpha=0.7, marker='s')
# Known CVs (if available)
known = df[df['is_known_cv'] == True]
if not known.empty:
    plt.scatter(known['true_period'], known['true_amplitude'], label='Known CVs', edgecolor='k', facecolor='none', s=80)
plt.xlabel('True Period (days)')
plt.ylabel('True Amplitude (mag)')
plt.title('Bailey Diagram: Period vs Amplitude')
plt.legend()
plt.grid(True)
plt.savefig("../PRIMVS/cv_results/post_bailey_diagram.png", dpi=300)
#plt.show()


# -------------------------
# Plot 3a: Spatial Plot in Galactic Coordinates with TESS Overlay
# -------------------------
plt.figure(figsize=(8,6))
plt.scatter(df['l'], df['b'], label='All Candidates', alpha=0.5)
plt.scatter(best['l'], best['b'], label='Best Candidates', alpha=0.7, marker='s')
if not known.empty:
    plt.scatter(known['l'], known['b'], label='Known CVs', edgecolor='k', facecolor='none', s=80)

ax_gal = plt.gca()
# Overlay TESS footprints using the provided class
tess_overlay = TESSCycle8Overlay()
tess_overlay.add_to_plot(ax_gal, focus_region=None, alpha=0.2)

plt.xlabel('Galactic Longitude (l)')
plt.ylabel('Galactic Latitude (b)')
plt.title('Spatial Distribution (Galactic Coordinates) with TESS Cycle 8 Footprints')
plt.legend()
plt.grid(True)
plt.savefig("../PRIMVS/cv_results/spatial_galactic_with_TESS.png", dpi=300)
#plt.show()

# -------------------------
# Plot 3b: Spatial Plot in Equatorial Coordinates with TESS Overlay
# -------------------------
# If 'ra' and 'dec' don't exist, compute from l and b
if 'ra' not in df.columns or 'dec' not in df.columns:
    coords = SkyCoord(l=df['l'].values*u.degree, b=df['b'].values*u.degree, frame='galactic')
    df['ra'] = coords.icrs.ra.deg
    df['dec'] = coords.icrs.dec.deg

plt.figure(figsize=(8,6))
plt.scatter(df['ra'], df['dec'], label='All Candidates', alpha=0.5)
plt.scatter(best['ra'], best['dec'], label='Best Candidates', alpha=0.7, marker='s')
if not known.empty:
    plt.scatter(known['ra'], known['dec'], label='Known CVs', edgecolor='k', facecolor='none', s=80)

ax_eq = plt.gca()
# Overlay TESS footprints in equatorial coordinates
ax_eq = add_tess_overlay_equatorial(ax_eq, alpha=0.2)

plt.xlabel('Right Ascension (deg)')
plt.ylabel('Declination (deg)')
plt.title('Spatial Distribution (Equatorial Coordinates) with TESS Cycle 8 Footprints')
plt.legend()
plt.grid(True)
plt.savefig("../PRIMVS/cv_results/spatial_equatorial_with_TESS.png", dpi=300)
#plt.show()
