import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from mpl_toolkits.mplot3d import Axes3D
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.patheffects as path_effects
from matplotlib.patches import Polygon
from scipy.stats import gaussian_kde
import umap

# TESS Cycle 8 Camera Footprint Class
class TESSCycle8Overlay:
    """TESS Cycle 8 camera footprint visualization for CV candidates"""
    
    def __init__(self):
        # TESS Year 8 camera positions (RA, Dec, Roll) in degrees
        self.camera_positions = {
            # Format: Sector: [(Camera1), (Camera2), (Camera3), (Camera4)]
            # Each camera: (RA, Dec, Roll)
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
            ]
        }
        
        # Convert to galactic coordinates
        self.galactic_positions = self._convert_to_galactic()
    
    def _convert_to_galactic(self):
        """Convert equatorial to galactic coordinates"""
        galactic_positions = {}
        
        for sector, cameras in self.camera_positions.items():
            galactic_positions[sector] = []
            
            for i, (ra, dec, roll) in enumerate(cameras):
                try:
                    # Convert coordinates
                    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
                    gal = coords.galactic
                    
                    # Convert l to -180 to 180 range for better visualization
                    l_deg = gal.l.degree
                    if l_deg > 180:
                        l_deg -= 360
                    
                    galactic_positions[sector].append((l_deg, gal.b.degree, roll))
                except:
                    # Use placeholder if conversion fails
                    galactic_positions[sector].append((0, 0, 0))
        
        return galactic_positions
    
    def add_to_plot(self, ax, focus_region=None, alpha=0.2):
        """Add TESS Cycle 8 camera footprints to plot"""
        # Define colors for cameras
        camera_colors = ['red', 'purple', 'blue', 'green']
        
        # Select sectors based on region focus
        if focus_region == 'bulge':
            # Bulge sectors (mainly 103-105) with cameras 1-2
            sectors = [104, 105]
            camera_indices = [0, 1]  # Cameras 1 and 2
        elif focus_region == 'disk':
            # Disk sectors (mainly 102-103) with cameras 2-3
            sectors = [103] 
            camera_indices = [1, 2]  # Cameras 2 and 3
        else:
            # All sectors and cameras
            sectors = list(self.galactic_positions.keys())
            camera_indices = [0, 1, 2, 3]  # All cameras
        
        # Add camera footprints
        for sector in sectors:
            for i in camera_indices:
                if i >= len(self.galactic_positions[sector]):
                    continue
                    
                l, b, roll = self.galactic_positions[sector][i]
                
                # Create camera footprint (24°×24° square with rotation)
                self._add_camera_footprint(
                    ax, l, b, roll, 
                    color=camera_colors[i],
                    alpha=alpha,
                    label=f"Camera {i+1}" if f"Camera {i+1}" not in [p.get_label() for p in ax.get_children() if hasattr(p, 'get_label')] else "",
                    sector=sector
                )
        
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
    
    def _add_camera_footprint(self, ax, l, b, roll, color='red', alpha=0.2, label="", sector=None, size=12):
        """Add a single camera footprint"""
        # Create square vertices (before rotation)
        vertices_l = np.array([-size, size, size, -size, -size])
        vertices_b = np.array([-size, -size, size, size, -size])
        
        # Apply rotation for camera orientation
        roll_rad = np.radians(roll-90)
        rotated_l = vertices_l * np.cos(roll_rad) - vertices_b * np.sin(roll_rad)
        rotated_b = vertices_l * np.sin(roll_rad) + vertices_b * np.cos(roll_rad)
        
        # Translate to camera center
        vertices_l = l + rotated_l
        vertices_b = b + rotated_b
        
        # Add polygon to plot
        polygon = Polygon(
            np.column_stack([vertices_l, vertices_b]),
            alpha=alpha,
            color=color,
            closed=True,
            label=label
        )
        ax.add_patch(polygon)
        
        # Add sector label
        if sector:
            text = ax.text(l, b, str(sector), fontsize=8, ha='center', va='center',
                     color='white', fontweight='bold')
            text.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()
            ])


def create_all_visualizations(cv_candidates, known_cvs=None, output_dir='./plots'):
    """
    Create comprehensive visualizations for CV candidate analysis
    
    Parameters:
    -----------
    cv_candidates : pandas.DataFrame
        DataFrame containing CV candidates
    known_cvs : pandas.DataFrame, optional
        DataFrame containing known CVs for comparison
    output_dir : str
        Directory to save output visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    # Ensure we have a 'confidence' column - use cv_prob, probability, or create one if needed
    if 'confidence' not in cv_candidates.columns:
        if 'cv_prob' in cv_candidates.columns:
            cv_candidates['confidence'] = cv_candidates['cv_prob']
        elif 'probability' in cv_candidates.columns:
            cv_candidates['confidence'] = cv_candidates['probability']
        else:
            # Default to 0.5 if no confidence measure available
            cv_candidates['confidence'] = 0.5
    
    # Define high-confidence candidates
    high_conf_threshold = 0.8
    high_conf_candidates = cv_candidates[cv_candidates['confidence'] >= high_conf_threshold]
    
    print(f"Visualizing {len(cv_candidates)} CV candidates")
    print(f"Including {len(high_conf_candidates)} high-confidence candidates")
    if known_cvs is not None:
        print(f"Comparing with {len(known_cvs)} known CVs")
    
    # Check for necessary columns
    required_cols = ['true_period', 'true_amplitude', 'l', 'b']
    missing_cols = [col for col in required_cols if col not in cv_candidates.columns]
    if missing_cols:
        print(f"Warning: Missing required columns: {', '.join(missing_cols)}")
    
    # 1. Bailey Diagram
    create_bailey_diagram(cv_candidates, high_conf_candidates, known_cvs, output_dir)
    
    # 2. Spatial Distribution Plots
    create_spatial_plots(cv_candidates, high_conf_candidates, known_cvs, output_dir)
    
    # 3. Dimension Reduction Visualizations (UMAP & PCA)
    create_dimension_reduction_plots(cv_candidates, high_conf_candidates, known_cvs, output_dir)
    
    # 4. CV Probability Distribution
    create_probability_distribution(cv_candidates, output_dir)
    
    # 5. ROC Curve (if known CVs available)
    if known_cvs is not None:
        create_roc_curve(cv_candidates, known_cvs, output_dir)
    
    # 6. Model Agreement Heatmap (if two-stage model was used)
    if all(col in cv_candidates.columns for col in ['cv_prob_trad', 'cv_prob_emb']):
        create_model_agreement_heatmap(cv_candidates, output_dir)


def create_bailey_diagram(cv_candidates, high_conf_candidates, known_cvs, output_dir):
    """Create Bailey diagram (Period-Amplitude)"""
    if not all(col in cv_candidates.columns for col in ['true_period', 'true_amplitude']):
        print("Cannot create Bailey diagram: missing period or amplitude data")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Convert periods to hours and take log
    period_hours = cv_candidates['true_period'] * 24  # Convert to hours
    log_period = np.log10(period_hours)
    
    # Plot all candidates
    plt.scatter(
        log_period, 
        cv_candidates['true_amplitude'],
        alpha=0.3,
        s=10,
        color='blue',
        label=f'All Candidates ({len(cv_candidates)})'
    )
    
    # Plot high confidence candidates
    if len(high_conf_candidates) > 0:
        high_conf_embeddings = high_conf_candidates[embedding_features].values
        high_conf_pca = pca.transform(high_conf_embeddings)
        
        ax.scatter(
            high_conf_pca[:, 0],
            high_conf_pca[:, 1],
            high_conf_pca[:, 2],
            alpha=0.8,
            s=30,
            color='red',
            edgecolors='black',
            linewidth=0.5,
            label=f'High Confidence ({len(high_conf_candidates)})'
        )
    
    # Plot known CVs if available
    if known_cvs is not None and all(feat in known_cvs.columns for feat in embedding_features):
        known_embeddings = known_cvs[embedding_features].values
        known_pca = pca.transform(known_embeddings)
        
        ax.scatter(
            known_pca[:, 0],
            known_pca[:, 1],
            known_pca[:, 2],
            alpha=1.0,
            s=80,
            color='green',
            marker='*',
            edgecolors='black',
            linewidth=0.5,
            label=f'Known CVs ({len(known_cvs)})'
        )
    
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
    ax.set_title('PCA 3D Projection of Contrastive Curves Embedding Space')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'pca_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. UMAP 2D
    try:
        # Apply UMAP for non-linear dimensionality reduction
        print("Applying UMAP transformation...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        
        embeddings_umap = reducer.fit_transform(embeddings)
        
        # 2D UMAP plot
        plt.figure(figsize=(12, 10))
        
        # Plot all candidates
        sc = plt.scatter(
            embeddings_umap[:, 0],
            embeddings_umap[:, 1],
            c=colors,
            cmap=cmap,
            alpha=0.5,
            s=10
        )
        
        # Add colorbar if applicable
        if cmap is not None:
            plt.colorbar(sc, label=cbar_label)
        
        # Plot high confidence candidates
        if len(high_conf_candidates) > 0:
            high_conf_embeddings = high_conf_candidates[embedding_features].values
            high_conf_umap = reducer.transform(high_conf_embeddings)
            
            plt.scatter(
                high_conf_umap[:, 0],
                high_conf_umap[:, 1],
                alpha=0.8,
                s=30,
                color='red',
                edgecolors='black',
                linewidth=0.5,
                label=f'High Confidence ({len(high_conf_candidates)})'
            )
        
        # Plot known CVs if available
        if known_cvs is not None and all(feat in known_cvs.columns for feat in embedding_features):
            known_embeddings = known_cvs[embedding_features].values
            known_umap = reducer.transform(known_embeddings)
            
            plt.scatter(
                known_umap[:, 0],
                known_umap[:, 1],
                alpha=1.0,
                s=80,
                color='green',
                marker='*',
                edgecolors='black',
                linewidth=0.5,
                label=f'Known CVs ({len(known_cvs)})'
            )
        
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.title('UMAP 2D Projection of Contrastive Curves Embedding Space')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'umap_2d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. UMAP 3D if available
        try:
            # Apply UMAP 3D
            reducer_3d = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                n_components=3,
                metric='euclidean',
                random_state=42
            )
            
            embeddings_umap_3d = reducer_3d.fit_transform(embeddings)
            
            # 3D UMAP plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot all candidates
            sc = ax.scatter(
                embeddings_umap_3d[:, 0],
                embeddings_umap_3d[:, 1],
                embeddings_umap_3d[:, 2],
                c=colors,
                cmap=cmap,
                alpha=0.5,
                s=10
            )
            
            # Add colorbar if applicable
            if cmap is not None:
                plt.colorbar(sc, ax=ax, pad=0.1, label=cbar_label)
            
            # Plot high confidence candidates
            if len(high_conf_candidates) > 0:
                high_conf_embeddings = high_conf_candidates[embedding_features].values
                high_conf_umap_3d = reducer_3d.transform(high_conf_embeddings)
                
                ax.scatter(
                    high_conf_umap_3d[:, 0],
                    high_conf_umap_3d[:, 1],
                    high_conf_umap_3d[:, 2],
                    alpha=0.8,
                    s=30,
                    color='red',
                    edgecolors='black',
                    linewidth=0.5,
                    label=f'High Confidence ({len(high_conf_candidates)})'
                )
            
            # Plot known CVs if available
            if known_cvs is not None and all(feat in known_cvs.columns for feat in embedding_features):
                known_embeddings = known_cvs[embedding_features].values
                known_umap_3d = reducer_3d.transform(known_embeddings)
                
                ax.scatter(
                    known_umap_3d[:, 0],
                    known_umap_3d[:, 1],
                    known_umap_3d[:, 2],
                    alpha=1.0,
                    s=80,
                    color='green',
                    marker='*',
                    edgecolors='black',
                    linewidth=0.5,
                    label=f'Known CVs ({len(known_cvs)})'
                )
            
            ax.set_xlabel('UMAP Dimension 1')
            ax.set_ylabel('UMAP Dimension 2')
            ax.set_zlabel('UMAP Dimension 3')
            ax.set_title('UMAP 3D Projection of Contrastive Curves Embedding Space')
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, 'umap_3d.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        except Exception as e:
            print(f"UMAP 3D visualization failed: {e}")
    
    except Exception as e:
        print(f"UMAP visualization failed: {e}")
        print("Skipping UMAP visualizations - consider installing umap-learn")


def create_probability_distribution(cv_candidates, output_dir):
    """Create histogram of CV probabilities"""
    # Check for probability column
    prob_cols = ['cv_prob', 'confidence', 'probability']
    prob_col = None
    
    for col in prob_cols:
        if col in cv_candidates.columns:
            prob_col = col
            break
    
    if prob_col is None:
        print("No probability column found - skipping probability distribution plot")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(cv_candidates[prob_col], bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Add vertical lines for thresholds
    plt.axvline(0.5, color='orange', linestyle='--', linewidth=2, label='Threshold (0.5)')
    plt.axvline(0.8, color='red', linestyle='--', linewidth=2, label='High Confidence (0.8)')
    
    # Calculate statistics
    thresh_count = (cv_candidates[prob_col] >= 0.5).sum()
    high_conf_count = (cv_candidates[prob_col] >= 0.8).sum()
    
    # Add text box with statistics
    stats_text = f"Total candidates: {len(cv_candidates)}\n"
    stats_text += f"P ≥ 0.5: {thresh_count} ({100*thresh_count/len(cv_candidates):.1f}%)\n"
    stats_text += f"P ≥ 0.8: {high_conf_count} ({100*high_conf_count/len(cv_candidates):.1f}%)"
    
    plt.text(
        0.98, 0.95, stats_text,
        transform=plt.gca().transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.xlabel(f'CV {prob_col.replace("_", " ").title()}')
    plt.ylabel('Number of Candidates')
    plt.title('Distribution of CV Probabilities')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_roc_curve(cv_candidates, known_cvs, output_dir):
    """Create ROC curve using known CVs as ground truth"""
    # Check if we have probability column
    prob_cols = ['cv_prob', 'confidence', 'probability']
    prob_col = None
    
    for col in prob_cols:
        if col in cv_candidates.columns:
            prob_col = col
            break
    
    if prob_col is None:
        print("No probability column found - skipping ROC curve")
        return
    
    # First make sure we have a common ID column
    id_cols = ['sourceid', 'source_id', 'primvs_id', 'id']
    id_col = None
    
    for col in id_cols:
        if col in cv_candidates.columns and col in known_cvs.columns:
            id_col = col
            break
    
    if id_col is None:
        print("No common ID column found between candidates and known CVs - skipping ROC curve")
        return
    
    # Create binary labels based on whether candidate is in known CVs list
    known_ids = set(known_cvs[id_col].astype(str))
    cv_candidates['is_known_cv'] = cv_candidates[id_col].astype(str).isin(known_ids)
    
    y_true = cv_candidates['is_known_cv'].astype(int)
    y_score = cv_candidates[prob_col]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    plt.plot(
        fpr, tpr, 
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.3f})'
    )
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    
    # Mark thresholds
    # Find index of threshold closest to 0.5
    thresh_idx_50 = np.argmin(np.abs(thresholds - 0.5))
    # Find index of threshold closest to 0.8
    thresh_idx_80 = np.argmin(np.abs(thresholds - 0.8))
    
    plt.scatter(
        fpr[thresh_idx_50], tpr[thresh_idx_50],
        marker='o',
        color='blue',
        s=100,
        label=f'Threshold = 0.5 (TPR={tpr[thresh_idx_50]:.2f}, FPR={fpr[thresh_idx_50]:.2f})'
    )
    
    plt.scatter(
        fpr[thresh_idx_80], tpr[thresh_idx_80],
        marker='s',
        color='red',
        s=100,
        label=f'Threshold = 0.8 (TPR={tpr[thresh_idx_80]:.2f}, FPR={fpr[thresh_idx_80]:.2f})'
    )
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for CV Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Add statistics in text box
    stats_text = f"Total candidates: {len(cv_candidates)}\n"
    stats_text += f"Known CVs: {y_true.sum()}\n"
    stats_text += f"AUC: {roc_auc:.3f}"
    
    plt.text(
        0.05, 0.95, stats_text,
        transform=plt.gca().transAxes,
        horizontalalignment='left',
        verticalalignment='top',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_model_agreement_heatmap(cv_candidates, output_dir):
    """Create heatmap showing agreement between traditional and embedding models"""
    if not all(col in cv_candidates.columns for col in ['cv_prob_trad', 'cv_prob_emb']):
        print("Missing two-stage model probabilities - skipping model agreement heatmap")
        return
    
    plt.figure(figsize=(12, 10))
    
    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        cv_candidates['cv_prob_trad'],
        cv_candidates['cv_prob_emb'],
        bins=50,
        range=[[0, 1], [0, 1]]
    )
    
    # Apply log scaling for better visualization
    heatmap = np.log1p(heatmap)  # log(1+x) to handle zeros
    
    # Create plot with histogram2d results
    plt.imshow(
        heatmap.T,
        origin='lower',
        aspect='auto',
        extent=[0, 1, 0, 1],
        cmap='viridis'
    )
    
    plt.colorbar(label='log(Count + 1)')
    
    # Add diagonal line for perfect agreement
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Agreement')
    
    # Add decision boundaries
    plt.axvline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Annotate quadrants
    plt.text(0.25, 0.75, "Traditional: No\nEmbedding: Yes", 
             ha='center', va='center', color='white', fontweight='bold')
    plt.text(0.75, 0.75, "Both: Yes", 
             ha='center', va='center', color='white', fontweight='bold')
    plt.text(0.25, 0.25, "Both: No", 
             ha='center', va='center', color='white', fontweight='bold')
    plt.text(0.75, 0.25, "Traditional: Yes\nEmbedding: No", 
             ha='center', va='center', color='white', fontweight='bold')
    
    # Add statistics
    agree_both_yes = ((cv_candidates['cv_prob_trad'] >= 0.5) & (cv_candidates['cv_prob_emb'] >= 0.5)).sum()
    agree_both_no = ((cv_candidates['cv_prob_trad'] < 0.5) & (cv_candidates['cv_prob_emb'] < 0.5)).sum()
    disagree_trad_yes = ((cv_candidates['cv_prob_trad'] >= 0.5) & (cv_candidates['cv_prob_emb'] < 0.5)).sum()
    disagree_emb_yes = ((cv_candidates['cv_prob_trad'] < 0.5) & (cv_candidates['cv_prob_emb'] >= 0.5)).sum()
    
    total = len(cv_candidates)
    agreement_pct = 100 * (agree_both_yes + agree_both_no) / total
    
    stats_text = f"Agreement: {agree_both_yes + agree_both_no} ({agreement_pct:.1f}%)\n"
    stats_text += f"Both Yes: {agree_both_yes} ({100*agree_both_yes/total:.1f}%)\n"
    stats_text += f"Both No: {agree_both_no} ({100*agree_both_no/total:.1f}%)\n"
    stats_text += f"Only Trad Yes: {disagree_trad_yes} ({100*disagree_trad_yes/total:.1f}%)\n"
    stats_text += f"Only Emb Yes: {disagree_emb_yes} ({100*disagree_emb_yes/total:.1f}%)"
    
    plt.text(
        0.98, 0.02, stats_text,
        transform=plt.gca().transAxes,
        horizontalalignment='right',
        verticalalignment='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )
    
    plt.xlabel('Traditional Features Probability')
    plt.ylabel('Embedding Features Probability')
    plt.title('Model Agreement Heatmap: Traditional vs. Embedding Features')
    plt.legend(loc='upper left')
    
    plt.savefig(os.path.join(output_dir, 'model_agreement_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additionally, create scatterplot showing model agreement vs ensemble score
    plt.figure(figsize=(10, 8))
    
    # Calculate agreement as 1 - absolute difference
    cv_candidates['model_agreement'] = 1.0 - np.abs(cv_candidates['cv_prob_trad'] - cv_candidates['cv_prob_emb'])
    
    # Calculate ensemble score as the average if not already present
    if 'cv_prob' not in cv_candidates.columns:
        cv_candidates['cv_prob'] = (cv_candidates['cv_prob_trad'] + cv_candidates['cv_prob_emb']) / 2
    
    # Create scatter plot
    sc = plt.scatter(
        cv_candidates['model_agreement'],
        cv_candidates['cv_prob'],
        c=cv_candidates['cv_prob_trad'],  # Color by traditional model
        cmap='coolwarm',
        alpha=0.7,
        s=20
    )
    
    plt.colorbar(sc, label='Traditional Model Probability')
    
    plt.xlabel('Model Agreement (1 - |Trad - Emb|)')
    plt.ylabel('Ensemble Probability')
    plt.title('Model Agreement vs. Ensemble Probability')
    plt.grid(True, alpha=0.3)
    
    # Add horizontal threshold line
    plt.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='High Confidence Threshold (0.8)')
    plt.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Decision Threshold (0.5)')
    
    # Add vertical threshold line for high agreement
    plt.axvline(0.8, color='green', linestyle='--', alpha=0.7, label='High Agreement (0.8)')
    
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'model_agreement_vs_probability.png'), dpi=300, bbox_inches='tight')
    plt.close()


    high_conf_period_hours = high_conf_candidates['true_period'] * 24
    high_conf_log_period = np.log10(high_conf_period_hours)
    
    plt.scatter(
        high_conf_log_period, 
        high_conf_candidates['true_amplitude'],
        alpha=0.7,
        s=30,
        color='red',
        edgecolors='black',
        linewidth=0.5,
        label=f'High Confidence ({len(high_conf_candidates)})'
    )

    # Plot known CVs if available
    if known_cvs is not None and all(col in known_cvs.columns for col in ['true_period', 'true_amplitude']):
        known_period_hours = known_cvs['true_period'] * 24
        known_log_period = np.log10(known_period_hours)
        
        plt.scatter(
            known_log_period, 
            known_cvs['true_amplitude'],
            alpha=1.0,
            s=80,
            color='green',
            marker='*',
            edgecolors='black',
            linewidth=0.5,
            label=f'Known CVs ({len(known_cvs)})'
        )
    
    # Add period gap region
    plt.axvspan(np.log10(2), np.log10(3), alpha=0.2, color='gray', label='Period Gap (2-3h)')
    
    # Add density contours
    try:
        # Create a 2D histogram
        H, xedges, yedges = np.histogram2d(
            log_period,
            cv_candidates['true_amplitude'],
            bins=[50, 50],
            range=[[-1, 3], [0, 2]]
        )
        
        # Apply Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        H_smooth = gaussian_filter(H, sigma=1.0)
        
        # Plot contours
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.contour(
            H_smooth.T,
            extent=extent,
            levels=5,
            colors='black',
            alpha=0.5,
            linewidths=0.5
        )
    except:
        pass  # Skip contours if they fail
    
    # Set axis labels and title
    plt.xlabel('log₁₀(Period) [hours]')
    plt.ylabel('Amplitude [mag]')
    plt.title('Bailey Diagram: Period-Amplitude Relation for CV Candidates')
    
    # Add annotations for key regions
    plt.annotate(
        'AM CVn & Ultracompact',
        xy=(-0.5, 0.3),
        xytext=(-0.5, 0.3),
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )
    
    plt.annotate(
        'SU UMa & DNe',
        xy=(0.3, 0.7),
        xytext=(0.3, 0.7),
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )
    
    plt.annotate(
        'Magnetic CVs',
        xy=(1.2, 1.2),
        xytext=(1.2, 1.2),
        ha='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )
    
    # Set reasonable limits
    plt.xlim(-1, 3)  # -1 to 3 corresponds to ~0.1h to ~1000h
    plt.ylim(0, 2)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'bailey_diagram.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_spatial_plots(cv_candidates, high_conf_candidates, known_cvs, output_dir):
    """Create galactic coordinate spatial distribution plots"""
    if not all(col in cv_candidates.columns for col in ['l', 'b']):
        print("Cannot create spatial plots: missing l or b coordinates")
        return
    
    tess_overlay = TESSCycle8Overlay()
    
    # Ensure l is in -180 to 180 range for better visualization
    cv_candidates = cv_candidates.copy()
    cv_candidates['l_centered'] = np.where(
        cv_candidates['l'] > 180,
        cv_candidates['l'] - 360,
        cv_candidates['l']
    )
    
    high_conf_candidates = high_conf_candidates.copy()
    high_conf_candidates['l_centered'] = np.where(
        high_conf_candidates['l'] > 180,
        high_conf_candidates['l'] - 360,
        high_conf_candidates['l']
    )
    
    if known_cvs is not None:
        known_cvs = known_cvs.copy()
        if all(col in known_cvs.columns for col in ['l', 'b']):
            known_cvs['l_centered'] = np.where(
                known_cvs['l'] > 180,
                known_cvs['l'] - 360,
                known_cvs['l']
            )
    
    # 1. Full survey area with TESS footprints
    plt.figure(figsize=(14, 10))
    
    # Plot candidates
    plt.scatter(
        cv_candidates['l_centered'],
        cv_candidates['b'],
        alpha=0.3,
        s=10,
        color='blue',
        label=f'All Candidates ({len(cv_candidates)})'
    )
    
    # Plot high confidence candidates
    if len(high_conf_candidates) > 0:
        plt.scatter(
            high_conf_candidates['l_centered'],
            high_conf_candidates['b'],
            alpha=0.7,
            s=30,
            color='red',
            edgecolors='black',
            linewidth=0.5,
            label=f'High Confidence ({len(high_conf_candidates)})'
        )
    
    # Plot known CVs if available
    if known_cvs is not None and all(col in known_cvs.columns for col in ['l', 'b']):
        plt.scatter(
            known_cvs['l_centered'],
            known_cvs['b'],
            alpha=1.0,
            s=80,
            color='green',
            marker='*',
            edgecolors='black',
            linewidth=0.5,
            label=f'Known CVs ({len(known_cvs)})'
        )
    
    # Add VVV survey boundaries
    # Bulge region: -10° < l < +10° and -10° < b < +5°
    bulge_x = np.array([-10, -10, 10, 10, -10])
    bulge_y = np.array([-10, 5, 5, -10, -10])
    plt.plot(bulge_x, bulge_y, 'r-', linewidth=2, label='VVV Bulge')
    
    # Disk region: -65° < l < -10° and -2° < b < -10°
    disk_x = np.array([-65, -65, -10, -10, -65])
    disk_y = np.array([-2, -10, -10, -2, -2])
    plt.plot(disk_x, disk_y, 'r-', linewidth=2, label='VVV Disk')
    
    # Add TESS Cycle 8 camera footprints
    tess_overlay.add_to_plot(plt.gca())
    
    plt.xlabel('Galactic Longitude (l) [deg]')
    plt.ylabel('Galactic Latitude (b) [deg]')
    plt.title('Spatial Distribution of CV Candidates with TESS Cycle 8 Coverage')
    plt.grid(True, alpha=0.3)
    plt.xlim(-70, 20)
    plt.ylim(-15, 10)
    
    plt.savefig(os.path.join(output_dir, 'spatial_distribution_full.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Bulge region with TESS footprints
    plt.figure(figsize=(10, 8))
    
    # Plot candidates
    plt.scatter(
        cv_candidates['l_centered'],
        cv_candidates['b'],
        alpha=0.3,
        s=10,
        color='blue',
        label=f'All Candidates ({len(cv_candidates)})'
    )
    
    # Plot high confidence candidates
    if len(high_conf_candidates) > 0:
        plt.scatter(
            high_conf_candidates['l_centered'],
            high_conf_candidates['b'],
            alpha=0.7,
            s=30,
            color='red',
            edgecolors='black',
            linewidth=0.5,
            label=f'High Confidence ({len(high_conf_candidates)})'
        )
    
    # Plot known CVs if available
    if known_cvs is not None and all(col in known_cvs.columns for col in ['l', 'b']):
        plt.scatter(
            known_cvs['l_centered'],
            known_cvs['b'],
            alpha=1.0,
            s=80,
            color='green',
            marker='*',
            edgecolors='black',
            linewidth=0.5,
            label=f'Known CVs ({len(known_cvs)})'
        )
    
    # Add VVV bulge boundary
    plt.plot(bulge_x, bulge_y, 'r-', linewidth=2, label='VVV Bulge')
    
    # Add TESS Cycle 8 camera footprints (focused on bulge)
    tess_overlay.add_to_plot(plt.gca(), focus_region='bulge')
    
    plt.xlabel('Galactic Longitude (l) [deg]')
    plt.ylabel('Galactic Latitude (b) [deg]')
    plt.title('CV Candidates in Galactic Bulge with TESS Cycle 8 Coverage')
    plt.grid(True, alpha=0.3)
    plt.xlim(-12, 12)
    plt.ylim(-10, 5)
    
    plt.savefig(os.path.join(output_dir, 'spatial_distribution_bulge.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Disk region with TESS footprints
    plt.figure(figsize=(12, 6))
    
    # Plot candidates
    plt.scatter(
        cv_candidates['l_centered'],
        cv_candidates['b'],
        alpha=0.3,
        s=10,
        color='blue',
        label=f'All Candidates ({len(cv_candidates)})'
    )
    
    # Plot high confidence candidates
    if len(high_conf_candidates) > 0:
        plt.scatter(
            high_conf_candidates['l_centered'],
            high_conf_candidates['b'],
            alpha=0.7,
            s=30,
            color='red',
            edgecolors='black',
            linewidth=0.5,
            label=f'High Confidence ({len(high_conf_candidates)})'
        )
    
    # Plot known CVs if available
    if known_cvs is not None and all(col in known_cvs.columns for col in ['l', 'b']):
        plt.scatter(
            known_cvs['l_centered'],
            known_cvs['b'],
            alpha=1.0,
            s=80,
            color='green',
            marker='*',
            edgecolors='black',
            linewidth=0.5,
            label=f'Known CVs ({len(known_cvs)})'
        )
    
    # Add VVV disk boundary
    plt.plot(disk_x, disk_y, 'r-', linewidth=2, label='VVV Disk')
    
    # Add TESS Cycle 8 camera footprints (focused on disk)
    tess_overlay.add_to_plot(plt.gca(), focus_region='disk')
    
    plt.xlabel('Galactic Longitude (l) [deg]')
    plt.ylabel('Galactic Latitude (b) [deg]')
    plt.title('CV Candidates in Galactic Disk with TESS Cycle 8 Coverage')
    plt.grid(True, alpha=0.3)
    plt.xlim(-65, -10)
    plt.ylim(-10, -1)
    
    plt.savefig(os.path.join(output_dir, 'spatial_distribution_disk.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Density heatmap with TESS footprints
    plt.figure(figsize=(14, 10))
    
    # Create 2D histogram 
    l_bins = np.linspace(-70, 20, 100)
    b_bins = np.linspace(-15, 10, 50)
    
    H, xedges, yedges = np.histogram2d(
        cv_candidates['l_centered'], 
        cv_candidates['b'],
        bins=[l_bins, b_bins]
    )
    
    # Apply Gaussian smoothing
    from scipy.ndimage import gaussian_filter
    H_smooth = gaussian_filter(H, sigma=1.0)
    
    # Plot heatmap
    plt.pcolormesh(
        xedges, yedges, H_smooth.T,
        norm=LogNorm(vmin=0.1, vmax=H_smooth.max()),
        cmap='inferno',
        alpha=0.7
    )
    
    plt.colorbar(label='Candidate Density (log scale)')
    
    # Add VVV survey boundary lines
    plt.plot(bulge_x, bulge_y, 'w--', alpha=0.8, linewidth=1.5)
    plt.plot(disk_x, disk_y, 'w--', alpha=0.8, linewidth=1.5)
    
    # Add TESS Cycle 8 camera footprints
    tess_overlay.add_to_plot(plt.gca(), alpha=0.3)
    
    # Plot known CVs if available
    if known_cvs is not None and all(col in known_cvs.columns for col in ['l', 'b']):
        plt.scatter(
            known_cvs['l_centered'],
            known_cvs['b'],
            alpha=1.0,
            s=80,
            color='lime',
            marker='*',
            edgecolors='black',
            label=f'Known CVs ({len(known_cvs)})'
        )
        plt.legend(loc='upper right')
    
    plt.xlabel('Galactic Longitude (l) [deg]')
    plt.ylabel('Galactic Latitude (b) [deg]')
    plt.title('Density Distribution of CV Candidates with TESS Cycle 8 Coverage')
    plt.grid(True, alpha=0.3)
    plt.xlim(-70, 20)
    plt.ylim(-15, 10)
    
    plt.savefig(os.path.join(output_dir, 'spatial_distribution_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_dimension_reduction_plots(cv_candidates, high_conf_candidates, known_cvs, output_dir):
    """Create UMAP and PCA visualizations in 2D and 3D"""
    # Check for embedding features
    embedding_features = []
    for i in range(64):
        if str(i) in cv_candidates.columns:
            embedding_features.append(str(i))
    
    # Check if we have sufficient embedding features
    if len(embedding_features) < 3:
        print("Insufficient embedding features for dimension reduction - skipping")
        return
    
    # Extract embeddings
    embeddings = cv_candidates[embedding_features].values
    
    # Create colormap based on confidence if available
    if 'confidence' in cv_candidates.columns:
        colors = cv_candidates['confidence']
        cmap = 'viridis'
        cbar_label = 'Confidence'
    else:
        colors = 'blue'
        cmap = None
        cbar_label = None
    
    # 1. PCA 2D
    from sklearn.decomposition import PCA
    
    # Apply PCA
    pca = PCA(n_components=3)
    embeddings_pca = pca.fit_transform(embeddings)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA explained variance: {explained_variance[0]:.2%}, {explained_variance[1]:.2%}, {explained_variance[2]:.2%}")
    
    # 2D PCA plot
    plt.figure(figsize=(12, 10))
    
    # Plot all candidates
    sc = plt.scatter(
        embeddings_pca[:, 0],
        embeddings_pca[:, 1],
        c=colors,
        cmap=cmap,
        alpha=0.5,
        s=10
    )
    
    # Add colorbar if applicable
    if cmap is not None:
        plt.colorbar(sc, label=cbar_label)
    
    # Plot high confidence candidates
    if len(high_conf_candidates) > 0:
        high_conf_embeddings = high_conf_candidates[embedding_features].values
        high_conf_pca = pca.transform(high_conf_embeddings)
        
        plt.scatter(
            high_conf_pca[:, 0],
            high_conf_pca[:, 1],
            alpha=0.8,
            s=30,
            color='red',
            edgecolors='black',
            linewidth=0.5,
            label=f'High Confidence ({len(high_conf_candidates)})'
        )
    
    # Plot known CVs if available
    if known_cvs is not None and all(feat in known_cvs.columns for feat in embedding_features):
        known_embeddings = known_cvs[embedding_features].values
        known_pca = pca.transform(known_embeddings)
        
        plt.scatter(
            known_pca[:, 0],
            known_pca[:, 1],
            alpha=1.0,
            s=80,
            color='green',
            marker='*',
            edgecolors='black',
            linewidth=0.5,
            label=f'Known CVs ({len(known_cvs)})'
        )
    
    plt.xlabel(f'PC1 ({explained_variance[0]:.2%})')
    plt.ylabel(f'PC2 ({explained_variance[1]:.2%})')
    plt.title('PCA 2D Projection of Contrastive Curves Embedding Space')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, 'pca_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PCA 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all candidates
    sc = ax.scatter(
        embeddings_pca[:, 0],
        embeddings_pca[:, 1],
        embeddings_pca[:, 2],
        c=colors,
        cmap=cmap,
        alpha=0.5,
        s=10
    )
    
    # Add colorbar if applicable
    if cmap is not None:
        plt.colorbar(sc, ax=ax, pad=0.1, label=cbar_label)
    
    # Plot high confidence candidates
    if len(high_