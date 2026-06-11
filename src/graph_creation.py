import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Initialize a publication-ready multi-panel canvas (2 rows, 2 columns)
fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# Load your completed project assets
# (Paths assume standard repository outputs based on your script completions)
# Load your completed project assets from their correct directories
img_a = mpimg.imread('results/exp04/shap_beeswarm.png')          # Fixed folder path (exp04)
img_b = mpimg.imread('results/exp05/umap_clusters.png')          # Correct
img_c = mpimg.imread('results/exp05/cluster_radar_charts.png')   # Correct
img_d = mpimg.imread('results/exp07/concordance_heatmap.png')    # Correct
# Panel A: Feature Importance
axs[0, 0].imshow(img_a)
axs[0, 0].set_title('A: XGBoost SHAP Feature Drivers', fontsize=14, fontweight='bold')
axs[0, 0].axis('off')

# Panel B: Phenotypic Spaces
axs[0, 1].imshow(img_b)
axs[0, 1].set_title('B: UMAP 2D Latent Clustering Space', fontsize=14, fontweight='bold')
axs[0, 1].axis('off')

# Panel C: Clinical Biomarkers
axs[1, 0].imshow(img_c)
axs[1, 0].set_title('C: Sub-Phenotype Characteristic Radar Profiles', fontsize=14, fontweight='bold')
axs[1, 1].axis('off')

# Panel D: AI-Driven Translation Cross-Validation
axs[1, 1].imshow(img_d)
axs[1, 1].set_title('D: LLM-SHAP Knowledge Concordance Matrix', fontsize=14, fontweight='bold')
axs[1, 1].axis('off')

# Clean layout presentation rules
plt.tight_layout()

# Save under strict APASL dimensions (Max 1024x768 pixels custom constraint)
# Adjusting figsize/dpi ensures maximum crispness within their size restriction
plt.savefig('results/exp10/Figure1_Combined_Panels.jpg', format='jpg', dpi=100, bbox_inches='tight')
print("APASL Multi-Panel Mega-Figure compiled successfully!")