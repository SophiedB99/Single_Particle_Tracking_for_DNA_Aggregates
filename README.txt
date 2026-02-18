# Single Particle Tracking Analysis Pipeline

## Overview

This repository contains a comprehensive Python-based analysis pipeline for single particle tracking (SPT) of DNA molecules in cellular environments. The pipeline processes raw microscopy data through particle localization, tracking, drift correction, and advanced statistical analysis of diffusion dynamics.

**Key Features:**
- Automated image preprocessing with bandpass filtering
- Sub-pixel accurate particle tracking using TrackPy
- Per-particle localization error correction based on brightness
- Mean squared displacement (MSD) analysis with anomalous diffusion characterization
- Van Hove correlation functions for non-Gaussian diffusion
- Angular bias analysis for directional motion detection
- Radius of gyration calculations for conformational analysis

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Pipeline Overview](#pipeline-overview)
4. [Detailed Workflow](#detailed-workflow)
5. [Configuration](#configuration)
6. [Output Files](#output-files)
7. [Citation](#citation)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware
- **RAM:** Minimum 16 GB (32 GB recommended for large datasets)
- **Storage:** 50+ GB free space for raw data and analysis outputs
- **Processor:** Multi-core CPU (4+ cores recommended for parallel processing)

### Software
- **Python:** 3.8 or higher
- **Operating System:** Windows, macOS, or Linux

### Required Python Packages
```
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
scipy >= 1.7.0
trackpy >= 0.5.0
pims >= 0.6.0
scikit-image >= 0.18.0
Pillow >= 8.0.0
imageio >= 2.9.0
```

---

## Installation

1. **Clone or download this repository:**
   ```bash
   git clone https://github.com/yourusername/spt-analysis.git
   cd spt-analysis
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv spt_env
   source spt_env/bin/activate  # On Windows: spt_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib scipy trackpy pims scikit-image Pillow imageio
   ```

4. **Verify installation:**
   ```bash
   python -c "import trackpy; print(trackpy.__version__)"
   ```

---

## Pipeline Overview

```
┌────────────────────────────────────────────────┐
│                    RAW MICROSCOPY DATA (.tif)                    │
└────────────────────────┬───────────────────────┘
                                  │
                ┌────────────▼────────────┐
                │  01_image_preprocessing │  ◄── Bandpass filtering
                └────────────┬────────────┘      Background removal
                                  │
                ┌────────────▼────────────┐
                │  02_particle_tracking   │  ◄── TrackPy localization
                └────────────────┬────────┘      Drift correction
                             	       │
          ┌────────────────────┼─────────────────┐
          │                  		   │                  		│
┌─────────▼─────────┐  ┌─────▼──────┐  ┌───────▼────────┐
│ 04a_calibration  			│  │ 04b_calib  	 │  │ 04c_validation 		│
│ _stuck_beads     			│  │ _dim_part  	 │  │                		│
└─────────┬─────────┘  └─────┬──────┘  └───────┬────────┘
              │                        │                      │
              └─────────────────┼────────────────┘
                               	       │
                           ┌────────▼────────┐
                           │ Localization           │  ◄── σ_stuck, σ_dim
                           │ Error Values           │      Mass references
                           └────────┬────────┘
                                       │
              ┌──────────────────┼───────────────┐
              │                         │                     │
┌─────────▼─────────┐  ┌─────▼──────────┐  ┌──▼────────────┐
│ 05_msd_analysis          │  │ 06_van_hove           │  │ 07_angular          │
│ _main                    │  │ _analysis             │  │ _bias              │
└───────────────────┘  └────────────────┘  └───────────────┘
                                       │
                           ┌────────▼────────┐
                           │ 08_radius_of           │
                           │ _gyration              │
                           └─────────────────┘
```

---

## Detailed Workflow

### Phase 1: Image Preprocessing & Particle Tracking

#### **Step 1: Image Preprocessing** (`01_image_preprocessing.py`)

**Purpose:** Remove background noise and enhance particle visibility.

**Key Parameters:**
- `lshort = 1`: Gaussian kernel size (particle size in pixels)
- `llong = 15`: Boxcar averaging window (background size)
- `threshold = 40`: Minimum intensity threshold

**Input:** Raw `.tif` image sequences
**Output:** Processed `.tiff` files in `B_ProcessedData/`

**Usage:**
```python
# Edit configuration section:
filepath = r'path/to/raw/data'
filepath2 = r'path/to/output'
N = 800  # Number of frames to process

# Run:
python 01_image_preprocessing.py
```

**Critical Notes:**
- Automatically detects experiment folders starting with 'Exp'
- Skips already-processed experiments
- Monitor console for processing status

---

#### **Step 2: Particle Tracking** (`02_particle_tracking.py`)

**Purpose:** Locate and link particles across frames.

**Key Parameters:**
- `diameter = 11`: Expected particle size (must be odd integer)
- `minmass = 1000`: Minimum integrated brightness
- `search_range = 2`: Maximum displacement between frames (pixels)
- `memory = 3`: Frames a particle can disappear
- `min_frames = 150`: Minimum trajectory length

**Filtering Criteria:**
```python
mass > 10
size < 3.6
ecc < 0.3  # Eccentricity (circularity)
```

**Input:** Processed images from Step 1
**Output:** 
- `f/` - Particle locations per frame
- `t/` - Raw trajectories (before filtering)
- `tf/` - Filtered trajectories (after quality control)
- `tm/` - Drift-corrected trajectories (ready for analysis)

**Usage:**
```python
# Edit paths:
Filepath = r'path/to/experiment/folder'
bp = '/ExperimentX'
name = 'ExperimentX'

# Run:
python 02_particle_tracking.py
```

**Quality Control Checks:**
1. **Subpixel accuracy:** Check histogram for uniform distribution
2. **KS test results:** p > 0.05 indicates good localization
3. **Drift magnitude:** Should be < 0.5 μm over experiment duration

---

### Phase 2: Localization Error Calibration

**Critical Concept:** Localization errors due to photon noise must be measured and corrected to obtain true MSD values.

#### **Step 4a: Stuck Bead Calibration** (`04a_calibration_stuck_beads.py`)

**Purpose:** Measure σ for bright, immobilized particles (reference).

**Input:** Trajectory data from immobilized beads (e.g., `0.22 μm` beads on glass)

**Method:**
- For truly immobile particles: all position variance = localization error
- σ_x, σ_y calculated from position fluctuations
- MSD plateau method: `σ = sqrt(MSD_plateau / 4)`

**Output:**
- `SIGMA_STUCK_MICRONS` = median σ (typically ~0.013 μm)
- `MASS_STUCK_REFERENCE` = median particle brightness
- Diagnostic plots in `Figures/`

**Usage:**
```python
STUCK_BEADS_PATH = 'path/to/stuck/beads/tm/'
python 04a_calibration_stuck_beads.py
```

**Expected Results:**
- σ_x ≈ σ_y (isotropic localization)
- σ inversely proportional to sqrt(mass) → validates mass scaling

---

#### **Step 4b: Dim Particle Calibration** (`04b_calibration_dim_particles.py`)

**Purpose:** Measure σ for dimmer particles (100-500 bp DNA fragments).

**Method:** Comparative subtraction
- Uses two identical DNA samples with different dyes:
  - **Cy3** (bright reference, low noise)
  - **YOYO-1** (dim target, higher noise)
- Assumes identical true motion
- Noise difference: `4σ²_dim ≈ MSD_YOYO1 - MSD_Cy3`

**Fallback Logic:** If subtraction yields negative values (indicating differential motion), switches to self-MSD upper-bound estimate.

**Output:**
- `SIGMA_DIM_MICRONS` = σ for dim particles (~0.015 μm)
- `MASS_DIM_REFERENCE` = reference mass for scaling

**Usage:**
```python
CY3_BRIGHT_PATH = 'path/to/cy3/data/tm/'
YOYO1_DIM_PATH = 'path/to/yoyo1/data/tm/'
python 04b_calibration_dim_particles.py
```

---

#### **Step 4c: Validation** (`04c_calibration_validation.py`)

**Purpose:** Test the fallback logic and validate calibration consistency.

**Features:**
- Implements both subtraction and self-MSD methods
- Automatic method selection based on data quality
- Diagnostic plots showing method comparison

---

### Phase 3: Main Analysis

#### **Step 5: MSD Analysis** (`05_msd_analysis_main.py`)

**Purpose:** Comprehensive MSD analysis with localization correction.

**Key Features:**

1. **Per-Particle Correction:**
   ```python
   σ_particle = σ_ref × sqrt(mass_ref / mass_particle)
   MSD_corrected = MSD_raw - 4σ²_particle
   ```

2. **Power-Law Fitting:**
   - **Short-time regime** (0.1-0.8s): Initial dynamics
   - **Long-time regime** (2-8s): Steady-state behavior
   - Extracts α (anomalous exponent) and D_app (apparent diffusion coefficient)

3. **Conditional Sigma Assignment:**
   - 100-500 bp fragments: Use `SIGMA_DIM_MICRONS` (fixed, no mass scaling)
   - Larger fragments (4713-9553 bp): Use `SIGMA_STUCK_MICRONS` (with mass scaling)
   - Beads: No correction (σ = 0)

**Configuration:**
```python
# Reference values (UPDATE from calibration steps)
SIGMA_STUCK_MICRONS = 0.013543
MASS_STUCK_REFERENCE = 3381.5
SIGMA_DIM_MICRONS = 0.015280
MASS_DIM_REFERENCE = 589.8

# Analysis options
USE_LOCALIZATION_CORRECTION = True
USE_MASS_SCALING = True
R_SQUARED_THRESHOLD = 0.6  # Minimum fit quality
```

**Outputs:**

**Figure 2 (Combined):**
- Panels A-F: Individual MSD curves per condition
- Panel G: D_app box plots
- Panel H: α box plots with statistical comparisons

**Figure 5:** Raw ensemble MSD curves (with inset zoom)
**Figure 6:** Corrected ensemble MSD curves (with inset zoom)
**Figure 4:** Mass vs. Sigma diagnostic plot

**Statistical Output:**
- One-way ANOVA across conditions
- Pairwise t-tests with significance levels
- α mean ± SEM for each condition

**Usage:**
```python
# Update paths in CONDITIONS_MAP
# Update reference values from calibration
python 05_msd_analysis_main.py
```

---

#### **Step 5b: Cellular MSD Analysis** (`05b_msd_analysis_cellular.py`)

**Purpose:** Specialized analysis for cell-type and ATP-depletion experiments.

**Conditions Analyzed:**
- MDA-MB-231 (breast cancer, highly invasive)
- MCF-7 (breast cancer, less invasive)
- MCF-10A (non-tumorigenic breast epithelial)
- ATP-depleted vs. Control

**Key Difference:** Uses same correction framework but with cellular-specific paths and plotting arrangements.

**Output:**
**Figure 7 (2×2 panel):**
- Panel A: MSD ensemble (cell types + beads)
- Panel B: α box plot (cell types + beads)
- Panel C: MSD ensemble (ATP vs. Control)
- Panel D: α box plot (ATP vs. Control)

---

#### **Step 6: Van Hove Analysis** (`06_van_hove_analysis.py`)

**Purpose:** Detect non-Gaussian diffusion and transient confinement.

**Theory:**
Van Hove correlation function G_s(Δr, Δt) is the probability distribution of single-particle displacements:
- **Gaussian diffusion:** G_s is Gaussian at all Δt
- **Non-Gaussian:** Shows exponential tails, multiple peaks
- **Caging/Confinement:** Sharp peak at small Δr

**Key Features:**

1. **Corrected Displacements:**
   ```python
   Δr²_real = Δr²_measured - 4σ²_particle
   Δr_real = sqrt(Δr²_real)
   ```

2. **Diffusion Peak Quantification:**
   - Ignores noise-dominated region (Δr < 0.01 μm)
   - Identifies peak position (most probable displacement)
   - Calculates peak height (probability density)

3. **Time Lags Analyzed:**
   - 0.1s (early dynamics)
   - 1.0s (intermediate)
   - 5.0s (late time)
   - 8.0s (steady state)

**Configuration:**
```python
VAN_HOVE_LAG_SECONDS = [0.1, 1.0, 5.0, 8.0]
BIN_COUNT = 50
MAX_DISPLACEMENT = 0.3  # μm
MIN_DISPLACEMENT_FOR_DIFFUSION_PEAK = 0.01  # μm (noise filter)
```

**Output:**

**Figure 3 (2×2 panels A-D):**
- Each panel shows G_s(Δr) for one Δt
- Colored curves for each DNA size
- Inset zoom on diffusion peak region
- Console output: Peak positions and heights

**Interpretation:**
- **Sharp, narrow peak:** Confined/caged motion
- **Broad peak, high Δr:** Free/directed diffusion
- **Peak shift with Δt:** Transition from confined → diffusive

**Usage:**
```python
python 06_van_hove_analysis.py
```

---

#### **Step 7: Angular Bias Analysis** (`07_angular_bias_analysis.py`)

**Purpose:** Quantify directionality vs. random walk behavior.

**Method: Next-Step Angle (NSA)**

For each trajectory, calculates the angle between consecutive displacement vectors:
```
v₁ = r(t+Δt) - r(t)
v₂ = r(t+2Δt) - r(t+Δt)
θ = arccos(v₁·v₂ / |v₁||v₂|)
```

**Noise Filtering:**
Only includes steps where displacement > 2σ_particle to exclude noise-dominated motion.

**Motion Classification:**
- **θ < 90°:** Directed/persistent motion (continues in same direction)
- **θ ≈ 90°:** Random walk (no angular correlation)
- **θ > 90°:** Confined/retrograde motion (reversal)

**Statistical Test:**
Chi-square test against uniform distribution (random walk expectation)

**Configuration:**
```python
LAG_TIME_STEPS_SHORT = 10  # 1.0s
LAG_TIME_STEPS_MID = 50    # 5.0s
LAG_TIME_STEPS_LONG = 80   # 8.0s
```

**Output:**

**Figure 4:**
- **Panels A-C:** Angular probability density at 3 time scales
- **Panel D:** Directed/Confined ratio bar chart
  - Significance markers: ****, ***, **, *, ns

**Console Output:**
- N_total, N_directed, N_confined for each condition
- D/C ratio
- P(θ < 90°) with 95% CI
- Chi² p-value

**Interpretation:**
- **D/C > 1:** Net directed motion
- **D/C ≈ 1:** Random walk
- **D/C < 1:** Net confinement

**Usage:**
```python
python 07_angular_bias_analysis.py
```

---

#### **Step 8: Radius of Gyration** (`08_radius_of_gyration.py`)

**Purpose:** Measure spatial extent and conformational dynamics.

**Theory:**
Radius of gyration R_g quantifies the root-mean-square distance of positions from the trajectory's center of mass:
```
R_g = sqrt(<(r - r_com)²>)
```

**Localization Correction:**
```
R²_g,corrected = R²_g,measured - 2σ²_particle
```

**Configuration:**
```python
mpp = 0.16  # microns per pixel (verify for your system)
USE_LOCALIZATION_CORRECTION = True
USE_MASS_SCALING = True
```

**Output:**
- Histogram of R_g distributions per condition
- Violin plots comparing conditions
- Theoretical R_g predictions (linear vs. circular DNA)

**Console Output:**
- Mean, median, std dev of R_g for each condition
- Comparison to theoretical values (Kuhn model)

**Usage:**
```python
python 08_radius_of_gyration.py
```

---

### Phase 4: Validation & Utilities

#### **Step 9: Drift Correction Validation** (`09_drift_correction_validation.py`)

**Purpose:** Assess impact of drift correction on MSD and α.

**Method:**
Compares trajectories from:
- `tf/` directory (before drift correction)
- `tm/` directory (after drift correction)

For each experiment, calculates:
- ΔMSD(τ) = MSD_raw - MSD_corrected
- α_raw vs. α_corrected
- Number of tracks passing quality filters

**Output:**
- Per-experiment comparison plots
- Summary table (CSV + visual)
- Statistics: Mean ΔMSD, α changes

**Usage:**
```python
BASE_DATA_PATH = r"path/to/experiments/E_output_data/"
python 09_drift_correction_validation.py
```

**Interpretation:**
- Large ΔMSD → significant drift present
- Δα > 0.1 → drift affects anomalous exponent classification

---

#### **Step 10: Trajectory Length Sensitivity** (`10_trajectory_length_sensitivity.py`)

**Purpose:** Determine optimal trajectory length for stable α estimation.

**Method:**
For each condition:
1. Bin trajectories by length (0-10s, 10-20s, 20-40s, 40s+)
2. Calculate ensemble MSD for each bin
3. Fit power law to extract α
4. Assess convergence (when Δα < 5% between bins)

**Output:**
- MSD curves colored by trajectory length bin
- Console: Optimal length threshold per condition
- Diagnostic plots showing α stability vs. length

**Usage:**
```python
python 10_trajectory_length_sensitivity.py
```

**Recommendation:** Use lengths > optimal threshold for final analysis.

---

#### **Step 11: Summary Trajectory Plots** (`11_trajectory_summary_plots.py`)

**Purpose:** Generate publication-quality figures for individual experiments.

**Features:**

1. **MSD Plots:**
   - Individual MSDs colored by α
   - Separate plots for subdiffusive (α < 1) and superdiffusive (α > 1)

2. **Trajectory Visualization:**
   - Color-coded by time (plasma colormap)
   - Incremental plots (n=1, 2, 3 trajectories)
   - Scale bars and axis formatting

3. **Statistical Histograms:**
   - D_app distribution (log scale)
   - α distribution
   - 2D histogram: α vs. log(D_app)

4. **Localization Precision:**
   - For stuck beads: σ histogram with mean/median

**Configuration:**
```python
R2_THRESHOLD = 0.3  # Lower for stuck beads, higher (0.9) for diffusion
ANALYZE_STUCK = True  # Set True for localization precision measurement
```

**Usage:**
```python
# Edit paths:
Filepath = 'path/to/experiment'
bp = 'Collected_XXX'
bps = 'Descriptive Label'

python 11_trajectory_summary_plots.py
```

---

## Configuration

### Global Parameters (Set in each analysis script)

**Microscopy Settings:**
```python
MPP = 0.102  # Microns per pixel (critical - verify for your system!)
FPS = 10     # Frames per second
```

**Localization Error References** (from calibration):
```python
SIGMA_STUCK_MICRONS = 0.013543  # Bright particle precision
MASS_STUCK_REFERENCE = 3381.5   # Reference brightness

SIGMA_DIM_MICRONS = 0.015280    # Dim particle precision  
MASS_DIM_REFERENCE = 589.8      # Dim reference brightness
```

**Analysis Options:**
```python
USE_LOCALIZATION_CORRECTION = True  # Enable/disable correction
USE_MASS_SCALING = True             # Scale σ by brightness
R_SQUARED_THRESHOLD = 0.6           # Fit quality cutoff
MAX_LAGTIME = 100                   # Maximum MSD lag (frames)
```

### Directory Structure (Required)

```
Experiment_Root/
├── A_rawtif/                    # Raw microscopy data
│   ├── Exp_1/
│   │   └── *.tif
│   └── Exp_2/
│       └── *.tif
├── B_ProcessedData/             # Bandpass-filtered images
│   ├── Exp_1/
│   └── Exp_2/
└── E_output_data/
    ├── Collected_XXX/
    │   ├── f/                   # Frame-by-frame localizations
    │   ├── t/                   # Raw trajectories
    │   ├── tf/                  # Filtered trajectories
    │   └── tm/                  # Drift-corrected (MAIN INPUT)
    ├── MSDcollected/
    ├── Trajcollected/
    ├── MSDindividual/
    └── Figures/                 # Output plots
```

---

## Output Files

### Trajectory Data Files

**`tm/tm_*.csv`** - Final drift-corrected trajectories
Columns:
- `frame`: Frame number
- `x`, `y`: Position in pixels
- `particle`: Unique trajectory ID
- `mass`: Integrated brightness
- `size`: Radius of gyration (pixels)
- `ecc`: Eccentricity
- `signal`: Peak intensity
- `raw_mass`: Uncorrected mass
- `ep`: Localization precision estimate

### MSD Analysis Output

**`MSDcollected/MSDcollected.csv`**
- Rows: Lag times (s)
- Columns: Individual trajectory MSDs (μm²)

**`MSDcollected/Alpha.csv`**
- Column 0: α (anomalous exponent)
- Column 1: A (prefactor, D_app = A/4)

### Statistical Summary

**Console Output Example:**
```
================================================================
MSD ANALYSIS WITH PER-PARTICLE LOCALIZATION CORRECTION
================================================================
Sigma (from stuck beads): 0.013543 μm
Mass reference: 3381.5
Mass scaling: ENABLED
R² threshold: 0.6

Processing 100 bp... using Dim Sigma (0.015280 μm) & NO MASS SCALING.
  Tracks accepted: 1523
  Mean mass: 612.3
  Mean sigma: 0.015280 μm

================================================================
STATISTICAL ANALYSIS - ALPHA VALUES (2-8s)
================================================================

One-way ANOVA:
  F-statistic: 45.2341
  P-value: 1.234e-28
  Significance: **** (P < 0.0001)

Pairwise two-sided t-tests:
  100 bp vs 250 bp: P = 0.0023 (**)
  100 bp vs 500 bp: P = 0.0001 (****)
  ...
```

### Figures

**All figures saved as 600 dpi .jpg or .png:**

- `Figure_2.jpg`: Main MSD analysis (8-panel combined)
- `Figure_3.jpg`: Van Hove distributions (4-panel)
- `Figure_4.jpg`: Angular bias analysis (4-panel)
- `Figure_5.jpg`: Raw ensemble MSD curves
- `Figure_6.jpg`: Corrected ensemble MSD curves
- `Mass_vs_Sigma_Diagnostic_FIG4.jpg`
- `Individual_MSDs_RAW_FIG1.jpg`

---

## Citation

If you use this pipeline in your research, please cite:

```
[Your publication details]
de Boer, S., et al. (2025). Single Particle Tracking Analysis of DNA Dynamics
in Cellular Environments. [Journal], [Volume], [Pages].
```

**Key Methods Papers:**
- Trackpy: Allan, D. B., et al. (2016). *Soft Matter*, 12(29), 6152-6158.
- MSD Analysis: Michalet, X. (2010). *Physical Review E*, 82(4), 041914.
- Van Hove Functions: Weeks, E. R., et al. (2000). *Science*, 287(5453), 627-631.

---

## Troubleshooting

### Common Issues

#### 1. **"No particles detected"**
**Causes:**
- `minmass` too high
- `diameter` incorrect

**Solutions:**
```python
# Lower minmass gradually
minmass = 500  # Try lower values

# Verify particle size visually
tp.annotate(tp.locate(frames[0], 11, minmass=100), frames[0])
```

---

#### 2. **"Subpixel bias detected"**
**Cause:** `diameter` too small

**Solution:**
```python
# Increase diameter by 2 pixels
diameter = 13  # Was 11
```

---

#### 3. **"High RMS drift (> 0.5 μm)"**
**Causes:**
- Stage drift during acquisition
- Sample movement
- Temperature fluctuations

**Solutions:**
- Check drift plot: `tm_drift_plot.png`
- Ensure `tp.subtract_drift()` was applied
- Consider shorter acquisition times
- Improve environmental stability

---

#### 4. **"α values unrealistic (α < 0 or α > 2)"**
**Causes:**
- Poor MSD fits (check R²)
- Insufficient trajectory length
- Inappropriate fit regime

**Solutions:**
```python
# Increase R² threshold
R_SQUARED_THRESHOLD = 0.8

# Check trajectory lengths
print(length_df.describe())

# Adjust fit regime (indices in powerfit function)
powerfit(msd, a=20, b=80)  # Fit from lag 20-80
```

---

#### 5. **"Memory Error / Crash during processing"**
**Causes:**
- Too many particles
- Insufficient RAM

**Solutions:**
```python
# Process fewer frames at a time
frames = frames[:400]  # Process first 400 frames

# Reduce parallel processing
processes = 1  # In tp.batch()

# Split dataset into smaller experiments
```

---

#### 6. **"Negative MSD after correction"**
**Cause:** Localization error overestimated or true MSD is very small

**Solutions:**
```python
# Check calibration values
print(f"Correction term: {4 * SIGMA_STUCK_MICRONS**2}")

# Verify stuck bead results
# Ensure beads were truly immobile

# Use MSD.clip(lower=0) to avoid negative values (already implemented)
```

---

#### 7. **"Inconsistent results between runs"**
**Causes:**
- Random seed variation
- Insufficient statistics

**Solutions:**
```python
# Set random seeds
np.random.seed(42)

# Check number of tracks
print(f"Total tracks: {n_tracks}")  # Should be > 50 per condition

# Verify trajectory length distribution
plt.hist(length_df, bins=50)
```

---

### Performance Optimization

**For large datasets (10,000+ particles):**

1. **Use parallel processing:**
```python
tp.batch(..., processes=4)  # Use 4 CPU cores
```

2. **Process experiments sequentially:**
```python
# Comment out unwanted conditions in CONDITIONS_MAP
```

3. **Reduce MAX_LAGTIME:**
```python
MAX_LAGTIME = 50  # Instead of 100
```

4. **Subsample trajectories for testing:**
```python
test_fraction = 0.1
if np.random.random() < test_fraction:
    # Process this trajectory
```

---

### Data Quality Checks

**Before running full analysis, verify:**

1. **Localization quality:**
   - Subpixel histogram should be flat
   - KS test p