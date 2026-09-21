# DNA Translocation Tracking - Analysis Pipeline

**Authors:** Sophie de Boer, Aswin Muralidharan  
**Project:** Single-particle tracking of DNA in cells  
**Language:** Python 3  

---

## Overview

This repository contains a sequential analysis pipeline for single-particle tracking (SPT) of fluorescently labelled DNA fragments moving through anomalous transport in cell environments. The pipeline takes raw fluorescence image sequences from microscopy and produces quantitative diffusion statistics including mean squared displacement (MSD), apparent diffusion coefficient, anomalous exponent, Van Hove distributions, angular bias, and radius of gyration.

The pipeline is divided into four conceptual stages:

| Stage 				| Scripts 		| Description 									|
|---------------------------------------|-----------------------|-------------------------------------------------------------------------------|
| **I. Image Processing & Tracking** 	| 01-04 		| Raw image preprocessing, particle detection, linking, and drift QC 		|
| **II. Calibration** 			| 05a-05c 		| Localization error quantification from stuck beads and dim dye references 	|
| **III. Core Analysis** 		| 06-09 		| MSD, Van Hove, angular bias, and radius of gyration 				|
| **IV. Supplementary ** 		| 10-12 		| Sensitivity checks and trajectory visualization 				|

---

## Stage I - Image Processing & Tracking

### `01_image_preprocessing.py`
Applies bandpass filtering to raw fluorescence image sequences to suppress noise and background before particle detection.

- Loads `.tif` image sequences from `A_rawtif/` folders
- Applies a Gaussian low-pass filter followed by a boxcar background subtraction
- Clips negative values (bandpass threshold)
- Saves processed frames as `.tiff` to `B_ProcessedData/`
- Automatically skips already-processed experiment folders

**Key parameters to set:** `filepath` (raw data), `filepath2` (output), `N` (number of frames), bandpass threshold (`40`)

---

### `02_particle_tracking.py`
Detects fluorescent particles in processed frames and links them into trajectories using the Trackpy library.

- Locates particles per frame (`tp.locate`) with configurable diameter and minimum brightness
- Links detections across frames into trajectories (`tp.link`)
- Filters short or spurious tracks by length, mass, size, and eccentricity
- Computes and subtracts ensemble drift (`tp.compute_drift`, `tp.subtract_drift`)
- Saves filtered, drift-corrected trajectories (`tm/`) and diagnostic plots
- Computes individual and ensemble MSDs as a quick preview

**Key parameters to set:** `EXPERIMENT`, `DIAMETER`, `MINMASS`, `MAX_DISPLACEMENT`, `MEMORY`, `MIN_FRAMES`, `EXCLUDE_PARTICLES`

---

### `03_drift_correction_validation.py`
Validates that drift subtraction worked correctly by comparing MSD curves before and after correction.

- Loads raw (tf) and drift-corrected (tm) trajectory files
- Computes per-particle MSD and fits power law curves
- Plots individual MSD trajectories colored by diffusion coefficient
- Intended as a quality checkpoint

> Drift validation should be confirmed immediately after tracking. If drift residuals remain, trajectory data must be processed again before calibration or analysis.

---

### `04_trajectory_length_and_gap_filling.py`
Prepares trajectories for MSD analysis by filling frame gaps with `NaN` and characterizing track length distributions.

- Reads linked trajectory CSV files
- Re-indexes trajectories to continuous frame ranges, inserting `NaN` for missing frames
- Computes per-particle MSD curves
- Plots trajectory length distributions and colored MSD overlays
- Applies an R² quality filter (`r² ≥ 0.9`) to remove poorly behaved tracks

> Gap-filling is required before MSD analysis because the MSD calculation assumes evenly spaced positions. Without this step, frame gaps produce incorrect lag-time assignments.

---

## Stage II - Calibration

All three calibration scripts must be run before the core analysis. The values they produce (σ in µm) are hard-coded into the analysis scripts.

### `05a_calibration_stuck_beads.py`
Measures the localization error (σ) from immobilized fluorescent beads.

- Loads trajectory data from stuck (immobile) beads
- For truly immobile particles, all position fluctuations are localization noise: `<Δx²> + <Δy²> = 2σ²`
- Reports per-trajectory σ values and fits a Gaussian to the distribution
- **Output:** `SIGMA_STUCK_MICRONS` and `MASS_STUCK_REFERENCE`, paste these into all downstream analysis scripts

---

### `05b_calibration_dim_particles.py`
Measures the localization error for dim fluorescent labels (YOYO-1 dye) by comparison with a brighter reference (Cy3).

- Uses the relationship: `4σ²_dim ≈ MSD_raw_dim - MSD_raw_bright` at short lag times
- Cy3 (bright) serves as the reference whose true MSD is known
- **Output:** `SIGMA_DIM_MICRONS` and `MASS_DIM_REFERENCE` - used for all YOYO-1-labelled DNA samples (100 bp, 250 bp, 500 bp, and cell-type experiments)

---

### `05c_calibration_validation.py`
Comprehensive validation of the σ values derived in `05a` and `05b`.

- Compares ensemble MSD curves with and without localization correction
- Checks consistency of σ estimates across bead, Cy3, and YOYO-1 datasets
- Produces diagnostic grid plots for visual inspection of correction quality
- Run this script and inspect the figures before trusting any corrected MSD values

---

## Stage III - Core Analysis

Scripts in this stage follow a consistent `a` / `b` naming convention:  
- **`a`** = DNA fragment size series (100 bp, 250 bp, 500 bp, 4713 bp, 7614 bp, 9553 bp) in CHO cells  
- **`b`** = Cell type comparison (MDA-MB-231, MCF-7, MCF-10A)

All scripts apply per-particle localization correction using the σ values from Stage II and filter trajectories against a bead-derived noise floor.

---

### `06a_msd_analysis_main.py`
Main MSD analysis for the DNA fragment size series.

- Loads drift-corrected trajectories for all six DNA sizes plus bead controls
- Applies per-particle localization correction: `MSD_corrected = MSD_raw - 4σ²`
- Filters trajectories whose corrected MSD does not exceed the bead noise floor for ≥ 98% of shared lag times
- Fits power-law `MSD = 4D·t^α` to extract diffusion coefficient D and anomalous exponent α between 2-8 s
- Produces log-log MSD plots with ensemble overlays, violin plots of D and α, and summary statistics

**Key output:** Apparent diffusion coefficient `Dapp` at 8 s lag time per condition

---

### `06b_msd_analysis_cellular.py`
MSD analysis for the cell-type comparison experiments.

- Same analysis pipeline as `06a` applied to MDA-MB-231, MCF-7, MCF-10A, and ATP-depletion conditions
- Includes WT control alongside treated conditions
- Produces equivalent figures to `06a` for direct comparison

---

### `06c_msd_dye_comparison.py`
Compares MSD results between WT Cy3 and YOYO-1 labelled samples to assess whether the dye affects diffusion behaviour.

- Loads both Cy3 and YOYO-1 datasets for the same cell type (MDA-MB-231)
- Applies the appropriate σ correction for each dye
- Produces overlaid MSD curves and violin plots for direct comparison
- Noise floor filtering and Dapp extraction consistent with `06a/b`

> **Why a separate script?** This is a specific methodological validation, not a primary result. Keeping it separate avoids cluttering the main analysis scripts.

---

### `07a_van_hove_analysis.py`
Van Hove displacement analysis for the DNA fragment size series.

- Collects single-lag-time displacements (Δx) across all particles and frames
- Fits Gaussian and non-Gaussian (Laplacian / sum-of-Gaussians) models to the distributions
- Produces 2×2 panel plots at multiple lag times
- Non-Gaussian parameter α₂ quantifies heterogeneity of motion
- Noise floor filtering consistent with `06a`

---

### `07b_van_hove_analysis_cellular.py`
Van Hove displacement analysis for the cell-type comparison.

- Same analysis as `07a` applied to MDA-MB-231, MCF-7, and MCF-10A
- σ correction uses dim-dye values for all three cell types

---

### `08a_angular_bias_analysis.py`
Analyses the directional persistence of particle steps for the DNA fragment size series.

- Computes the angle between consecutive displacement vectors for each trajectory
- A backward bias (peak near 180°) indicates confined/corralled motion; a forward bias indicates directed motion
- Produces polar histograms and reports the directed vs confined ratio per condition
- Noise floor filtering consistent with `06a`

---

### `08b_angular_bias_analysis_cellular.py`
Angular bias analysis for the cell-type comparison.

- Same analysis as `08a` applied to MDA-MB-231, MCF-7, and MCF-10A
- 1×3 linear subplot layout with directed vs confined ratio panel

---

### `09_radius_of_gyration.py`
Computes the radius of gyration Rg for each particle trajectory.

- Uses the first 800 frames of each track
- Rg = √(⟨(x - x_com)²⟩ + ⟨(y - y_com)²⟩): the RMS deviation from the centre of mass
- Optional localization correction: `Rg_corrected² = Rg_raw² - 2σ²`
- Produces per-condition histograms and a combined violin plot
- Compares measured Rg to theoretical values for linear rods and Worm-Like Chain coils

---

## Stage IV - Supplementary & Visualisation

### `10_traj_length_sensitivity.py`
Sensitivity analysis for the effect of trajectory length cutoff on anomalous exponent estimates.

- Re-runs MSD fitting at multiple minimum track length thresholds
- Assesses whether the choice of `MIN_FRAMES` biases α estimates
- Use this to justify the cutoff values used in Stage III scripts

---

### `11_multiple_overlayed_trajectories.py`
Visualises multiple particle trajectories overlaid, colored by diffusion coefficient or time.

- Loads trajectory CSV files and renders `LineCollection` plots
- Useful for visually inspecting trajectory quality and spatial distribution
- Also includes power-law fitting and localization precision reporting (suitable for stuck beads)

---

### `12_single_trajectory_visualization.py`
Detailed visualisation of individual particle trajectories.

- Overlays all trajectories on a brightfield image
- Renders single-particle trajectory colored by time (blue → red)
- Produces displacement-vs-time panels (Δx, Δy, |Δr|)
- Per-particle configuration (zoom window, time truncation, y-axis limits) set in `PARTICLE_CONFIGS`

---

## Dependencies

```
numpy
pandas
scipy
matplotlib
trackpy
pims
Pillow (PIL)
imageio
scikit-image
opencv-python (cv2)
```

Install with:
```bash
pip install numpy pandas scipy matplotlib trackpy pims Pillow imageio scikit-image opencv-python
```

---

## Global Parameters (shared across scripts)

| Parameter 			| Value 		| Description 							|
|-------------------------------|-----------------------|---------------------------------------------------------------|
| `mpp` 			| 0.10091 µm/px 	| Microns per pixel (spatial calibration) 			|
| `fps` 			| 10 			| Frames per second 						|
| `SIGMA_STUCK_MICRONS` 	| 0.013543 µm 		| Localization error - bright beads / Cy3 			|
| `SIGMA_DIM_MICRONS` 		| 0.008834 µm 		| Localization error - dim YOYO-1 label 			|
| `MASS_STUCK_REFERENCE` 	| 3381.5 		| Reference brightness for stuck beads 				|
| `MASS_DIM_REFERENCE` 		| 589.8 		| Reference brightness for dim dye 				|
| `NOISE_FLOOR_FRACTION` 	| 0.98 			| Fraction of lag times a track must exceed bead noise floor 	|

> Update `SIGMA_*` and `MASS_*` values in all analysis scripts after running `05a` and `05b` (and 05c as a check on whether they are good estimates) on your own bead and dye reference data.

---

## Directory Structure (expected)

```
Corrected_Data/
├── Aswin Data/
│   ├── Aswin Data CHO Corrected for Drift/
│   │   └── E_output_data/
│   │       ├── Collected_100bp/tm/
│   │       ├── Collected_250bp/tm/
│   │       └── Collected_500bp/tm/
│   └── Aswin Data MDA-MB-231 Corrected for Drift/
│   │   └── E_output_data/
│   │       ├── Collected_Collected_MDAMB231/tm/
│   └── Aswin Data MCF7 Corrected for Drift/
│   │   └── E_output_data/
│   │       ├── Collected_Collected_MCF7/tm/
│   └── Aswin Data MCF10A Corrected for Drift/
│   │   └── E_output_data/
│   │       ├── Collected_Collected_MCF10A/tm/


├── Sophie Corrected for Gaussian Filter/
│   └── E_output_data/
│       ├── Collected_pEGFP-C1/tm/
│       ├── Collected_BPK4410/tm/
│       └── Collected_hCas9D10A/tm/

├── Control Experiments_Beads/
│   └── 0.22 um/E_output_data/Collected_0.22um/tm/

├── MDA-MB-231/
│   ├── Cy3 MDA-MB-231 WT/
│   │       ├── 21.11.2025/
│   │		 │   ├──E_output_data/
│   │		 │   |		├──Collected_MDAMB231_WT/tm
│   ├── Cy3 MDA-MB-231/
│   │       ├── 500bp_most_recent/
│   │		 │   ├──E_output_data/
│   │		 │   |		├──Collected_Cy3_MDAMB231/tm
│   └── ATP Depletion MDA-MB-231/
│   │       ├── 500bp_most_recent/
│   │		 │   ├──E_output_data/
│   │		 │   |		├──Collected_ATPdepl_MDAMB231/tm
└── Figures/
```

Each experiment folder processed by `02_particle_tracking.py` produces an `E_output_data/<ExperimentName>/` directory containing:

```
E_output_data/ExperimentName/
├── f/          # Raw located features
├── t/          # Linked trajectories (unfiltered)
├── tf/         # Filtered trajectories (pre-drift)
├── tm/         # Drift-corrected, filtered trajectories  ← used by analysis scripts
├── trajectories/
├── im_plot/	 # Individual MSDs of each DNA aggregate located
└── em_plot/	 # Ensemble MSDs of that particular experiment (could be that one cell has multiple DNA aggregates)
```