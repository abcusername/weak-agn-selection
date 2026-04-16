# Weak AGN Selection from Spectroscopic Diagnostics and Time-Domain Variability

This repository contains my research workflow for identifying weak active galactic nucleus (AGN) candidates by combining spectroscopic classification from DESI and optical variability analysis from ZTF light curves.

## Project Overview

Weak or obscured AGN can be difficult to identify using optical spectroscopy alone, especially in host-dominated systems. In this project, I explored whether time-domain optical variability can help recover AGN activity that is weak, diluted, or ambiguous in standard spectroscopic diagnostics.

The current workflow combines:

- **BPT-based spectroscopic classification** of galaxies into Seyfert, LINER, Composite, and Star-forming classes
- **ZTF light-curve download and cleaning**
- **Variability quantification** using fractional variability statistics
- **Noise-control analysis** based on magnitude-dependent baselines
- **Quality-control inspection** of representative light curves
- **Center-matching and follow-up checks** motivated by advisor feedback

## Scientific Goal

The main goal of this project is to test whether optical variability can help identify weak AGN that may be missed, diluted, or ambiguously classified in emission-line diagnostics.

More specifically, this project aims to answer:

1. Which spectroscopic classes show excess optical variability?
2. Can variability recover AGN-like behavior in weak or host-dominated systems?
3. How much of the apparent variability is real, and how much is driven by photometric noise, systematics, or off-center contamination?

## What I Have Done

### 1. BPT Classification of DESI Sources

I first classified DESI galaxies into standard emission-line categories using BPT diagnostics:

- Seyfert
- LINER
- Composite
- Star-forming

This provided the spectroscopic sample basis for the later time-domain analysis.

### 2. Download and Cleaning of ZTF Light Curves

I downloaded ZTF light curves for selected subsamples and built a cleaning workflow for the r-band data.

A key issue in the early version was that multiple nearby sources could be mixed into a single light curve. In the current **MINFIX** version, I improved the pipeline by:

- enforcing **central-source uniqueness**
- keeping only the **closest OID** to the target position
- using a **single band (r-band)** consistently
- applying quality-control filters on photometric measurements

### 3. Variability Statistics

I computed variability statistics, including **fractional variability** (`Fvar`), and compared the variability distributions across different BPT classes.

Initial results showed that LINER, Seyfert, Composite, and Star-forming samples can all contain apparently high-Fvar objects. However, later analysis showed that a simple fixed `Fvar > 5%` threshold is not reliable across the full magnitude range.

### 4. Magnitude-Dependent Noise Diagnosis

A major result of the current stage is that `Fvar` increases systematically toward fainter sources, indicating that photometric noise and systematics dominate at the faint end.

To address this, I introduced a **magnitude-dependent noise baseline**:

- within each mean-magnitude bin
- use the **95th percentile of Fvar in Star-forming galaxies**
- as a reference noise threshold

This provides a more controlled way to evaluate whether Seyfert or LINER samples show **excess variability above the expected noise floor**.

### 5. QC on Bright Seyfert Control Sample

To ensure physical reliability before moving to weaker systems, I built a **bright, well-sampled Seyfert control sample**, using criteria such as:

- `Mean_mag < 18`
- `N > 500`

For this control sample, I generated QC booklets with:

- cleaned light curves
- error bars
- outlier flags
- near-limit flags
- metadata summaries

### 6. High-Variability Case Inspection

I also inspected representative high-variability objects individually, especially:

- high-Fvar LINERs
- Seyfert–LINER morphological comparisons
- Star-forming high-Fvar cases that may be noise-dominated or transient-like

This step is important because high `Fvar` alone is not sufficient to identify AGN activity.

### 7. Current Focus

Based on advisor feedback, the project is now focused on:

- checking **off-center spectra**
- avoiding **double counting**
- comparing **multi-band variability**
- examining **low-variability Seyferts**
- improving variability metrics beyond a fixed Fvar threshold

## Repository Structure

```text
weak-agn-selection/
├── docs/        # progress reports, summaries, advisor-related materials
├── figures/     # selected BPT plots, variability figures, and example light curves
├── results/     # summary tables and compact outputs
├── src/         # main analysis scripts
