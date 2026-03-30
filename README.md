# Antimicrobial Resistance (AMR) Tracker

A Python and SQLite surveillance tool that tracks bacterial
resistance patterns, predicts resistance spread using ML
and generates WHO-style surveillance reports.
Built as Phase 3 - Project 2 of my bioinformatics-to-tech portfolio.

## Why this matters

AMR kills 700,000 people every year.
By 2050 it could kill 10 million people annually.
This tool tracks resistance patterns to help fight superbugs.

## Features

- SQLite database for resistance records and patient cases
- Tracks bacteria and antibiotic resistance combinations
- Critical resistance alerts (above 50% threshold)
- Resistance pattern analysis with visual bar charts
- ML model to predict resistance in new patients (80% accuracy)
- WHO-style surveillance report generation
- AMR dashboard with 5 visualization charts
- Export all data to CSV files

## Critical Findings from Sample Data

Staphylococcus aureus  + Methicillin    : 85.0% (MRSA)
Escherichia coli       + Ampicillin     : 82.0% (MDR E.coli)
Acinetobacter baumannii + Carbapenem   : 78.0% (CRAB)
Klebsiella pneumoniae  + Ciprofloxacin : 78.0% (CRKP)
Escherichia coli       + Tetracycline  : 75.0%

10 critical resistance combinations detected!

## Database Structure

- bacteria table         — 8 clinically important strains
- antibiotics table      — 10 major antibiotic classes
- resistance_records     — 19 surveillance records
- patient_cases          — 100 simulated patient cases

## Tech Stack

- Python 3.13
- SQLite3
- Pandas
- Scikit-learn (Random Forest)
- Matplotlib
- Git + GitHub

## Usage

python amr_tracker.py

Menu options:
1. View resistance summary
2. View critical resistance alerts
3. Analyze resistance patterns
4. Train ML resistance predictor
5. Generate surveillance report
6. Visualize AMR dashboard
7. Export data to CSV
8. Load sample data
0. Exit

## Project Roadmap

- Phase 3 Project 1 - Plant Disease Detector    - Done
- Phase 3 Project 2 - AMR Tracker               - Done
- Phase 3 Project 3 - Pandemic Spread Simulator - Coming soon
- Phase 3 Project 4 - Cancer Biomarker Discovery- Coming soon
- Phase 3 Project 5 - Mental Health Screener    - Coming soon

## Author

Padma Shree Jena
Bioinformatics + Tech Enthusiast | Python | R | Bash
GitHub: https://github.com/Paddu2006

## License

MIT License
