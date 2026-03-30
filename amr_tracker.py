# ============================================
# Antimicrobial Resistance (AMR) Tracker
# Author: Padma Shree
# Phase 3 - Project 2
# Features: SQLite DB, Resistance tracking,
#            Pattern analysis, ML prediction,
#            Surveillance report, Visualization
# ============================================

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime, timedelta
import csv
import os
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

DB_FILE = "amr_database.db"

def create_database():
    """Create AMR SQLite database."""
    conn = sqlite3.connect(DB_FILE)
    c    = conn.cursor()

    # Bacteria table
    c.execute("""
        CREATE TABLE IF NOT EXISTS bacteria (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            gram_type   TEXT,
            category    TEXT,
            description TEXT
        )
    """)

    # Antibiotics table
    c.execute("""
        CREATE TABLE IF NOT EXISTS antibiotics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL UNIQUE,
            class       TEXT,
            mechanism   TEXT
        )
    """)

    # Resistance records table
    c.execute("""
        CREATE TABLE IF NOT EXISTS resistance_records (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            bacteria_id     INTEGER,
            antibiotic_id   INTEGER,
            resistance_pct  REAL,
            sample_size     INTEGER,
            location        TEXT,
            date_recorded   TEXT,
            source          TEXT,
            FOREIGN KEY (bacteria_id)   REFERENCES bacteria(id),
            FOREIGN KEY (antibiotic_id) REFERENCES antibiotics(id)
        )
    """)

    # Patients table
    c.execute("""
        CREATE TABLE IF NOT EXISTS patient_cases (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      TEXT,
            bacteria        TEXT,
            antibiotic      TEXT,
            resistant       INTEGER,
            age             INTEGER,
            location        TEXT,
            date_recorded   TEXT,
            outcome         TEXT
        )
    """)

    conn.commit()
    conn.close()
    print("   AMR Database created successfully!")

def get_conn():
    return sqlite3.connect(DB_FILE)

# ─────────────────────────────────────────────
# LOAD SAMPLE DATA
# ─────────────────────────────────────────────

def load_sample_data():
    """Load realistic AMR sample data."""
    conn = get_conn()
    c    = conn.cursor()

    # Bacteria
    bacteria = [
        ("Staphylococcus aureus",     "Gram-positive", "ESKAPE", "Causes skin, blood and lung infections"),
        ("Klebsiella pneumoniae",     "Gram-negative", "ESKAPE", "Causes pneumonia and bloodstream infections"),
        ("Acinetobacter baumannii",   "Gram-negative", "ESKAPE", "Hospital-acquired infections"),
        ("Pseudomonas aeruginosa",    "Gram-negative", "ESKAPE", "Lung infections in immunocompromised"),
        ("Enterococcus faecium",      "Gram-positive", "ESKAPE", "Urinary tract and bloodstream infections"),
        ("Escherichia coli",          "Gram-negative", "Common", "UTIs, gastroenteritis, sepsis"),
        ("Mycobacterium tuberculosis","Acid-fast",     "Critical","Tuberculosis - major global killer"),
        ("Streptococcus pneumoniae",  "Gram-positive", "Common", "Pneumonia, meningitis, ear infections"),
    ]
    for b in bacteria:
        try:
            c.execute("INSERT INTO bacteria (name, gram_type, category, description) VALUES (?,?,?,?)", b)
        except:
            pass

    # Antibiotics
    antibiotics = [
        ("Methicillin",    "Beta-lactam",     "Cell wall synthesis inhibition"),
        ("Vancomycin",     "Glycopeptide",    "Cell wall synthesis inhibition"),
        ("Ciprofloxacin",  "Fluoroquinolone", "DNA gyrase inhibition"),
        ("Carbapenem",     "Beta-lactam",     "Cell wall synthesis inhibition"),
        ("Colistin",       "Polymyxin",       "Cell membrane disruption"),
        ("Tetracycline",   "Tetracycline",    "Protein synthesis inhibition"),
        ("Rifampicin",     "Rifamycin",       "RNA polymerase inhibition"),
        ("Azithromycin",   "Macrolide",       "Protein synthesis inhibition"),
        ("Ampicillin",     "Beta-lactam",     "Cell wall synthesis inhibition"),
        ("Levofloxacin",   "Fluoroquinolone", "DNA gyrase inhibition"),
    ]
    for a in antibiotics:
        try:
            c.execute("INSERT INTO antibiotics (name, class, mechanism) VALUES (?,?,?)", a)
        except:
            pass

    conn.commit()

    # Resistance records — realistic data
    c.execute("SELECT id, name FROM bacteria")
    bact_map = {name: bid for bid, name in c.fetchall()}
    c.execute("SELECT id, name FROM antibiotics")
    anti_map = {name: aid for aid, name in c.fetchall()}

    resistance_data = [
        # MRSA (Methicillin-resistant S. aureus)
        ("Staphylococcus aureus", "Methicillin",   85.0, 1200, "India",  "2024-01-15"),
        ("Staphylococcus aureus", "Methicillin",   72.0, 980,  "USA",    "2024-02-20"),
        ("Staphylococcus aureus", "Vancomycin",    12.0, 1200, "India",  "2024-01-15"),
        ("Staphylococcus aureus", "Ciprofloxacin", 65.0, 800,  "India",  "2024-03-10"),
        # CRKP (Carbapenem-resistant K. pneumoniae)
        ("Klebsiella pneumoniae", "Carbapenem",    45.0, 650,  "India",  "2024-01-20"),
        ("Klebsiella pneumoniae", "Carbapenem",    28.0, 520,  "Europe", "2024-02-15"),
        ("Klebsiella pneumoniae", "Colistin",      18.0, 650,  "India",  "2024-01-20"),
        ("Klebsiella pneumoniae", "Ciprofloxacin", 78.0, 900,  "India",  "2024-03-05"),
        # MDR E. coli
        ("Escherichia coli",      "Ampicillin",    82.0, 2100, "India",  "2024-01-10"),
        ("Escherichia coli",      "Ciprofloxacin", 68.0, 2100, "India",  "2024-01-10"),
        ("Escherichia coli",      "Tetracycline",  75.0, 1800, "India",  "2024-02-28"),
        ("Escherichia coli",      "Carbapenem",    22.0, 1500, "India",  "2024-03-15"),
        # TB
        ("Mycobacterium tuberculosis", "Rifampicin",  32.0, 450, "India", "2024-01-25"),
        ("Mycobacterium tuberculosis", "Rifampicin",  18.0, 320, "Africa","2024-02-10"),
        # Pseudomonas
        ("Pseudomonas aeruginosa","Carbapenem",    52.0, 380,  "India",  "2024-02-05"),
        ("Pseudomonas aeruginosa","Colistin",      8.0,  380,  "India",  "2024-02-05"),
        ("Pseudomonas aeruginosa","Ciprofloxacin", 71.0, 420,  "India",  "2024-03-20"),
        # Acinetobacter
        ("Acinetobacter baumannii","Carbapenem",   78.0, 290,  "India",  "2024-01-30"),
        ("Acinetobacter baumannii","Colistin",     15.0, 290,  "India",  "2024-01-30"),
    ]

    for bact, anti, res_pct, n, loc, date in resistance_data:
        bid = bact_map.get(bact)
        aid = anti_map.get(anti)
        if bid and aid:
            c.execute("""
                INSERT INTO resistance_records
                (bacteria_id, antibiotic_id, resistance_pct,
                 sample_size, location, date_recorded, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (bid, aid, res_pct, n, loc, date, "WHO Surveillance"))

    # Patient cases
    np.random.seed(42)
    bacteria_list  = list(bact_map.keys())[:6]
    antibiotic_list = list(anti_map.keys())[:8]
    locations      = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]
    outcomes       = ["Recovered", "Recovered", "Recovered", "Deceased", "Transferred"]

    for i in range(100):
        bact  = random.choice(bacteria_list)
        anti  = random.choice(antibiotic_list)
        res   = random.choice([0, 0, 1])
        age   = random.randint(18, 85)
        loc   = random.choice(locations)
        date  = (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 364))).strftime("%Y-%m-%d")
        outcome = random.choice(outcomes)
        c.execute("""
            INSERT INTO patient_cases
            (patient_id, bacteria, antibiotic, resistant,
             age, location, date_recorded, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (f"PT{1000+i}", bact, anti, res, age, loc, date, outcome))

    conn.commit()
    conn.close()
    print("   Sample data loaded successfully!")
    print("   - 8 bacteria strains")
    print("   - 10 antibiotics")
    print("   - 19 resistance records")
    print("   - 100 patient cases")

# ─────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ─────────────────────────────────────────────

def get_resistance_summary():
    """Get resistance summary statistics."""
    conn = get_conn()
    df   = pd.read_sql_query("""
        SELECT b.name as bacteria, a.name as antibiotic,
               r.resistance_pct, r.sample_size, r.location,
               r.date_recorded
        FROM resistance_records r
        JOIN bacteria b ON r.bacteria_id = b.id
        JOIN antibiotics a ON r.antibiotic_id = a.id
        ORDER BY r.resistance_pct DESC
    """, conn)
    conn.close()
    return df

def get_critical_resistance():
    """Get critically resistant combinations (>50%)."""
    df = get_resistance_summary()
    return df[df["resistance_pct"] > 50].sort_values(
        "resistance_pct", ascending=False
    )

def analyze_patterns():
    """Analyze resistance patterns."""
    conn = get_conn()

    # Resistance by bacteria
    bact_avg = pd.read_sql_query("""
        SELECT b.name, AVG(r.resistance_pct) as avg_resistance,
               COUNT(*) as records
        FROM resistance_records r
        JOIN bacteria b ON r.bacteria_id = b.id
        GROUP BY b.name
        ORDER BY avg_resistance DESC
    """, conn)

    # Resistance by antibiotic
    anti_avg = pd.read_sql_query("""
        SELECT a.name, AVG(r.resistance_pct) as avg_resistance,
               COUNT(*) as records
        FROM resistance_records r
        JOIN antibiotics a ON r.antibiotic_id = a.id
        GROUP BY a.name
        ORDER BY avg_resistance DESC
    """, conn)

    # Resistance by location
    loc_avg = pd.read_sql_query("""
        SELECT location, AVG(resistance_pct) as avg_resistance,
               COUNT(*) as records
        FROM resistance_records
        GROUP BY location
        ORDER BY avg_resistance DESC
    """, conn)

    conn.close()
    return bact_avg, anti_avg, loc_avg

# ─────────────────────────────────────────────
# ML PREDICTION
# ─────────────────────────────────────────────

def train_resistance_predictor():
    """Train ML model to predict resistance."""
    conn = get_conn()
    df   = pd.read_sql_query("""
        SELECT resistant, age, location, bacteria, antibiotic
        FROM patient_cases
    """, conn)
    conn.close()

    # Encode categorical variables
    df["location_enc"] = pd.Categorical(df["location"]).codes
    df["bacteria_enc"] = pd.Categorical(df["bacteria"]).codes
    df["antibiotic_enc"] = pd.Categorical(df["antibiotic"]).codes

    X = df[["age", "location_enc", "bacteria_enc", "antibiotic_enc"]]
    y = df["resistant"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model    = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n   ML Model trained!")
    print(f"   Resistance prediction accuracy: {accuracy*100:.2f}%")

    return model, accuracy

# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def visualize_amr():
    """Generate AMR surveillance dashboard."""
    df_summary          = get_resistance_summary()
    bact_avg, anti_avg, loc_avg = analyze_patterns()

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("AMR Surveillance Dashboard",
                fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    # Chart 1 — Top resistant bacteria
    ax1 = fig.add_subplot(gs[0, 0])
    colors1 = ["#E91E63" if r > 70 else "#FF9800" if r > 40 else "#4CAF50"
               for r in bact_avg["avg_resistance"]]
    bars1 = ax1.barh(bact_avg["name"].str[:20],
                    bact_avg["avg_resistance"],
                    color=colors1, edgecolor="black")
    ax1.axvline(x=50, color="red", linestyle="--", alpha=0.7,
               label="Critical threshold (50%)")
    ax1.set_title("Average Resistance by Bacteria", fontweight="bold")
    ax1.set_xlabel("Resistance (%)")
    ax1.legend(fontsize=7)

    # Chart 2 — Antibiotic effectiveness
    ax2 = fig.add_subplot(gs[0, 1])
    colors2 = ["#E91E63" if r > 70 else "#FF9800" if r > 40 else "#4CAF50"
               for r in anti_avg["avg_resistance"]]
    bars2 = ax2.bar(anti_avg["name"].str[:10],
                   anti_avg["avg_resistance"],
                   color=colors2, edgecolor="black")
    ax2.axhline(y=50, color="red", linestyle="--", alpha=0.7)
    for bar, val in zip(bars2, anti_avg["avg_resistance"]):
        ax2.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5, f"{val:.0f}%",
                ha="center", fontsize=7, fontweight="bold")
    ax2.set_title("Resistance by Antibiotic", fontweight="bold")
    ax2.set_ylabel("Avg Resistance (%)")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Chart 3 — Resistance by location
    ax3 = fig.add_subplot(gs[0, 2])
    colors3 = ["#E91E63" if r > 60 else "#FF9800" if r > 40 else "#4CAF50"
               for r in loc_avg["avg_resistance"]]
    bars3 = ax3.bar(loc_avg["location"],
                   loc_avg["avg_resistance"],
                   color=colors3, edgecolor="black")
    for bar, val in zip(bars3, loc_avg["avg_resistance"]):
        ax3.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5, f"{val:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax3.set_title("Resistance by Location", fontweight="bold")
    ax3.set_ylabel("Avg Resistance (%)")

    # Chart 4 — Critical resistance heatmap
    ax4 = fig.add_subplot(gs[1, 0:2])
    pivot = df_summary.pivot_table(
        values="resistance_pct",
        index="bacteria",
        columns="antibiotic",
        aggfunc="mean"
    ).fillna(0)
    im = ax4.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=100)
    ax4.set_xticks(range(len(pivot.columns)))
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax4.set_yticklabels([n[:20] for n in pivot.index], fontsize=7)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if val > 0:
                ax4.text(j, i, f"{val:.0f}",
                        ha="center", va="center",
                        fontsize=7, fontweight="bold",
                        color="white" if val > 60 else "black")
    plt.colorbar(im, ax=ax4, label="Resistance %")
    ax4.set_title("Resistance Heatmap (Bacteria vs Antibiotic)",
                 fontweight="bold")

    # Chart 5 — Patient outcomes
    ax5 = fig.add_subplot(gs[1, 2])
    conn = get_conn()
    outcomes = pd.read_sql_query("""
        SELECT outcome, COUNT(*) as count
        FROM patient_cases GROUP BY outcome
    """, conn)
    conn.close()
    out_colors = {"Recovered": "#4CAF50", "Deceased": "#E91E63",
                 "Transferred": "#FF9800"}
    colors5 = [out_colors.get(o, "#9E9E9E") for o in outcomes["outcome"]]
    ax5.pie(outcomes["count"], labels=outcomes["outcome"],
           colors=colors5, autopct="%1.1f%%", startangle=90)
    ax5.set_title("Patient Outcomes", fontweight="bold")

    plt.tight_layout()
    filename = "amr_dashboard.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   Dashboard saved as: {filename}")

# ─────────────────────────────────────────────
# SURVEILLANCE REPORT
# ─────────────────────────────────────────────

def generate_surveillance_report():
    """Generate AMR surveillance report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"amr_surveillance_report_{timestamp}.txt"
    df        = get_resistance_summary()
    critical  = get_critical_resistance()
    bact_avg, anti_avg, loc_avg = analyze_patterns()

    conn = get_conn()
    total_records = pd.read_sql_query(
        "SELECT COUNT(*) as n FROM resistance_records", conn
    ).iloc[0]["n"]
    total_patients = pd.read_sql_query(
        "SELECT COUNT(*) as n FROM patient_cases", conn
    ).iloc[0]["n"]
    conn.close()

    with open(filename, "w") as f:
        f.write("="*65 + "\n")
        f.write("ANTIMICROBIAL RESISTANCE SURVEILLANCE REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}\n")
        f.write("Author: Padma Shree | Phase 3 - Project 2\n")
        f.write("="*65 + "\n\n")

        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Total resistance records  : {total_records}\n")
        f.write(f"Total patient cases       : {total_patients}\n")
        f.write(f"Critical combinations (>50%): {len(critical)}\n\n")

        f.write("CRITICAL RESISTANCE ALERTS\n")
        f.write("-"*40 + "\n")
        for _, row in critical.iterrows():
            f.write(f"  {row['bacteria'][:30]:<30} + "
                   f"{row['antibiotic']:<15}: "
                   f"{row['resistance_pct']:.1f}% resistant\n")

        f.write("\nTOP RESISTANT BACTERIA\n")
        f.write("-"*40 + "\n")
        for _, row in bact_avg.head(5).iterrows():
            f.write(f"  {row['name'][:35]:<35}: "
                   f"{row['avg_resistance']:.1f}% avg resistance\n")

        f.write("\nMOST COMPROMISED ANTIBIOTICS\n")
        f.write("-"*40 + "\n")
        for _, row in anti_avg.head(5).iterrows():
            f.write(f"  {row['name']:<20}: "
                   f"{row['avg_resistance']:.1f}% avg resistance\n")

        f.write("\nRECOMMENDATIONS\n")
        f.write("-"*40 + "\n")
        f.write("  1. Restrict use of Ciprofloxacin and Ampicillin\n")
        f.write("  2. Increase surveillance for Carbapenem resistance\n")
        f.write("  3. Implement antibiotic stewardship programs\n")
        f.write("  4. Promote infection prevention measures\n")
        f.write("  5. Invest in new antibiotic development\n")
        f.write("\n" + "="*65 + "\n")

    print(f"   Surveillance report saved: {filename}")
    return filename

# ─────────────────────────────────────────────
# EXPORT TO CSV
# ─────────────────────────────────────────────

def export_to_csv():
    """Export all data to CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = get_resistance_summary()
    df.to_csv(f"amr_resistance_data_{timestamp}.csv", index=False)
    print(f"   Resistance data exported!")

    conn = get_conn()
    patients = pd.read_sql_query(
        "SELECT * FROM patient_cases", conn
    )
    conn.close()
    patients.to_csv(f"amr_patient_cases_{timestamp}.csv", index=False)
    print(f"   Patient cases exported!")

# ─────────────────────────────────────────────
# DISPLAY MENU
# ─────────────────────────────────────────────

def show_menu():
    conn = get_conn()
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM resistance_records")
    records = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM patient_cases")
    patients = c.fetchone()[0]
    conn.close()

    print("\n" + "="*55)
    print("  AMR TRACKER")
    print("="*55)
    print(f"  Resistance records : {records}")
    print(f"  Patient cases      : {patients}")
    print("="*55)
    print("  1. View resistance summary")
    print("  2. View critical resistance alerts")
    print("  3. Analyze resistance patterns")
    print("  4. Train ML resistance predictor")
    print("  5. Generate surveillance report")
    print("  6. Visualize AMR dashboard")
    print("  7. Export data to CSV")
    print("  8. Load sample data")
    print("  0. Exit")
    print("="*55)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("    Antimicrobial Resistance Tracker")
    print("    Author: Padma Shree | Phase 3 - Project 2")
    print("="*55)

    create_database()

    while True:
        show_menu()
        choice = input("\nEnter choice: ").strip()

        if choice == "1":
            df = get_resistance_summary()
            print(f"\n   Resistance Summary ({len(df)} records):")
            print(f"   {'Bacteria':<35} {'Antibiotic':<15} {'Resistance':>10} {'Location'}")
            print("   " + "-"*75)
            for _, row in df.iterrows():
                alert = " ← CRITICAL" if row["resistance_pct"] > 70 else ""
                print(f"   {row['bacteria'][:34]:<35} "
                      f"{row['antibiotic']:<15} "
                      f"{row['resistance_pct']:>9.1f}% "
                      f"{row['location']}{alert}")

        elif choice == "2":
            critical = get_critical_resistance()
            print(f"\n   CRITICAL RESISTANCE ALERTS (>50%)")
            print(f"   {len(critical)} critical combinations found!")
            print(f"\n   {'Bacteria':<35} {'Antibiotic':<15} {'Resistance':>10}")
            print("   " + "-"*65)
            for _, row in critical.iterrows():
                print(f"   {row['bacteria'][:34]:<35} "
                      f"{row['antibiotic']:<15} "
                      f"{row['resistance_pct']:>9.1f}%")

        elif choice == "3":
            bact_avg, anti_avg, loc_avg = analyze_patterns()
            print("\n   TOP RESISTANT BACTERIA:")
            for _, row in bact_avg.iterrows():
                bar = "█" * int(row["avg_resistance"] / 5)
                print(f"   {row['name'][:30]:<30} {bar} {row['avg_resistance']:.1f}%")
            print("\n   MOST COMPROMISED ANTIBIOTICS:")
            for _, row in anti_avg.iterrows():
                bar = "█" * int(row["avg_resistance"] / 5)
                print(f"   {row['name']:<20} {bar} {row['avg_resistance']:.1f}%")

        elif choice == "4":
            model, accuracy = train_resistance_predictor()

        elif choice == "5":
            generate_surveillance_report()

        elif choice == "6":
            print("   Generating AMR dashboard...")
            visualize_amr()

        elif choice == "7":
            export_to_csv()

        elif choice == "8":
            load_sample_data()

        elif choice == "0":
            print("\n   Thank you Paddu! AMR data saved! 🦠")
            break
        else:
            print("   Invalid choice!")