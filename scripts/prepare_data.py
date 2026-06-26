#!/usr/bin/env python3
"""
Preparation / verification des donnees
======================================
Verifie la base MedlinePlus et affiche ses caracteristiques (nb de maladies,
nb de symptomes, taille du modele correspondant). A lancer avant une simulation.

  python scripts/prepare_data.py
  python scripts/prepare_data.py --csv data/MedlinePlus10.csv
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT
from jobs.rl_diagnosis.knowledge_base import KnowledgeBase
from jobs.rl_diagnosis.job import RLDiagnosisJob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT.data.csv_file)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERREUR : fichier introuvable : {args.csv}")
        sys.exit(1)

    kb = KnowledgeBase.from_csv(args.csv)
    job = RLDiagnosisJob(kb, DEFAULT)
    sizes = [len(s) for s in kb.diseases]
    print("Base de connaissances :", args.csv)
    print(f"  maladies            : {kb.n_diseases}")
    print(f"  symptomes           : {kb.n_symptoms}")
    print(f"  symptomes/maladie   : min {min(sizes)}, max {max(sizes)}, "
          f"moyenne {sum(sizes)/len(sizes):.1f}")
    print(f"  parametres du modele: {job.n_params():,}")
    print(f"  patients valides/eval : {DEFAULT.server.validation_patients}")
    print("\nBase prete. Lancer ensuite :")
    print("  python scripts/run_simulation.py     (demonstration distribuee locale)")
    print("  python scripts/run_sequential.py     (reference sequentielle)")


if __name__ == "__main__":
    main()
