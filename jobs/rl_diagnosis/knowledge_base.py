"""
VolunteerRL Framework
----------------------------------------------

Distributed Reinforcement Learning
for Medical Diagnosis using Volunteer Computing

Developed by:
Tankeu Frederick

Department of ICT (University of Ebolowa)
Department of Physics (University of Yaoundé I)

2026
"""

"""
Base de connaissances medicale (MedlinePlus)
============================================
Charge le fichier CSV de reference (chaque maladie -> liste de symptomes) et
genere des patients synthetiques selon le procede :
  * pour un patient d'une maladie donnee, on tire un sous-ensemble de ses
    symptomes (loi de Poisson, borne) ;
  * une partie est "connue d'emblee" (presente dans l'etat initial), le reste
    doit etre DECOUVERT en interrogeant le patient (action de la politique).
Les symptomes hors de cet ensemble sont absents.
"""

import csv
import ast
import numpy as np


class KnowledgeBase:
    def __init__(self, diseases, n_symptoms):
        # diseases : liste de listes d'indices de symptomes (un element par maladie)
        self.diseases = diseases
        self.n_diseases = len(diseases)
        self.n_symptoms = n_symptoms

    @classmethod
    def from_csv(cls, path):
        diseases, mx = [], 0
        with open(path) as f:
            r = csv.reader(f)
            next(r)  # entete
            for row in r:
                if not row or not row[1].strip():
                    continue
                syms = [int(s) for s in ast.literal_eval(row[1])]
                diseases.append(syms)
                mx = max(mx, max(syms))
        return cls(diseases, mx + 1)

    def sample_patient(self, rng):
        """
        Retourne (maladie, present[set], connus_initiaux[list]).
        `present` = tous les symptomes reellement presents chez le patient.
        `connus_initiaux` = ceux reveles dans l'etat de depart.
        """
        d = int(rng.integers(self.n_diseases))
        glob = self.diseases[d]
        L = len(glob)
        # nombre de symptomes presents : Poisson(8) borne a [L/2, L]
        n = int(rng.poisson(8))
        n = max(L // 2, min(n, L))
        n = max(1, n)
        chosen = list(rng.permutation(glob)[:n])
        # nombre de symptomes connus d'emblee : Poisson(2)+1
        n_known = int(rng.poisson(2)) + 1
        n_known = min(n_known, max(1, n - 1)) if n > 1 else 1
        known = chosen[:n_known]
        return d, set(int(x) for x in chosen), [int(x) for x in known]
