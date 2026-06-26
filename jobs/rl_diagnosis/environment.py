"""
Environnement de dialogue de diagnostic
=======================================
Reproduit le processus de l'article EAAI : un agent interroge le patient
symptome par symptome ; l'etat est le vecteur des symptomes
  inconnu = 0, present = +1, absent = -1.
A chaque tour l'agent (1) interroge un symptome via la politique, puis
(2) pose un diagnostic provisoire via le classifieur. Le dialogue s'arrete
quand le classifieur est suffisamment sur (faible entropie) ou au tour maximal.
"""

import numpy as np


class DiagnosisEnv:
    def __init__(self, kb, cfg):
        self.kb = kb
        self.cfg = cfg
        self.n = kb.n_symptoms

    def reset(self, rng):
        self.true_disease, self.present, known = self.kb.sample_patient(rng)
        self.state = np.full(self.n, self.cfg.none_val, dtype=np.float32)
        for s in known:
            self.state[s] = self.cfg.pres_val
        self.queried = set(known)
        self.turn = 0
        self.done = False
        return self.state.copy()

    def step(self, action, current_pred):
        """
        action       : indice du symptome interroge
        current_pred : diagnostic provisoire du classifieur (pour la recompense)
        Retourne (etat_suivant, recompense, termine).
        """
        c = self.cfg
        reward = c.reward_step
        if action not in self.queried:
            if action in self.present:
                self.state[action] = c.pres_val
                reward += c.reward_relevant_finding     # symptome present revele : tres informatif
            else:
                self.state[action] = c.abs_val
                reward += c.reward_other_finding        # symptome absent : informatif (exclusion)
            self.queried.add(action)
        # bonus si le diagnostic provisoire est correct
        if current_pred == self.true_disease:
            reward += c.reward_correct

        self.turn += 1
        if self.turn >= c.max_turns:
            self.done = True
            if current_pred != self.true_disease:
                reward += c.reward_wrong_end
        return self.state.copy(), float(reward), self.done
