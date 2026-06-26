"""
Agent DQN : collecte d'experience + calcul du gradient
======================================================
Cote volontaire. A partir des parametres globaux theta (figes le temps de la
sous-tache), l'agent :
  1. joue `n_episodes` dialogues de diagnostic (politique epsilon-gloutonne) ;
  2. en deduit un LOT de transitions ;
  3. calcule un unique gradient (politique Q par equation de Bellman en
     semi-gradient + classifieur par entropie croisee).
L'optimiseur n'est PAS ici : le volontaire ne renvoie qu'un gradient brut.
"""

import numpy as np
from .nn import DiagnosisModel


class DQNAgent:
    def __init__(self, kb, env_cfg, model_cfg, seed=0):
        self.kb = kb
        self.env_cfg = env_cfg
        self.model_cfg = model_cfg
        self.model = DiagnosisModel(kb.n_symptoms, kb.n_diseases, model_cfg, seed)

    def load_params(self, flat):
        self.model.set_flat(flat)

    # ----- collecte d'un dialogue ----- #
    def _run_episode(self, env, rng, epsilon):
        s = env.reset(rng)
        trans, clf_samples = [], []
        done = False
        correct_final = 0
        while not done:
            pred, entropy = self.model.diagnose(s)
            # arret par entropie (classifieur sur de lui) apres un minimum de tours
            if entropy < self.env_cfg.entropy_stop and env.turn >= self.env_cfg.min_turns_before_stop:
                correct_final = int(pred == env.true_disease)
                break
            # choix de l'action (symptome a interroger), epsilon-glouton, masque les deja interroges
            q = self.model.q_values(s).copy()
            for a in env.queried:
                q[a] = -1e9
            if rng.random() < epsilon:
                cand = [a for a in range(env.n) if a not in env.queried]
                action = int(rng.choice(cand)) if cand else int(np.argmax(q))
            else:
                action = int(np.argmax(q))
            s2, r, done = env.step(action, pred)
            trans.append((s.copy(), action, r, s2.copy(), done))
            clf_samples.append((s2.copy(), env.true_disease))
            s = s2
            if done:
                pred2, _ = self.model.diagnose(s)
                correct_final = int(pred2 == env.true_disease)
        return trans, clf_samples, correct_final, env.turn

    def collect_batch(self, env, rng, n_episodes, epsilon):
        T, C, correct, turns = [], [], 0, 0
        for _ in range(n_episodes):
            tr, cs, ok, t = self._run_episode(env, rng, epsilon)
            T.extend(tr); C.extend(cs); correct += ok; turns += t
        return T, C, correct / max(1, n_episodes), turns / max(1, n_episodes)

    # ----- gradient sur le lot collecte ----- #
    def gradient(self, transitions, clf_samples):
        gamma = self.model_cfg.gamma
        m = self.model

        # --- gradient de la politique Q (DQN, semi-gradient, cible figee a theta) ---
        if transitions:
            S = np.array([t[0] for t in transitions], dtype=np.float32)
            A = np.array([t[1] for t in transitions], dtype=np.int64)
            R = np.array([t[2] for t in transitions], dtype=np.float32)
            S2 = np.array([t[3] for t in transitions], dtype=np.float32)
            D = np.array([t[4] for t in transitions], dtype=np.float32)
            B = len(transitions)
            Qs, cache = m.q.forward(S)
            Qs2, _ = m.q.forward(S2)
            target = R + gamma * Qs2.max(axis=1) * (1.0 - D)
            d_out = np.zeros_like(Qs)
            d_out[np.arange(B), A] = (Qs[np.arange(B), A] - target) / B
            gq = m.q.backward(cache, d_out)
        else:
            gq = np.zeros(m.q.n_params(), dtype=np.float32)

        # --- gradient du classifieur (entropie croisee softmax) ---
        if clf_samples:
            Sc = np.array([c[0] for c in clf_samples], dtype=np.float32)
            Y = np.array([c[1] for c in clf_samples], dtype=np.int64)
            M = len(clf_samples)
            logits, cache = m.clf.forward(Sc)
            logits = logits - logits.max(axis=1, keepdims=True)
            P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
            d_out = P.copy()
            d_out[np.arange(M), Y] -= 1.0
            d_out /= M
            gc = m.clf.backward(cache, d_out)
        else:
            gc = np.zeros(m.clf.n_params(), dtype=np.float32)

        return np.concatenate([gq, gc]).astype(np.float32)
