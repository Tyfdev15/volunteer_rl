"""
Reseaux de neurones en NumPy pur
================================
Aucune dependance lourde (ni TensorFlow ni PyTorch) : c'est le choix qui permet
a un volontaire de tourner sur un smartphone (Termux / Pydroid) sans GPU.

Particularite : on SEPARE le calcul du gradient (cote volontaire) de l'application
de l'optimiseur Adam (cote serveur). Un `MLP` sait donc :
  * forward / backward -> produire un gradient plat sur un lot ;
  * get_flat / set_flat -> lire/ecrire ses parametres sous forme de vecteur plat.
L'optimiseur Adam (adam_step) opere, lui, sur des vecteurs plats cote serveur.
"""

import numpy as np


def relu(x):
    return np.maximum(0.0, x)


class MLP:
    """Perceptron a une couche cachee (ReLU), initialisation He."""

    def __init__(self, n_in, n_hidden, n_out, seed=0):
        rng = np.random.default_rng(seed)
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.W1 = rng.standard_normal((n_in, n_hidden)).astype(np.float32) * np.sqrt(2.0 / n_in)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = rng.standard_normal((n_hidden, n_out)).astype(np.float32) * np.sqrt(2.0 / n_hidden)
        self.b2 = np.zeros(n_out, dtype=np.float32)

    # --- parametres sous forme de vecteur plat --- #
    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def get_flat(self):
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_flat(self, v):
        i = 0
        s = self.W1.size; self.W1 = v[i:i+s].reshape(self.W1.shape).astype(np.float32); i += s
        s = self.b1.size; self.b1 = v[i:i+s].astype(np.float32); i += s
        s = self.W2.size; self.W2 = v[i:i+s].reshape(self.W2.shape).astype(np.float32); i += s
        s = self.b2.size; self.b2 = v[i:i+s].astype(np.float32); i += s

    # --- passes avant / arriere --- #
    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return z2, (X, z1, a1)

    def backward(self, cache, d_out):
        """d_out = dL/dz2 (forme [B, n_out]). Retourne le gradient plat."""
        X, z1, a1 = cache
        gW2 = a1.T @ d_out
        gb2 = d_out.sum(axis=0)
        da1 = d_out @ self.W2.T
        dz1 = da1 * (z1 > 0)
        gW1 = X.T @ dz1
        gb1 = dz1.sum(axis=0)
        return np.concatenate([gW1.ravel(), gb1, gW2.ravel(), gb2]).astype(np.float32)


class DiagnosisModel:
    """
    Deux reseaux : une politique Q (choisit quel symptome interroger) et un
    classifieur (predit la maladie, avec entropie pour l'arret). Les parametres
    et gradients des deux reseaux sont concatenes en UN seul vecteur plat, afin
    que le transport et l'agregation soient de simples operations vectorielles.
    """

    def __init__(self, n_symptoms, n_diseases, cfg, seed=0):
        self.q = MLP(n_symptoms, cfg.hidden, n_symptoms, seed=seed)         # Q(s) sur les actions=symptomes
        self.clf = MLP(n_symptoms, cfg.hidden, n_diseases, seed=seed + 1)   # P(maladie | s)
        self._nq = self.q.n_params()

    def n_params(self):
        return self.q.n_params() + self.clf.n_params()

    def get_flat(self):
        return np.concatenate([self.q.get_flat(), self.clf.get_flat()])

    def set_flat(self, v):
        self.q.set_flat(v[:self._nq])
        self.clf.set_flat(v[self._nq:])

    # --- inferences --- #
    def q_values(self, state):
        out, _ = self.q.forward(state[None, :])
        return out[0]

    def disease_dist(self, state):
        logits, _ = self.clf.forward(state[None, :])
        logits = logits[0] - logits[0].max()
        e = np.exp(logits)
        return e / e.sum()

    def diagnose(self, state):
        p = self.disease_dist(state)
        entropy = float(-(p * np.log(p + 1e-12)).sum())
        return int(np.argmax(p)), entropy


# --------------------------------------------------------------------------- #
#  Optimiseur Adam sur vecteurs plats (cote serveur)
# --------------------------------------------------------------------------- #
def adam_init(n):
    return {"m": np.zeros(n, dtype=np.float32), "v": np.zeros(n, dtype=np.float32), "t": 0}


def adam_step(params, grad, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    """Un pas d'Adam. Modifie/retourne params et l'etat (m, v, t)."""
    state["t"] += 1
    t = state["t"]
    state["m"] = beta1 * state["m"] + (1 - beta1) * grad
    state["v"] = beta2 * state["v"] + (1 - beta2) * (grad * grad)
    mhat = state["m"] / (1 - beta1 ** t)
    vhat = state["v"] / (1 - beta2 ** t)
    params = params - lr * mhat / (np.sqrt(vhat) + eps)
    return params.astype(np.float32), state
