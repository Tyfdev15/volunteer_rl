"""
Serveur de parametres
=====================
Detient les parametres globaux theta et l'etat de l'optimiseur Adam. Recoit des
GRADIENTS compresses de la part des volontaires, les applique (SGD asynchrone
avec gestion de la peremption / staleness), et sert theta aux volontaires.

Comptabilise aussi la bande passante (brute vs compressee) pour DEMONTRER le gain
du transport par gradients compresses.
"""

import threading
import numpy as np

from .compression import encode_vector, decode_vector, raw_size_bytes


class ParameterServer:
    def __init__(self, job, transport_cfg):
        self.job = job
        self.tcfg = transport_cfg
        self.theta = job.init_params().astype(np.float32)
        self.opt = job.init_opt_state()
        self.version = 0
        self.lock = threading.Lock()
        # comptabilite reseau
        self.bytes_up = 0          # volontaire -> serveur (gradients compresses)
        self.bytes_down = 0        # serveur -> volontaire (parametres compresses)
        self.raw_up_fp32 = 0       # ce qu'auraient coute des gradients/poids bruts fp32
        self.n_grads_applied = 0
        self.n_grads_dropped = 0

    # ----- service des parametres (telechargement par le volontaire) ----- #
    def params_payload(self):
        with self.lock:
            payload, nbytes = encode_vector(self.theta, dtype=self.tcfg.dtype, topk=1.0)
            v = self.version
        self.bytes_down += nbytes
        return payload, v

    def get_theta(self):
        with self.lock:
            return self.theta.copy(), self.version

    # ----- assimilation d'un gradient (televersement par le volontaire) ----- #
    def assimilate(self, grad_payload, params_version, n_samples, lr, staleness_max):
        grad = decode_vector(grad_payload)
        comp_bytes = len(grad_payload)  # taille reçue (approx base64) -> comptee ci-dessous proprement
        # taille reellement transferee (octets compresses) : recalculee par l'appelant idealement ;
        # ici on estime via la longueur du payload base64 -> octets
        comp_bytes = int(len(grad_payload) * 3 / 4)
        raw = raw_size_bytes(self.job.n_params(), "fp32")

        with self.lock:
            staleness = self.version - params_version
            if staleness > staleness_max:
                # peremption non bornee : on rejette (le travail sera refait sur theta a jour)
                self.n_grads_dropped += 1
                self.bytes_up += comp_bytes
                self.raw_up_fp32 += raw
                return False, staleness
            # peremption BORNEE (style Hogwild!/DistBelief) : on applique a pleine vitesse,
            # avec une legere attenuation seulement pour les retards importants
            lr_eff = lr / (1.0 + 0.25 * max(0, staleness))
            self.theta, self.opt = self.job.apply_gradient(self.theta, grad, self.opt, lr_eff)
            self.version += 1
            self.n_grads_applied += 1
            self.bytes_up += comp_bytes
            self.raw_up_fp32 += raw
        return True, staleness

    # ----- statistiques reseau ----- #
    def bandwidth_stats(self):
        total = self.bytes_up + self.bytes_down
        raw_equiv = self.raw_up_fp32 * 2  # aller (poids) + retour (gradients) en fp32 brut
        return {
            "octets_televerses_gradients": self.bytes_up,
            "octets_telecharges_parametres": self.bytes_down,
            "octets_total_compresse": total,
            "octets_equivalent_brut_fp32": raw_equiv,
            "facteur_reduction": (raw_equiv / total) if total else 0.0,
            "gradients_appliques": self.n_grads_applied,
            "gradients_perimes_rejetes": self.n_grads_dropped,
        }
