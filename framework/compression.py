"""
Transport et compression des gradients
======================================
Pour alleger les echanges entre volontaires et serveur (essentiel quand les
volontaires sont des smartphones sur reseau mobile), on ne transmet PAS les poids
mais les GRADIENTS, et on les compresse de deux facons combinables :

  1. Quantification  : float32 -> float16  (x2 de reduction, perte negligeable).
  2. Sparsification top-k : on ne transmet que les k% de gradients de plus grande
     amplitude (les autres sont consideres nuls). Compression supplementaire
     pouvant atteindre x10 a x100 selon k.

Le tout est ensuite compresse (zlib) puis encode en base64 pour le transport HTTP.
On expose aussi la taille des charges utiles pour MESURER le gain de bande passante.
"""

import json
import zlib
import base64
import numpy as np


def _dtype_np(name):
    return np.float16 if name == "fp16" else np.float32


def encode_vector(vec, dtype="fp16", topk=1.0):
    """
    Encode un vecteur (gradient) -> (chaine base64, taille_octets_utiles).
    `topk` dans ]0,1] : fraction des coefficients de plus grande amplitude transmis.
    """
    vec = np.asarray(vec, dtype=np.float32)
    n = vec.size
    npdt = _dtype_np(dtype)

    if topk >= 1.0:
        header = {"mode": "dense", "dtype": dtype, "n": int(n)}
        blob = vec.astype(npdt).tobytes()
    else:
        k = max(1, int(n * topk))
        idx = np.argpartition(np.abs(vec), n - k)[n - k:]   # k plus grandes amplitudes
        idx.sort()
        header = {"mode": "sparse", "dtype": dtype, "n": int(n), "k": int(k)}
        blob = idx.astype("<u4").tobytes() + vec[idx].astype(npdt).tobytes()

    hjson = json.dumps(header).encode("utf-8")
    raw = len(hjson).to_bytes(4, "little") + hjson + blob
    comp = zlib.compress(raw, level=6)
    return base64.b64encode(comp).decode("ascii"), len(comp)


def decode_vector(payload):
    """Decode une chaine base64 -> vecteur dense numpy (float32) reconstruit."""
    comp = base64.b64decode(payload.encode("ascii"))
    raw = zlib.decompress(comp)
    hlen = int.from_bytes(raw[:4], "little")
    header = json.loads(raw[4:4 + hlen].decode("utf-8"))
    blob = raw[4 + hlen:]
    npdt = _dtype_np(header["dtype"])
    n = header["n"]

    if header["mode"] == "dense":
        return np.frombuffer(blob, dtype=npdt).astype(np.float32).copy()

    k = header["k"]
    idx_bytes = 4 * k
    idx = np.frombuffer(blob[:idx_bytes], dtype="<u4")
    vals = np.frombuffer(blob[idx_bytes:], dtype=npdt).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    out[idx] = vals
    return out


def raw_size_bytes(n_params, dtype="fp32"):
    """Taille brute (sans compression) d'un vecteur de poids/gradients, en octets."""
    return n_params * (2 if dtype == "fp16" else 4)
