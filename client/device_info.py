"""
Detection des caracteristiques de l'appareil volontaire
=======================================================
Sert a (1) annoncer le type d'appareil au serveur et (2) estimer sa PUISSANCE,
qui determine combien de sous-taches lui sont confiees par requete.
"""

import os
import platform
import multiprocessing
import time
import math


def is_android():
    return "ANDROID_ROOT" in os.environ or "ANDROID_DATA" in os.environ


def get_ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError, AttributeError):
        return None


def get_device_info(label=None):
    cpu = multiprocessing.cpu_count()
    osname = "android" if is_android() else platform.system().lower()
    return {
        "device": label or osname,
        "os": osname,
        "cpu": cpu,
        "ram_gb": get_ram_gb(),
        "python": platform.python_version(),
        "machine": platform.machine(),
    }

def benchmark_2s(duration=2.0):
    """
    Benchmark CPU court exécuté au lancement du volontaire.
    Il permet d'estimer automatiquement la puissance réelle de l'appareil.
    """
    start = time.time()
    ops = 0
    x = 0.0

    while time.time() - start < duration:
        for i in range(1000):
            x += math.sin(i) * math.cos(i)
        ops += 1000

    elapsed = time.time() - start
    return int(ops / elapsed)

def estimate_power(info):
    """
    Puissance calculée à partir du benchmark local.
    Cette valeur détermine combien de sous-tâches le volontaire reçoit par requête.
    """
    score = info.get("benchmark_score", 0)

    if score < 300_000:
        return 1
    if score < 700_000:
        return 2
    if score < 1_200_000:
        return 3
    return 4
