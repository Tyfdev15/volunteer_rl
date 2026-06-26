"""
Detection des caracteristiques de l'appareil volontaire
=======================================================
Sert a (1) annoncer le type d'appareil au serveur et (2) estimer sa PUISSANCE,
qui determine combien de sous-taches lui sont confiees par requete.
"""

import os
import platform
import multiprocessing


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


def estimate_power(info):
    """
    Puissance = nombre de sous-taches accordees par requete. Heuristique simple
    fondee sur le type d'appareil et le nombre de coeurs (un PC traite plusieurs
    sous-taches par aller-retour, un smartphone une seule).
    """
    cpu = info.get("cpu", 1)
    if info.get("os") == "android":
        return 1
    return max(1, min(4, cpu // 2))
