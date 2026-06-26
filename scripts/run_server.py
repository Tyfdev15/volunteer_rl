#!/usr/bin/env python3
"""
Lancement du serveur de coordination
====================================
  python scripts/run_server.py                 # ecoute sur 127.0.0.1:5000
  python scripts/run_server.py --host 0.0.0.0  # accessible aux volontaires du reseau

Affiche au demarrage l'adresse locale a communiquer aux volontaires, et signale
la fin du calcul.
"""

import os
import sys
import socket
import argparse
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT
from jobs.rl_diagnosis.knowledge_base import KnowledgeBase
from jobs.rl_diagnosis.job import RLDiagnosisJob
from server.app import create_app


def local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def banner(host, port, job):
    ip = local_ip() if host == "0.0.0.0" else host
    line = "=" * 64
    print(line)
    print("  SERVEUR DE CALCUL VOLONTAIRE  --  DEMARRE")
    print(line)
    print(f"  Algorithme         : {job.name}")
    print(f"  Parametres modele  : {job.n_params():,}")
    print(f"  Adresse locale     : http://{ip}:{port}")
    print(f"  Tableau de bord    : http://{ip}:{port}/  (evolution en temps reel)")
    print(f"  Transport          : gradients {DEFAULT.transport.dtype} "
          f"top-k={DEFAULT.transport.topk}")
    print(line)
    print("  Commande volontaire :")
    print(f"    python scripts/run_volunteer.py --server http://{ip}:{port}")
    print(line, flush=True)


def watch_completion(app):
    while not app.coord.finished:
        time.sleep(0.5)

    h = app.coord.history[-1] if app.coord.history else {}

    elapsed = (
        (app.coord.end_time or time.time()) - app.coord.start_time
    ) if app.coord.start_time else 0

    line = "=" * 64
    print("\n" + line)
    print("  CALCUL TERMINE  --  le serveur a atteint le critere d'arret")
    print(line)
    print(f"  Epoques            : {h.get('epoch')}")
    print(f"  Precision finale   : {h.get('accuracy', 0):.3f}")
    print(f"  Top-3 / F1 macro   : {h.get('top3', 0):.3f} / {h.get('macro_f1', 0):.3f}")
    print(f"  Tours moyens       : {h.get('avg_turns', 0):.1f}")
    print(f"  Temps total        : {elapsed:.1f} s")
    print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=DEFAULT.server.host)
    ap.add_argument("--port", type=int, default=DEFAULT.server.port)
    ap.add_argument("--csv", default=DEFAULT.data.csv_file)
    args = ap.parse_args()

    kb = KnowledgeBase.from_csv(args.csv)
    job = RLDiagnosisJob(kb, DEFAULT)
    app = create_app(job, DEFAULT)
    app.coord.start()

    banner(args.host, args.port, job)
    threading.Thread(target=watch_completion, args=(app,), daemon=True).start()

    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
