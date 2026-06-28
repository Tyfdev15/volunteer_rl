"""
Client volontaire
=================
Tourne sur un smartphone (Termux/Pydroid), un PC Linux ou Windows. Boucle :
  1. telecharge les parametres globaux theta depuis le serveur ;
  2. recoit un lot de sous-taches (proportionnel a sa puissance) ;
  3. pour chaque sous-tache, calcule un GRADIENT et le COMPRESSE ;
  4. renvoie les gradients au serveur.
Robustesse : en cas d'erreur reseau, on reessaie ; les sous-taches perdues sont
reattribuees par le serveur (le volontaire n'a rien de special a faire).
"""

import time
import requests
import numpy as np

from config import DEFAULT
from framework.compression import encode_vector, decode_vector
from jobs.rl_diagnosis.job import RLDiagnosisJob
from client.device_info import get_device_info, estimate_power, benchmark_2s


class VolunteerClient:
    def __init__(self, server_url, device_label=None, power=None,
                 slowdown=0.0, max_iters=100000, cfg=DEFAULT):
        self.server = server_url.rstrip("/")
        self.cfg = cfg
        self.slowdown = slowdown
        self.max_iters = max_iters
        self.info = get_device_info(device_label)

        print("Benchmark local de 2 secondes en cours...")
        benchmark_score = benchmark_2s(2.0)
        self.info["benchmark_score"] = benchmark_score

        self.power = power if power is not None else estimate_power(self.info)

        print(f"Benchmark terminé : score={benchmark_score}, puissance={self.power}")
        self.client_id = f"{self.info['device']}-{int(time.time()*1000) % 100000}"
        self.job = None
        self.tcfg = cfg.transport

    def _get_job(self):
        for _ in range(30):
            try:
                r = requests.get(f"{self.server}/kb", timeout=10).json()
                self.job = RLDiagnosisJob.from_kb_spec(r["kb"], self.cfg)
                self.tcfg.dtype = r["transport"]["dtype"]
                self.tcfg.topk = r["transport"]["topk"]
                return
            except requests.RequestException:
                time.sleep(1.0)
        raise RuntimeError("serveur injoignable")
    
    def wait_for_server(self):
        """
        Attend que le serveur autorise le démarrage.
        """

        while True:
            try:
                r = requests.get(
                    f"{self.server}/training_status",
                    timeout=10
                ).json()

                if r["started"]:
                    print("\n>>> Départ reçu du serveur.")
                    return

                print("En attente du signal du serveur...")

            except requests.RequestException:
                pass

            time.sleep(2)    
            
    def run(self, verbose=True):
        self._get_job()
        print("\nConnexion réussie.")
        print("Le volontaire attend le signal de départ...")
        self.wait_for_server()

        if verbose:
            print(f"[volontaire {self.client_id}] {self.info['os']} "
                  f"cpu={self.info['cpu']} puissance={self.power} -> {self.server}")
        it = 0
        while it < self.max_iters:
            it += 1
            try:
                request_start = time.time()

                response = requests.post(f"{self.server}/request_work", timeout=15, json={
                    "client_id": self.client_id,
                    "info": self.info,
                    "power": self.power
                })

                request_work_seconds = time.time() - request_start
                resp = response.json()

            except requests.RequestException:
                time.sleep(0.5)
                continue

            if resp.get("finished"):
                if verbose:
                    print(f"[volontaire {self.client_id}] calcul termine, arret.")
                return
            tasks = resp.get("tasks", [])
            if not tasks:
                time.sleep(0.2); continue

            theta = decode_vector(resp["params"])
            version = resp["params_version"]
            results = []
            for task in tasks:
                task_start = time.time()

                grad, n_samples, lm = self.job.compute_gradient(theta, task)

                task_duration = time.time() - task_start

                payload, _ = encode_vector(grad, dtype=self.tcfg.dtype, topk=self.tcfg.topk)

                lm["duration_seconds"] = task_duration
                lm["request_work_seconds"] = request_work_seconds / max(1, len(tasks))
                lm["client_device"] = self.info.get("device")
                lm["client_os"] = self.info.get("os")
                lm["client_cpu"] = self.info.get("cpu")
                lm["client_ram_gb"] = self.info.get("ram_gb")

                results.append({
                    "task_id": task["task_id"],
                    "grad": payload,
                    "params_version": version,
                    "n_samples": n_samples,
                    "local_metrics": lm
                })
                if self.slowdown:
                    time.sleep(self.slowdown)
            try:
                report_start = time.time()

                requests.post(
                    f"{self.server}/report",
                    timeout=15,
                    json={"client_id": self.client_id, "results": results}
                )

                report_seconds = time.time() - report_start

                requests.post(
                    f"{self.server}/client_comm_metrics",
                    timeout=10,
                    json={
                        "client_id": self.client_id,
                        "task_ids": [r["task_id"] for r in results],
                        "report_seconds": report_seconds,
                        "tasks_count": len(results)
                    }
                )

                if verbose:
                    print(
                        f"[communication] request_work={request_work_seconds:.4f}s "
                        f"report={report_seconds:.4f}s "
                        f"tasks={len(results)}"
                    )

            except requests.RequestException:
                pass
        if verbose:
            print(f"[volontaire {self.client_id}] limite d'iterations atteinte.")