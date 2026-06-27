"""
Coordinateur
============
Chef d'orchestre du calcul distribue. A chaque epoque :
  1. genere les sous-taches et les confie a l'ordonnanceur ;
  2. distribue le travail aux volontaires (avec les parametres globaux) ;
  3. assimile les gradients reçus dans le serveur de parametres (Adam) ;
  4. des que `completion_fraction` des sous-taches sont rendues, FINALISE l'epoque
     sans attendre les retardataires (asynchronisme + tolerance aux pannes),
     evalue le modele global, enregistre les metriques, et decide de continuer
     ou de s'ARRETER (convergence atteinte -> le serveur signale la fin).
"""

import math
import time
import threading

from .parameter_server import ParameterServer
from .scheduler import Scheduler
from .work_generator import WorkGenerator


class Coordinator:
    def __init__(self, job, cfg):
        self.job = job
        self.cfg = cfg
        self.ps = ParameterServer(job, cfg.transport)
        self.sched = Scheduler(cfg)
        self.wg = WorkGenerator(job, cfg)
        self.epoch = 0
        self.epsilon = cfg.model.epsilon_start
        self.history = []
        self.finished = False
        self.start_time = None
        self.end_time = None
        self.epoch_start = None
        self._finalizing = False
        self.lock = threading.Lock()
        # metriques locales agregees de l'epoque courante
        self._epoch_local_acc = []
        self._epoch_local_turns = []
        self.events = []

    def add_event(self, event_type, message):
        """
        Ajoute un événement visible dans le journal temps réel du dashboard.
        """
        self.events.append({
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "message": message
        })

        if len(self.events) > 200:
            self.events = self.events[-200:]

    # ----- demarrage ----- #
    def start(self):
        self.start_time = time.time()
        self._begin_epoch(1)

    def _begin_epoch(self, e):
        self.epoch = e
        self.epoch_start = time.time()
        self._finalizing = False
        self._epoch_local_acc = []
        self._epoch_local_turns = []
        tasks = self.wg.create_epoch(e, self.epsilon)
        self.sched.load_epoch(tasks)

    # ----- API appelee par les volontaires ----- #
    def request_work(self, client_id, info, power):
        self.sched.register(client_id, info, power)
        if self.finished:
            return {"tasks": [], "finished": True}
        batch = self.sched.assign(client_id)
        if batch:
            device = info.get("device", client_id)
            for task in batch:
                self.add_event(
                    "SEND_TASK",
                    f"{device} reçoit {task['task_id']}"
                )
        else:
            self.add_event(
                "NO_TASK",
                f"{info.get('device', client_id)} attend : aucune sous-tâche disponible"
            )
        payload, version = self.ps.params_payload()
        return {"tasks": batch, "params": payload, "params_version": version,
                "finished": False}

    def report_gradients(self, client_id, results):
        for r in results:
            self.ps.assimilate(
                r["grad"], r["params_version"], r.get("n_samples", 1),
                self.cfg.model.lr, self.cfg.train.staleness_max)
            lm = r.get("local_metrics") or {}
            self.sched.report(
                client_id,
                r["task_id"],
                lm.get("duration_seconds", 0.0),
                lm.get("request_work_seconds", 0.0)
            )

            device = lm.get("client_device", client_id)

            self.add_event(
                "TASK_DONE",
                f"{device} termine {r['task_id']} en {lm.get('duration_seconds', 0):.2f}s"
            )

            self.add_event(
                "GRADIENT",
                f"Gradient appliqué pour {r['task_id']}"
            )
            #lm = r.get("local_metrics") or {}
            if "local_accuracy" in lm:
                self._epoch_local_acc.append(lm["local_accuracy"])
            if "avg_turns" in lm:
                self._epoch_local_turns.append(lm["avg_turns"])
        self._maybe_finalize()


    def record_client_comm_metrics(self, client_id, task_ids, report_seconds, tasks_count):
        self.sched.add_report_communication(client_id, report_seconds, tasks_count)

        self.add_event(
            "COMMUNICATION",
            f"{client_id} POST /report = {float(report_seconds or 0):.4f}s pour {tasks_count} tâche(s)"
        )
    # ----- finalisation d'epoque ----- #
    def _maybe_finalize(self):
        done, total = self.sched.progress()
        threshold = math.ceil(self.cfg.train.completion_fraction * total)
        with self.lock:
            if self._finalizing or self.finished or total == 0 or done < threshold:
                return
            self._finalizing = True

        theta, version = self.ps.get_theta()
        metrics = self.job.evaluate(theta, self.cfg.server.validation_patients)
        bw = self.ps.bandwidth_stats()
        rec = {
            "epoch": self.epoch,
            "accuracy": metrics["accuracy"],
            "top3": metrics.get("top3"),
            "macro_f1": metrics.get("macro_f1"),
            "avg_turns": metrics.get("avg_turns"),
            "elapsed": time.time() - self.start_time,
            "epoch_time": time.time() - self.epoch_start,
            "version": version,
            "reassigned": self.sched.stats()["reassigned"],
            "grads_applied": bw["gradients_appliques"],
            "grads_dropped": bw["gradients_perimes_rejetes"],
        }
        self.history.append(rec)

        #if metrics["accuracy"] >= self.cfg.train.target_accuracy or \
         #       self.epoch >= self.cfg.train.max_epochs:
         #   self.finished = True

         # Pour les expériences comparatives, on ne s'arrête pas dès que la précision cible est atteinte.
         # Cela permet de comparer 1 volontaire vs plusieurs volontaires sur la même quantité de travail.
        if self.epoch >= self.cfg.train.max_epochs:
            self.finished = True
            self.end_time = time.time()
        else:
            self.epsilon = max(self.cfg.model.epsilon_end,
                               self.epsilon * self.cfg.model.epsilon_decay)
            self._begin_epoch(self.epoch + 1)

    # ----- etat (tableau de bord) ----- #
    def status(self):
        done, total = self.sched.progress()
        last = self.history[-1] if self.history else {}
        sched_stats = self.sched.stats()
        elapsed = ((self.end_time or time.time()) - self.start_time) if self.start_time else 0
        total_compute = sched_stats.get("total_compute_seconds", 0)
        workers = max(1, sched_stats.get("active_workers", 1))

        speedup = total_compute / elapsed if elapsed > 0 else 0
        efficiency = speedup / workers if workers > 0 else 0

        bandwidth = self.ps.bandwidth_stats()
        gradients = bandwidth.get("gradients_appliques", 0)
        throughput_gradients = gradients / elapsed if elapsed > 0 else 0        
        return {
            "job": self.job.name,
            "epoch": self.epoch,
            "max_epochs": self.cfg.train.max_epochs,
            "finished": self.finished,
            "progress": {"done": done, "total": total},
            "target_accuracy": self.cfg.train.target_accuracy,
            "last_accuracy": last.get("accuracy"),
            "last_top3": last.get("top3"),
            "last_f1": last.get("macro_f1"),
            "last_turns": last.get("avg_turns"),
            "epsilon": round(self.epsilon, 3),
            "elapsed": elapsed,
            "bandwidth": bandwidth,
            "scheduler": sched_stats,
            "distributed_metrics": {
                "total_compute_seconds": round(total_compute, 3),
                "speedup": round(speedup, 3),
                "efficiency": round(efficiency, 3),
                "throughput_gradients_per_second": round(throughput_gradients, 4),
            },
            "events": self.events[-100:],
            "history": self.history,
        }
