"""
Ordonnanceur facon BOINC
========================
Distribue les sous-taches aux volontaires et assure la robustesse :

  * Distribution PROPORTIONNELLE A LA PUISSANCE : un volontaire puissant reçoit
    plusieurs sous-taches par requete (moins d'allers-retours), un smartphone une
    seule. La puissance est declaree par le volontaire (auto-benchmark).
  * TOLERANCE AUX PANNES : une sous-tache non rendue avant un delai est
    automatiquement REATTRIBUEE a un autre volontaire (le travail global ne
    s'arrete jamais).
  * Suivi de FIABILITE par volontaire (taux de sous-taches menees a terme).
"""

import time
import threading
from collections import deque


class ClientStat:
    def __init__(self, info, power):
        self.info = info
        self.power = power          # nb de sous-taches accordees par requete
        self.completed = 0
        self.timeouts = 0
        self.assigned = 0
        self.last_seen = time.time()
        self.total_compute_seconds = 0.0
        self.last_task_seconds = 0.0

    @property
    def reliability(self):
        tot = self.completed + self.timeouts
        return self.completed / tot if tot else 1.0


class Scheduler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.clients = {}
        self.pending = deque()
        self.inflight = {}          # task_id -> (client_id, deadline)
        self.tasks = {}             # task_id -> task
        self.done = 0
        self.total = 0
        self.reassigned = 0
        self.lock = threading.Lock()

    # ----- volontaires ----- #
    def register(self, client_id, info, power):
        with self.lock:
            if client_id not in self.clients:
                self.clients[client_id] = ClientStat(info, max(1, int(power)))
            self.clients[client_id].last_seen = time.time()

    # ----- chargement d'une epoque ----- #
    def load_epoch(self, tasks):
        with self.lock:
            self.pending.clear()
            self.inflight.clear()
            self.tasks = {t["task_id"]: t for t in tasks}
            self.pending.extend(t["task_id"] for t in tasks)
            self.done = 0
            self.total = len(tasks)
            self.reassigned = 0

    # ----- tolerance aux pannes : reattribution des sous-taches expirees ----- #
    def _reclaim_expired(self):
        now = time.time()
        expired = [tid for tid, (cid, dl) in self.inflight.items() if now > dl]
        for tid in expired:
            cid, _ = self.inflight.pop(tid)
            if cid in self.clients:
                self.clients[cid].timeouts += 1
            self.pending.appendleft(tid)        # remis en tete de file
            self.reassigned += 1

    # ----- assignation (par lot proportionnel a la puissance) ----- #
    def assign(self, client_id):
        with self.lock:
            self._reclaim_expired()
            if client_id not in self.clients:
                return []
            n = self.clients[client_id].power
            batch = []
            deadline = time.time() + self.cfg.train.task_timeout
            while self.pending and len(batch) < n:
                tid = self.pending.popleft()
                self.inflight[tid] = (client_id, deadline)
                self.clients[client_id].assigned += 1
                batch.append(self.tasks[tid])
            self.clients[client_id].last_seen = time.time()
            return batch

    # ----- compte rendu d'une sous-tache ----- #
    def report(self, client_id, task_id, duration_seconds=0.0):
        with self.lock:
            if task_id in self.inflight:
                del self.inflight[task_id]
                self.done += 1

                if client_id in self.clients:
                    c = self.clients[client_id]
                    c.completed += 1
                    c.last_task_seconds = float(duration_seconds or 0.0)
                    c.total_compute_seconds += float(duration_seconds or 0.0)

                return True

            return False # sous-tache deja reattribuee/terminee (rendu tardif ignore)

    # ----- etat ----- #
    def progress(self):
        with self.lock:
            return self.done, self.total

    def stats(self):
        with self.lock:
            pending_count = len(self.pending)
            inflight_count = len(self.inflight)
            done_count = self.done

            return {
                "tasks": {
                    "total": self.total,
                    "pending": pending_count,
                    "inflight": inflight_count,
                    "done": done_count,
                    "reassigned": self.reassigned,
                },
                "clients": {
                    cid: {
                        "info": c.info,
                        "completed": c.completed,
                        "timeouts": c.timeouts,
                        "assigned": c.assigned,
                        "reliability": round(c.reliability, 3),
                        "power": c.power,
                        "last_seen": c.last_seen,
                        "avg_task_seconds": round(c.total_compute_seconds / c.completed, 3) if c.completed else 0,
                        "last_task_seconds": round(c.last_task_seconds, 3),
                    }
                    for cid, c in self.clients.items()
                },
                "reassigned": self.reassigned,
            }    
