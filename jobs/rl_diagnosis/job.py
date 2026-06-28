"""
VolunteerRL Framework
----------------------------------------------

Distributed Reinforcement Learning
for Medical Diagnosis using Volunteer Computing

Developed by:
Tankeu Frederick

Department of ICT (University of Ebolowa)
Department of Physics (University of Yaoundé I)

2026
"""

"""
Job : apprentissage par renforcement pour le diagnostic de maladies
===================================================================
Implemente l'interface `TrainingJob` du framework. C'est le "plugin" branche sur
le BOINC. Pour un autre algorithme apres le memoire, on ecrira un autre fichier
de ce type sans toucher au reste.
"""

import numpy as np

from framework.job import TrainingJob
from .knowledge_base import KnowledgeBase
from .environment import DiagnosisEnv
from .agent import DQNAgent
from .nn import DiagnosisModel, adam_init, adam_step


class RLDiagnosisJob(TrainingJob):
    name = "RL-Diagnostic (DQN sur dialogue patient)"

    def __init__(self, kb, cfg):
        self.kb = kb
        self.cfg = cfg
        self._template = DiagnosisModel(kb.n_symptoms, kb.n_diseases, cfg.model)
        self._n = self._template.n_params()

    # --- specification serialisable de la base (pour transmettre au volontaire) --- #
    def kb_spec(self):
        return {"diseases": self.kb.diseases, "n_symptoms": self.kb.n_symptoms}

    @classmethod
    def from_kb_spec(cls, spec, cfg):
        kb = KnowledgeBase(spec["diseases"], spec["n_symptoms"])
        return cls(kb, cfg)

    # --- interface TrainingJob --- #
    def init_params(self):
        return DiagnosisModel(self.kb.n_symptoms, self.kb.n_diseases,
                              self.cfg.model, seed=0).get_flat()

    def n_params(self):
        return self._n

    def init_opt_state(self):
        return adam_init(self._n)

    def make_task(self, epoch, index, seed, epsilon):
        return {"epoch": epoch, "index": index, "seed": int(seed),
                "n_episodes": self.cfg.train.episodes_per_task,
                "epsilon": float(epsilon)}

    def compute_gradient(self, params_flat, task):
        rng = np.random.default_rng(task["seed"])
        agent = DQNAgent(self.kb, self.cfg.env, self.cfg.model)
        agent.load_params(np.asarray(params_flat, dtype=np.float32))
        env = DiagnosisEnv(self.kb, self.cfg.env)
        T, C, acc, turns = agent.collect_batch(env, rng, task["n_episodes"], task["epsilon"])
        grad = agent.gradient(T, C)
        return grad, len(T), {"local_accuracy": acc, "avg_turns": turns}

    def apply_gradient(self, params_flat, grad_flat, opt_state, lr):
        return adam_step(params_flat, grad_flat, opt_state, lr)

    # --- evaluation du modele global (politique gloutonne + arret par entropie) --- #
    def evaluate(self, params_flat, n):
        m = DiagnosisModel(self.kb.n_symptoms, self.kb.n_diseases, self.cfg.model)
        m.set_flat(np.asarray(params_flat, dtype=np.float32))
        env = DiagnosisEnv(self.kb, self.cfg.env)
        rng = np.random.default_rng(12345)
        D = self.kb.n_diseases
        conf = np.zeros((D, D), dtype=int)
        top3, turns_tot = 0, 0
        for _ in range(n):
            s = env.reset(rng)
            done = False
            while not done:
                pred, ent = m.diagnose(s)
                if ent < self.cfg.env.entropy_stop and env.turn >= self.cfg.env.min_turns_before_stop:
                    break
                q = m.q_values(s).copy()
                for a in env.queried:
                    q[a] = -1e9
                s, _, done = env.step(int(np.argmax(q)), pred)
            dist = m.disease_dist(s)
            pred = int(np.argmax(dist))
            conf[env.true_disease, pred] += 1
            top3 += int(env.true_disease in np.argsort(dist)[-3:])
            turns_tot += env.turn

        acc = float(np.trace(conf) / max(1, conf.sum()))
        f1s = []
        for d in range(D):
            tp = conf[d, d]; fp = conf[:, d].sum() - tp; fn = conf[d, :].sum() - tp
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        return {"accuracy": acc, "top3": top3 / max(1, n),
                "macro_f1": float(np.mean(f1s)), "avg_turns": turns_tot / max(1, n)}
