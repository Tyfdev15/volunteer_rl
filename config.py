"""
Configuration centrale
=======================
Tous les hyperparametres du systeme, regroupes en dataclasses. Modifier ici
suffit a re-parametrer une simulation (aucune valeur magique dispersee ailleurs).
"""

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
#  Donnees / base de connaissances (MedlinePlus)
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    csv_file: str = "data/MedlinePlus20.csv"  # base de reference (20 maladies, 110 symptomes)
    patients_per_disease: int = 120           # patients synthetiques generes par maladie
    train_fraction: float = 0.85              # part train / test
    seed: int = 42


# --------------------------------------------------------------------------- #
#  Environnement de dialogue patient (article EAAI)
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    max_turns: int = 20            # T : nb maximal de questions posees
    none_val: float = 0.0         # symptome inconnu
    pres_val: float = 1.0         # symptome present
    abs_val: float = -1.0         # symptome absent (interroge et nie)
    reward_step: float = -1.0          # cout d'une question
    reward_relevant_finding: float = 1.7  # interroger un symptome pertinent
    reward_other_finding: float = 0.7     # interroger un symptome non pertinent
    reward_correct: float = 1.0        # bonus si le diagnostic courant est correct
    reward_wrong_end: float = -1.0     # malus si fin atteinte sans bon diagnostic
    entropy_stop: float = 0.5          # seuil d'entropie pour stopper (seuil par maladie, init)
    min_turns_before_stop: int = 2


# --------------------------------------------------------------------------- #
#  Reseaux de neurones (NumPy pur, executables sur smartphone)
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    hidden: int = 128
    gamma: float = 0.95           # facteur d'actualisation (Lambda dans la reference)
    lr: float = 1e-3              # pas d'apprentissage (optimiseur Adam, cote serveur)
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.92


# --------------------------------------------------------------------------- #
#  Transport des gradients (allegement des echanges volontaire <-> serveur)
# --------------------------------------------------------------------------- #
@dataclass
class TransportConfig:
    dtype: str = "fp16"      # "fp32" | "fp16" : quantification des gradients
    topk: float = 1.0        # 1.0 = dense ; <1.0 = ne transmet que les k% plus grands gradients
    # exemple : dtype="fp16", topk=0.25  ->  ~ x8 de reduction de bande passante


# --------------------------------------------------------------------------- #
#  Entrainement distribue (le "BOINC")
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    episodes_per_task: int = 1000     # episodes collectes par sous-tache -> un lot de gradient ini48 900
    n_tasks_per_epoch: int = 100     # sous-taches (= pas de gradient) par epoque ini24 60
    max_epochs: int = 50            #ini40
    target_accuracy: float = 0.985   # critere d'arret (convergence) ini 0.9
    completion_fraction: float = 0.85   # une epoque se finalise des 85% des taches rendues ini 0.85
    task_timeout: float = 300.0           # s avant reattribution (tolerance aux pannes) ini 8.0
    staleness_max: int = 60            # gradients trop perimes (async) sont rejetes ini 12 30


# --------------------------------------------------------------------------- #
#  Serveur de coordination
# --------------------------------------------------------------------------- #
@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    validation_patients: int = 1500  #ini 400


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


DEFAULT = Config()
