# Calcul volontaire pour la parallélisation de l'apprentissage par renforcement appliqué au diagnostic médical

Application Python de type **BOINC** (notre propre BOINC), conçue pour démontrer
qu'on peut exploiter la **puissance de calcul simultanée de smartphones, de PC
Linux et de PC Windows** — abondants mais sous-utilisés — pour réaliser de
l'**apprentissage par renforcement** appliqué à la **détection et au diagnostic
de maladies**, **sans super-calculateur** coûteux et énergivore.

Le système est **générique** : l'algorithme de RL pour le diagnostic n'est qu'un
*plugin*. Après le mémoire, d'autres algorithmes pourront tourner sur la même
infrastructure sans la modifier (voir « Extensibilité »).

---

## 1. Idée et architecture

Modèle de calcul : **SGD distribué avec serveur de paramètres**.

```
   Serveur (coordination)                         Volontaires (calcul)
   ┌───────────────────────────┐                  ┌────────────────────┐
   │ paramètres globaux θ       │  ── θ (poids) ─▶ │ smartphone Android │
   │ optimiseur Adam            │                  │ PC Linux           │
   │ ordonnanceur BOINC         │ ◀─ gradients ──  │ PC Windows         │
   │ tolérance aux pannes       │   (compressés)   └────────────────────┘
   │ tableau de bord temps réel │
   └───────────────────────────┘
```

* Le serveur détient les paramètres θ et l'optimiseur. Il découpe le travail en
  **sous-tâches**, les distribue **selon la puissance** de chaque volontaire,
  **réattribue** les sous-tâches non rendues (tolérance aux pannes), agrège les
  **gradients** reçus et applique un pas d'**Adam**.
* Chaque volontaire télécharge θ, calcule un **gradient** sur un lot local et le
  renvoie **compressé**. Il ne calcule qu'un gradient brut (pas d'état d'optimiseur).
* **On transmet des gradients, pas des poids**, et on les compresse (fp16 +
  sparsification top-k) pour **alléger les échanges** — essentiel pour des
  smartphones sur réseau mobile.

Pourquoi des gradients ? Le code C++ de référence agrège lui-même des gradients
(« agrégation implicite »). Les envoyer (au lieu des poids) permet de les
**compresser fortement** et garantit un modèle global **cohérent** (le nombre de
questions par patient décroît alors naturellement à mesure que la précision monte).

---

## 2. Arborescence (où placer chaque fichier)

```
volunteer_rl/
├── README.md
├── requirements.txt
├── config.py                  ← tous les hyperparamètres (un seul endroit)
│
├── framework/                 ← LE "BOINC" générique (agnostique à l'algorithme)
│   ├── __init__.py
│   ├── job.py                 ← interface TrainingJob (point d'extension)
│   ├── compression.py         ← (dé)compression des gradients fp16 + top-k
│   ├── parameter_server.py    ← agrège les gradients + Adam + comptabilité réseau
│   ├── work_generator.py      ← fabrique les sous-tâches d'une époque
│   ├── scheduler.py           ← ordonnanceur : puissance, timeouts, fiabilité
│   └── coordinator.py         ← cycle des époques, finalisation, signal de fin
│
├── jobs/                      ← les ALGORITHMES branchables
│   ├── __init__.py
│   └── rl_diagnosis/          ← notre algorithme (RL de diagnostic)
│       ├── __init__.py
│       ├── knowledge_base.py  ← base MedlinePlus + génération de patients
│       ├── environment.py     ← environnement de dialogue patient
│       ├── nn.py              ← réseaux NumPy (politique Q + classifieur) + Adam
│       ├── agent.py           ← collecte d'expérience + calcul du gradient
│       └── job.py             ← RLDiagnosisJob (implémente TrainingJob)
│
├── server/
│   ├── __init__.py
│   ├── app.py                 ← API HTTP (Flask)
│   └── dashboard.html         ← tableau de bord temps réel
│
├── client/
│   ├── __init__.py
│   ├── device_info.py         ← détection OS / cœurs / RAM / Android, puissance
│   └── volunteer.py           ← client volontaire (boucle de gradient)
│
├── data/
│   ├── MedlinePlus20.csv       ← 20 maladies, 110 symptômes (base par défaut)
│   └── MedlinePlus10.csv       ← 10 maladies (base réduite)
│
└── scripts/                   ← points d'entrée
    ├── prepare_data.py        ← vérifie/affiche la base
    ├── run_server.py          ← lance le serveur (déploiement réel)
    ├── run_volunteer.py       ← lance un volontaire (Android/Windows/Linux)
    ├── run_simulation.py      ← simulation locale de bout en bout (flotte simulée)
    ├── run_sequential.py      ← référence séquentielle (1 appareil)
    └── analyze.py             ← toutes les métriques + figures du mémoire
```

---

## 3. Installation

Sur le serveur (et pour l'analyse) :

```bash
pip install -r requirements.txt
```

Sur un **volontaire**, le strict minimum suffit :

```bash
pip install numpy requests
```

---

## 4. Démarrage rapide (sur une seule machine)

```bash
python scripts/prepare_data.py        # vérifie la base MedlinePlus
python scripts/run_simulation.py      # serveur + flotte hétérogène simulée, de bout en bout
```

Pour observer la **tolérance aux pannes** (un smartphone se déconnecte en cours) :

```bash
python scripts/run_simulation.py --kill
```

Pour un transport **encore plus léger** (sparsification des gradients) :

```bash
python scripts/run_simulation.py --topk 0.10
```

À la fin, toutes les métriques sont écrites dans `results/distributed_run.json`.

---

## 5. Déploiement réel (plusieurs appareils)

### 5.1 Lancer le serveur (sur la machine de coordination)

```bash
python scripts/run_server.py --host 0.0.0.0
```

Au démarrage, le serveur **affiche son adresse locale** (ex. `http://192.168.1.10:5000`)
et la **commande à donner aux volontaires**. Le **tableau de bord temps réel** est
à cette même adresse dans un navigateur. À la fin du calcul, le serveur **signale**
la convergence avec les métriques finales.

### 5.2 Participer depuis un volontaire

Remplacez `IP_DU_SERVEUR` par l'adresse affichée par le serveur.

**Android (Termux)**
```bash
pkg install python
pip install numpy requests
python scripts/run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device smartphone
```

**Windows (invite de commande)**
```bat
pip install numpy requests
python scripts\run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device windows-pc
```

**Linux (terminal)**
```bash
pip install numpy requests
python scripts/run_volunteer.py --server http://IP_DU_SERVEUR:5000 --device linux-pc
```

Options utiles : `--power N` (nombre de sous-tâches par requête, sinon automatique),
`--slowdown S` (ralentissement artificiel par sous-tâche, pour simuler un appareil lent).

> Les volontaires n'ont **rien de spécial** à gérer en cas de coupure : si un
> appareil se déconnecte, le serveur **réattribue** automatiquement ses sous-tâches.

---

## 6. Obtenir les métriques et figures du mémoire

```bash
python scripts/run_sequential.py --epochs 16   # référence séquentielle (courbes complètes)
python scripts/run_simulation.py               # exécution distribuée
python scripts/analyze.py                       # génère les 7 figures + metrics_memoire.json
```

Figures produites dans `results/` :

| Fichier | Démontre |
|---|---|
| `fig1_convergence_qualite.png` | qualité médicale : précision/top-3/F1 montent, **questions par patient diminuent** |
| `fig2_plus_value.png` | **plus-value vs séquentiel** : cible atteinte ~K× plus vite avec K volontaires |
| `fig3_passage_echelle.png` | temps pour atteindre la cible vs **nombre de volontaires** |
| `fig4_bande_passante.png` | **allègement du transport** : gradients compressés vs poids bruts |
| `fig5_heterogeneite.png` | **répartition de la charge** selon la puissance de chaque appareil |
| `fig6_tolerance_pannes.png` | **convergence malgré** la déconnexion de volontaires |
| `fig7_cout_accessibilite.png` | **coût et énergie** : flotte volontaire vs GPU cloud vs serveur dédié |

---

## 7. Extensibilité : faire tourner un autre algorithme après le mémoire

Le `framework/` ne connaît rien au diagnostic. Pour un nouvel algorithme, créez
`jobs/mon_algo/job.py` avec une classe héritant de `TrainingJob` (`framework/job.py`)
et implémentez :

```python
class MonJob(TrainingJob):
    name = "Mon algorithme"
    def init_params(self): ...            # vecteur de paramètres initial
    def n_params(self): ...               # taille du vecteur
    def init_opt_state(self): ...         # état initial de l'optimiseur (serveur)
    def make_task(self, epoch, index, seed, epsilon): ...   # décrit une sous-tâche
    def compute_gradient(self, params, task): ...           # côté volontaire -> gradient
    def apply_gradient(self, params, grad, opt_state, lr): ...# côté serveur -> pas d'optimiseur
    def evaluate(self, params, n): ...    # métriques (au moins 'accuracy')
```

Puis branchez-le dans `scripts/run_server.py` (à la place de `RLDiagnosisJob`).
L'ordonnancement, la tolérance aux pannes, le transport par gradients compressés,
l'hétérogénéité et le tableau de bord fonctionnent **sans aucune modification**.

---

## 8. Note d'honnêteté sur les mesures

La machine de développement peut ne disposer que d'**un seul cœur**. Le gain de
temps **parallèle** n'y est donc pas directement observable (les fils se partagent
le CPU). En conséquence :

* sont **mesurés réellement** : la convergence, la qualité du diagnostic, la
  tolérance aux pannes, la répartition selon l'hétérogénéité, et la réduction de
  bande passante ;
* est **modélisé** (à partir du temps par gradient réellement mesuré) : le gain de
  temps parallèle de `fig2`/`fig3`. Il se matérialise sur un **vrai déploiement
  multi-appareils**.

Le **modèle de coût** (`fig7`) repose sur des hypothèses de prix et de puissance
clairement indiquées et **éditables** en tête de `scripts/analyze.py` — à remplacer
par vos chiffres locaux. Le message est le **rapport** entre options (la flotte ne
paie que l'électricité marginale d'appareils déjà possédés, capital ≈ 0), invariant
d'échelle.
