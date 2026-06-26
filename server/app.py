"""
Serveur HTTP de coordination (Flask)
====================================
Expose l'API REST utilisee par les volontaires et sert le tableau de bord
temps reel. Le transport (poids descendants, gradients montants) passe par des
chaines base64 compressees.
"""

import os
from flask import Flask, request, jsonify, send_from_directory

from framework.coordinator import Coordinator

HERE = os.path.dirname(os.path.abspath(__file__))

# Le serveur ne démarre pas immédiatement l'entraînement.
# Les volontaires attendront que cette variable passe à True.
TRAINING_STARTED = False

def create_app(job, cfg):
    app = Flask(__name__)
    coord = Coordinator(job, cfg)
    app.coord = coord

    @app.get("/")
    def dashboard():
        return send_from_directory(HERE, "dashboard.html")

    @app.get("/kb")
    def kb():
        # specification de la base + reglages de transport (le volontaire reconstruit le job)
        return jsonify({"kb": job.kb_spec(),
                        "transport": {"dtype": cfg.transport.dtype, "topk": cfg.transport.topk}})

    @app.post("/request_work")
    def request_work():
        body = request.get_json(force=True)

        client_id = body["client_id"]
        info = body.get("info", {})
        power = body.get("power", 1)

        print(
            f"[POST /request_work] client={client_id} "
            f"device={info.get('device')} os={info.get('os')} "
            f"cpu={info.get('cpu')} ram={info.get('ram_gb')}Go power={power}"
        )

        out = coord.request_work(client_id, info, power)

        print(
            f"[SEND] client={client_id} "
            f"tasks={len(out.get('tasks', []))} "
            f"finished={out.get('finished')}"
        )

        return jsonify(out)

    @app.post("/report")
    def report():
        body = request.get_json(force=True)

        client_id = body["client_id"]
        results = body.get("results", [])

        print(f"[POST /report] client={client_id} results={len(results)}")

        for r in results:
            lm = r.get("local_metrics", {})
            print(
                f"  -> task={r.get('task_id')} "
                f"duration={lm.get('duration_seconds', 0):.2f}s "
                f"samples={r.get('n_samples')}"
            )

        coord.report_gradients(client_id, results)

        return jsonify({"ok": True, "finished": coord.finished})

    @app.get("/status")
    def status():
        return jsonify(coord.status())
    
    @app.post("/start_training")
    def start_training():
        global TRAINING_STARTED
        TRAINING_STARTED = True
        return jsonify({"started": True})

    @app.get("/training_status")
    def training_status():
        return jsonify({"started": TRAINING_STARTED})
    return app
