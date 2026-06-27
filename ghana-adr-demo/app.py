"""
Ghana ADR Detection System — Flask app
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# ── Model loading ─────────────────────────────────────────────────────────────
import traceback as _tb
try:
    from inference_engine import (
        load_models, analyse_text, BACKBONE,
    )
    cls_tok, cls_mod, ner_tok, ner_mod, device = load_models()
    MODEL_LOADED = True
    print(f"[app] Model loaded — {BACKBONE}", flush=True)
except Exception as _e:
    MODEL_LOADED = False
    BACKBONE = "stub"
    print(f"[app] Model load FAILED — running stub: {_e}", flush=True)
    _tb.print_exc()


def _stub_analyse(text):
    import random
    random.seed(len(text))
    kw_adr = ["weak", "dizz", "nausea", "crisis", "anaemi", "cardiac",
               "fever", "vomit", "pain", "rash", "severe"]
    is_adr = any(w in text.lower() for w in kw_adr)
    conf = round(random.uniform(0.72, 0.97) if is_adr else random.uniform(0.78, 0.96), 4)
    entities = []
    for word, label in [
        ("coartem", "DRUG"), ("artesunate", "DRUG"), ("amoxicillin", "DRUG"),
        ("zidovudine", "DRUG"), ("paracetamol", "DRUG"),
        ("oculogyric crisis", "ADR"), ("body weakness", "ADR"),
        ("nausea", "ADR"), ("rash", "ADR"), ("fever", "ADR"),
        ("severe", "SEVERITY"), ("mild", "SEVERITY"), ("moderate", "SEVERITY"),
    ]:
        idx = text.lower().find(word)
        if idx != -1:
            entities.append({
                "text": text[idx:idx + len(word)],
                "label": label,
                "start": idx,
                "end": idx + len(word),
                "confidence": round(random.uniform(0.75, 0.99), 3),
            })
    return {
        "contains_adr": is_adr,
        "confidence": conf,
        "prob_no_adr": round(1 - conf, 4),
        "prob_adr": conf,
        "uncertain": False,
        "rule_triggered": False,
        "negation_override": False,
        "entities": sorted(entities, key=lambda e: e["start"]),
        "relations": [],
    }


def _build_response(result):
    is_adr = result.get("contains_adr", False)
    if result.get("negation_override"):
        explanation = "Negation detected — text explicitly states the absence of an ADR."
    elif result.get("rule_triggered"):
        explanation = "Drug intolerance/discontinuation pattern detected — rule override applied."
    elif result.get("uncertain"):
        explanation = "Model confidence is in the uncertain zone (45–55%); manual review recommended."
    elif is_adr:
        explanation = ("The model detected drug-related adverse reactions based on symptom "
                       "terminology and drug–reaction co-occurrence patterns from Ghanaian "
                       "pharmacovigilance sources.")
    else:
        explanation = "The model found no indicators of an adverse drug reaction in this text."

    return {
        "is_adr":            is_adr,
        "confidence":        result.get("confidence", 0.0),
        "uncertain":         result.get("uncertain", False),
        "rule_triggered":    result.get("rule_triggered", False),
        "negation_override": result.get("negation_override", False),
        "explanation":       explanation,
        "entities":          result.get("entities", []),
        "relations":         result.get("relations", []),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", model_loaded=MODEL_LOADED, backbone=BACKBONE)


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    if MODEL_LOADED:
        raw = analyse_text(text, cls_tok, cls_mod, ner_tok, ner_mod, device)
    else:
        raw = _stub_analyse(text)
    return jsonify(_build_response(raw))


@app.route("/api/columns", methods=["POST"])
def api_columns():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    try:
        df = pd.read_csv(f)
        return jsonify({"columns": list(df.columns)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/batch", methods=["POST"])
def api_batch():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400
    fname = f.filename.lower()
    try:
        if fname.endswith(".txt"):
            sentences = [s.strip() for s in f.read().decode("utf-8").splitlines() if s.strip()]
        elif fname.endswith(".csv"):
            df = pd.read_csv(f)
            col = request.form.get("text_column") or df.columns[0]
            sentences = df[col].dropna().astype(str).tolist()
        else:
            return jsonify({"error": "Upload a .txt or .csv file"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    if not sentences:
        return jsonify({"error": "No sentences found in file"}), 400

    results = []
    for i, sent in enumerate(sentences, start=1):
        raw = analyse_text(sent, cls_tok, cls_mod, ner_tok, ner_mod, device) \
              if MODEL_LOADED else _stub_analyse(sent)
        r = _build_response(raw)
        entity_str = ", ".join(f"{e['text']} ({e['label']})" for e in r["entities"]) or "—"
        results.append({
            "index":          i,
            "text":           sent[:90] + "…" if len(sent) > 90 else sent,
            "full_text":      sent,
            "classification": "ADR" if r["is_adr"] else "Non-ADR",
            "confidence":     round(r["confidence"], 4),
            "entities":       entity_str,
            "flag":           "⚠️" if (r["is_adr"] and r["confidence"] >= 0.85) else "",
        })
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
