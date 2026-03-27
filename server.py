"""
Persona Dialogue System — Flask API.
Local: python server.py  ->  http://127.0.0.1:5000
Vercel: deployed automatically via vercel.json
"""
import os
import sys
import re
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

try:
    import sacrebleu
    from rouge_score import rouge_scorer
    _SCORING_OK = True
except ImportError:
    _SCORING_OK = False

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from train_bart import (load_model, generate_persona_response, reinforcement_update,
                        compute_reward, _hf_embed, _IS_VERCEL, _cos_sim_np)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True
CORS(app)

# --- Load model & data once at startup ---
model_data  = load_model()
personas_df = None

try:
    personas_df = pd.read_csv(os.path.join(PROJECT_DIR, "personality.csv"), encoding="latin1")
except Exception:
    pass


# ── Keyword-based grounding (fallback layer) ───────────────────────────────────
_STOPWORDS = {
    'what', 'is', 'are', 'my', 'your', 'the', 'a', 'an', 'do', 'does',
    'i', 'you', 'me', 'we', 'he', 'she', 'it', 'to', 'of', 'and', 'or',
    'in', 'on', 'at', 'be', 'was', 'were', 'have', 'has', 'had', 'that',
    'this', 'for', 'with', 'about', 'tell', 'wt', 'did', 'can', 'could',
    'would', 'will', 'not', 'no', 'so', 'if', 'but', 'by', 'from',
}

_BROAD_TRIGGERS = {
    'like', 'love', 'enjoy', 'interest', 'hobby', 'hobbies', 'interests',
    'things', 'activities', 'favorites', 'favourite', 'passions',
}


def _keywords(text: str) -> set:
    tokens = re.sub(r"[^a-z\s]", "", text.lower()).split()
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


def _persona_sentences(persona: str):
    """Split persona into sentences, protecting abbreviations like B.Tech, Ph.D, 8.5."""
    protected = re.sub(r'([A-Za-z])\.([A-Za-z])', r'\1<P>\2', persona)
    protected = re.sub(r'(\d)\.(\d)', r'\1<P>\2', protected)
    parts = re.split(r'[.!?]+', protected)
    return [s.replace('<P>', '.').strip() for s in parts if len(s.strip()) > 3]


def _extract_from_persona(persona: str, question: str) -> str:
    """Keyword-matching fallback — always grounded in the persona."""
    q_kw  = _keywords(question)
    sents = _persona_sentences(persona)
    if not sents:
        return persona.strip()

    scored = sorted(
        [(len(q_kw & _keywords(s)), s) for s in sents],
        key=lambda x: x[0], reverse=True,
    )

    q_lower  = question.lower()
    is_broad = any(t in q_lower for t in _BROAD_TRIGGERS)
    if is_broad:
        relevant = [s for sc, s in scored if sc > 0] or [s for _, s in scored[:3]]
        parts = []
        for s in relevant[:5]:
            s = s.strip()
            parts.append(s if s.endswith(".") else s + ".")
        return "  ".join(parts)

    best = scored[0][1].strip()
    return best if best.endswith(".") else best + "."


def _is_empty(response: str) -> bool:
    return not response or not response.strip()


def _compute_scores(bot_response: str, ideal_response: str):
    if not _SCORING_OK:
        return {"error": "sacrebleu / rouge-score not installed"}
    try:
        bleu   = sacrebleu.corpus_bleu([bot_response], [[ideal_response]])
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(ideal_response, bot_response)
        return {
            "bleu":   f"{bleu.score:.2f}",
            "rouge1": f"{scores['rouge1'].fmeasure:.2f}",
            "rougeL": f"{scores['rougeL'].fmeasure:.2f}",
        }
    except Exception as e:
        return {"error": str(e)}


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_persona", methods=["GET"])
def get_persona():
    if personas_df is not None and len(personas_df) > 0:
        return jsonify({"persona": personas_df.sample(1).iloc[0].Persona})
    return jsonify({"persona": "I am an AI assistant."})


@app.route("/chat", methods=["POST"])
def chat():
    data     = request.get_json() or {}
    persona  = (data.get("persona")  or "").strip()
    question = (data.get("message")  or "").strip()

    if not question:
        return jsonify({"response": "Enter a question."})

    try:
        bot_response = generate_persona_response(model_data, persona, question)
        if _is_empty(bot_response):
            bot_response = _extract_from_persona(persona, question)
    except Exception as e:
        bot_response = _extract_from_persona(persona, question) if persona else str(e)

    # Guarantee capital first letter on every path
    bot_response = (bot_response or "I'm not sure.").strip()
    if bot_response:
        bot_response = bot_response[0].upper() + bot_response[1:]

    return jsonify({"response": bot_response})


@app.route("/evaluate_and_update", methods=["POST"])
def evaluate_and_update():
    data         = request.get_json() or {}
    persona      = (data.get("persona")      or "").strip()
    question     = (data.get("question")     or "").strip()
    bot_response = (data.get("bot_response") or "").strip()
    ideal        = (data.get("ideal")        or "").strip()

    if not ideal:
        return jsonify({"error": "No ideal response provided."})

    result = _compute_scores(bot_response, ideal)
    if "error" in result:
        return jsonify(result)

    result["ideal"] = ideal
    b   = float(result["bleu"])
    r1  = float(result["rouge1"])
    rL  = float(result["rougeL"])
    reward = compute_reward(b, r1, rL)
    result["reward"]  = f"{reward:.4f}"
    result["updated"] = False
    result["message"] = "Scores computed. No RL update (missing persona / question)."

    if persona and question and bot_response:
        try:
            analytics = reinforcement_update(
                model_data, persona, question,
                bot_response, ideal_text=ideal, reward=reward,
            )

            # ── Build a human-readable learning report ───────────────────────
            trend_emoji = {"improving": "(improving)", "needs work": "(needs work)", "stable": "(stable)"}.get(
                analytics.get("trend", "stable"), "(stable)")

            result["updated"]       = True
            result["mlp_updated"]   = analytics.get("mlp_updated", False)
            result["memory_stored"] = analytics.get("memory_stored", False)
            result["total_learned"] = analytics.get("total_learned", 0)
            result["trend"]         = analytics.get("trend", "stable")
            result["quality"]       = analytics.get("quality", "")

            result["message"] = (
                f"{analytics.get('quality','Done')}  |  "
                f"Trend: {analytics.get('trend','stable')} {trend_emoji}  |  "
                f"Total examples learned: {analytics.get('total_learned', 0)}\n"
                f"{analytics.get('action','')}"
            )
        except Exception as e:
            result["message"] = f"RL error: {e}"

    return jsonify(result)




@app.route("/chat_multi", methods=["POST"])
def chat_multi():
    data     = request.get_json() or {}
    personas = data.get("personas", [])
    question = (data.get("message") or "").strip()

    if not question:
        return jsonify({"error": "No message provided."})
    if not personas:
        return jsonify({"error": "No personas provided."})

    # Generate a response for each persona
    responses = []
    for persona in personas:
        persona = (persona or "").strip()
        try:
            resp = generate_persona_response(model_data, persona, question)
            if _is_empty(resp):
                resp = _extract_from_persona(persona, question)
        except Exception as e:
            resp = _extract_from_persona(persona, question) if persona else str(e)
        resp = (resp or "I'm not sure.").strip()
        if resp:
            resp = resp[0].upper() + resp[1:]
        responses.append(resp)

    # Compute pairwise similarities to detect conflict
    try:
        if _IS_VERCEL:
            # Use text similarity on Vercel — no API call needed
            from difflib import SequenceMatcher
            all_agree = True
            n = len(responses)
            for i in range(n):
                for j in range(i + 1, n):
                    sim = SequenceMatcher(None, responses[i].lower(), responses[j].lower()).ratio()
                    if sim <= 0.60:
                        all_agree = False
                        break
                if not all_agree:
                    break
        else:
            st_model = model_data.get("model")
            if st_model is None:
                raise ValueError("No embedding model available")
            emb_matrix = st_model.encode(responses, convert_to_numpy=True)
            all_agree = True
            n = len(responses)
            for i in range(n):
                for j in range(i + 1, n):
                    sim = _cos_sim_np(emb_matrix[i], emb_matrix[j])
                    if sim <= 0.80:
                        all_agree = False
                        break
                if not all_agree:
                    break
    except Exception:
        all_agree = False

    if all_agree:
        # Return merged — use the first response as the canonical one
        return jsonify({
            "mode": "merged",
            "response": responses[0],
            "responses": responses,
        })
    else:
        # Return split — each persona responds separately
        split_responses = [
            {"persona_idx": i + 1, "response": r}
            for i, r in enumerate(responses)
        ]
        return jsonify({
            "mode": "split",
            "responses": split_responses,
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server ready:  http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
