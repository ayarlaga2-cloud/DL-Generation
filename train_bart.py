import os
import re
import sys
import time
import pickle
from difflib import SequenceMatcher
import random
import requests
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy imports — available locally, not on Vercel
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util as _st_util
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer as _rs
    _SCORING_AVAILABLE = True
except ImportError:
    _SCORING_AVAILABLE = False

try:
    from spellchecker import SpellChecker
    _spell = SpellChecker()
    _SPELL_AVAILABLE = True
except ImportError:
    _SPELL_AVAILABLE = False
    _spell = None

# ---------------------------------------------------------------------------
# Vercel detection
# ---------------------------------------------------------------------------
_IS_VERCEL = bool(os.environ.get("VERCEL")) or (not _TORCH_AVAILABLE)

# ---------------------------------------------------------------------------
# Hugging Face Inference API (used on Vercel instead of local models)
# ---------------------------------------------------------------------------
_HF_TOKEN    = os.environ.get("HF_API_TOKEN", "")
_HF_EMBED_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
_HF_GEN_URL   = "https://api-inference.huggingface.co/models/google/flan-t5-base"


def _hf_headers():
    return {"Authorization": f"Bearer {_HF_TOKEN}"} if _HF_TOKEN else {}


def _hf_embed(texts):
    """Get sentence embeddings via HF Inference API."""
    if isinstance(texts, str):
        texts = [texts]
    try:
        resp = requests.post(
            _HF_EMBED_URL,
            headers=_hf_headers(),
            json={"inputs": texts, "options": {"wait_for_model": True}},
            timeout=30,
        )
        data = resp.json()
        if isinstance(data, list) and len(data) > 0:
            arr = np.array(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
    except Exception:
        pass
    return np.zeros((len(texts), 384), dtype=np.float32)


def _hf_generate(prompt):
    """Generate text via HF Inference API."""
    try:
        resp = requests.post(
            _HF_GEN_URL,
            headers=_hf_headers(),
            json={
                "inputs": prompt,
                "parameters": {"max_new_tokens": 120},
                "options": {"wait_for_model": True},
            },
            timeout=30,
        )
        data = resp.json()
        if isinstance(data, list) and data:
            return (data[0].get("generated_text") or "").strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
    except Exception:
        pass
    return ""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EPOCHS         = 15
BATCH_SIZE     = 64
LEARNING_RATE  = 0.001
TEST_SPLIT     = 0.20
VAL_SPLIT      = 0.10
MAX_QA_PAIRS   = 14000
MAX_PERSONAS   = 5000

PICKLE_PATH    = "persona_model.pkl"
DATA_PATH      = "personality.csv"
QA_CONF_THRESH = 0.08

INPUT_DIM  = 769
HIDDEN1    = 256
HIDDEN2    = 64
NUM_LABELS = 2

_st_model_instance     = None
_gen_pipeline_instance = None

# ---------------------------------------------------------------------------
# Local model loaders (only used when torch/transformers available)
# ---------------------------------------------------------------------------

def _get_st_model():
    global _st_model_instance
    if _IS_VERCEL:
        return None
    if _st_model_instance is None and _ST_AVAILABLE:
        print("Loading sentence-transformer: all-MiniLM-L6-v2  (~80 MB, one-time)")
        _st_model_instance = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model_instance


def _get_gen_pipeline():
    global _gen_pipeline_instance
    if _IS_VERCEL:
        return None
    if _gen_pipeline_instance is None and _TRANSFORMERS_AVAILABLE:
        print("Loading generative model: google/flan-t5-base  (~900 MB, one-time)")
        _gen_pipeline_instance = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=150,
        )
        print("Generative model ready.")
    return _gen_pipeline_instance


# ---------------------------------------------------------------------------
# MLP model (PyTorch, used locally only)
# ---------------------------------------------------------------------------

if _TORCH_AVAILABLE:
    class PersonaQAClassifier(torch.nn.Module):
        def __init__(self, input_dim=INPUT_DIM, hidden1=HIDDEN1,
                     hidden2=HIDDEN2, num_labels=NUM_LABELS):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden1),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.30),
                torch.nn.Linear(hidden1, hidden2),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.20),
                torch.nn.Linear(hidden2, num_labels),
            )

        def forward(self, x):
            return self.net(x)


def _extract_numpy_weights(mlp):
    """Extract MLP weights as plain numpy arrays for Vercel deployment."""
    sd = mlp.state_dict()
    return {
        "fc1_w": sd["net.0.weight"].cpu().numpy(),
        "fc1_b": sd["net.0.bias"].cpu().numpy(),
        "fc2_w": sd["net.3.weight"].cpu().numpy(),
        "fc2_b": sd["net.3.bias"].cpu().numpy(),
        "fc3_w": sd["net.6.weight"].cpu().numpy(),
        "fc3_b": sd["net.6.bias"].cpu().numpy(),
    }


def _numpy_mlp_forward(nw, x):
    """Pure numpy forward pass — no torch needed."""
    x = np.maximum(0, x @ nw["fc1_w"].T + nw["fc1_b"])
    x = np.maximum(0, x @ nw["fc2_w"].T + nw["fc2_b"])
    x = x @ nw["fc3_w"].T + nw["fc3_b"]
    ex = np.exp(x - x.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Spell-checker helpers
# ---------------------------------------------------------------------------

_TEXTSPEAK = {
    r"\bwt\b":    "what",    r"\bwht\b":  "what",
    r"\bu\b":     "you",     r"\bur\b":   "your",
    r"\br\b":     "are",     r"\bim\b":   "i am",
    r"\bfav\b":   "favorite", r"\bfave\b": "favorite",
    r"\bdnt\b":   "do not",  r"\bdont\b": "do not",
    r"\bcant\b":  "cannot",  r"\bwont\b": "will not",
    r"\bgonna\b": "going to", r"\bwanna\b": "want to",
    r"\bhv\b":    "have",    r"\bplz\b":  "please",
    r"\bpls\b":   "please",  r"\bbtw\b":  "by the way",
    r"\bidk\b":   "i do not know", r"\bngl\b": "not going to lie",
    r"\bthx\b":   "thanks",
}


def _register_persona_vocab(df: pd.DataFrame):
    if not _SPELL_AVAILABLE or _spell is None:
        return
    vocab = set()
    for persona in df["Persona"].dropna():
        for word in re.findall(r"[A-Za-z]+", str(persona)):
            vocab.add(word.lower())
    if vocab:
        _spell.word_frequency.load_words(vocab)
        print(f"Spell-checker updated with {len(vocab)} persona vocabulary words.")


def _normalize_question(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\?\.\!,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    for pat, rep in _TEXTSPEAK.items():
        text = re.sub(pat, rep, text)

    if _SPELL_AVAILABLE and _spell is not None:
        tokens     = text.split()
        only_words = [t for t in tokens if t.isalpha() and len(t) > 2]
        misspelled = _spell.unknown(only_words)
        corrected  = []
        for tok in tokens:
            if tok in misspelled:
                fix = _spell.correction(tok)
                corrected.append(fix if fix else tok)
            else:
                corrected.append(tok)
        text = " ".join(corrected)

    if text and text[-1] not in ".?!":
        text += "?"
    return text


# ---------------------------------------------------------------------------
# Sentence utilities
# ---------------------------------------------------------------------------

def _norm(s):
    return " ".join(s.strip().lower().split()).replace(".", "").replace("?", "")


def _split_sentences(text: str):
    protected = re.sub(r"([A-Za-z])\.([A-Za-z])", r"\1<P>\2", text)
    protected = re.sub(r"(\d)\.(\d)", r"\1<P>\2", protected)
    parts = re.split(r"[.!?]+", protected)
    return [s.replace("<P>", ".").strip() for s in parts if len(s.strip()) > 3]


def _format_response(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


# ---------------------------------------------------------------------------
# Embedding helper — works both locally and on Vercel
# ---------------------------------------------------------------------------

def _encode(texts, model_data):
    """Encode texts to numpy embeddings using local model or HF API."""
    if isinstance(texts, str):
        texts = [texts]
    if _IS_VERCEL:
        return _hf_embed(texts)
    st = model_data.get("model")
    if st is not None:
        embs = st.encode(texts, convert_to_numpy=True, batch_size=64)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        return embs.astype(np.float32)
    return _hf_embed(texts)


def _cos_sim_np(a, b):
    """Cosine similarity between two 1-D numpy arrays."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def _build_feature_matrix(q_embs: np.ndarray, s_embs: np.ndarray) -> np.ndarray:
    q_n = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)
    s_n = s_embs / (np.linalg.norm(s_embs, axis=1, keepdims=True) + 1e-9)
    cos = np.einsum("ij,ij->i", q_n, s_n).reshape(-1, 1)
    return np.concatenate([q_n, s_n, cos], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# QA pair generation
# ---------------------------------------------------------------------------

def questions_for_sentence(sentence):
    s   = sentence.strip()
    if not s or len(s) < 3:
        return []
    nrm   = _norm(s)
    words = nrm.split()
    if not words:
        return []

    out    = []
    answer = s if s.endswith(".") else s + "."

    if re.search(r"\b(won|have won)\b", nrm):
        out += [("What have you won?", answer), ("What did you win?", answer)]
    if re.search(r"\b(my mom|my mother)\b", nrm):
        sub = "mom" if "mom" in nrm else "mother"
        out += [(f"What does your {sub} do?", answer),
                (f"Tell me about your {sub}.", answer)]
    if re.search(r"\b(my dad|my father)\b", nrm):
        sub = "dad" if "dad" in nrm else "father"
        out += [(f"What does your {sub} do?", answer),
                (f"Tell me about your {sub}.", answer)]
    if "mom" in nrm and "dad" in nrm:
        out += [("What do your parents do?", answer),
                ("Tell me about your parents.", answer)]
    if "have always wanted" in nrm or "always wanted" in nrm:
        out += [("What have you always wanted?", answer),
                ("What do you want most?", answer)]
    if words[0] == "i" and len(words) > 2 and words[1] in ("love", "like", "enjoy"):
        v = words[1]
        out += [(f"What do you {v}?", answer),
                ("What are your hobbies?", answer),
                ("What are your interests?", answer)]
    if "favorite" in nrm or "favourite" in nrm:
        out += [("What is your favorite thing?", answer),
                ("What do you like the most?", answer)]
        m = re.search(r"(?:favorite|favourite)\s+(\w+)", nrm)
        if m:
            out += [(f"What is your favorite {m.group(1)}?", answer)]
    if re.match(r"^i\s+(?:am|'m)\s+", nrm):
        out += [("What are you?", answer), ("Who are you?", answer),
                ("Tell me about yourself.", answer)]
    if words[0] == "i" and len(words) > 2 and words[1] == "have":
        out += [("What do you have?", answer), ("What do you own?", answer)]
    if re.search(r"\bwork\b|\bjob\b|\boccupation\b", nrm):
        out += [("What do you do for work?", answer),
                ("What is your job?", answer)]
    if re.search(r"\bname\b", nrm):
        out += [("What is your name?", answer),
                ("What is my name?", answer), ("Who are you?", answer)]
    if re.search(r"\bstud(y|ying|ied)\b|\buniversity\b|\bcollege\b|\bschool\b", nrm):
        out += [("Where do you study?", answer),
                ("What do you study?", answer),
                ("What is your university?", answer)]
    if re.search(r"\bgpa\b|\bgrade\b|\bscore\b", nrm):
        out += [("What is your GPA?", answer),
                ("What are your grades?", answer)]
    if not out:
        topic = max(words, key=len) if words else "that"
        out += [(f"Tell me about {topic}.", answer),
                ("What can you tell me about yourself?", answer)]
    return out


def build_persona_qa_df(persona_df):
    rows = []
    for _, row in persona_df.iterrows():
        context   = str(row["Persona"]).strip()
        if not context:
            continue
        sentences = _split_sentences(context)
        for sent in sentences:
            sent = sent if sent.endswith(".") else sent + "."
            for q, a in questions_for_sentence(sent):
                rows.append({
                    "context":       context,
                    "question":      q.strip(),
                    "answer":        a.strip(),
                    "all_sentences": sentences,
                })
        for q in ["Who are you?", "Tell me about yourself.", "Describe yourself."]:
            rows.append({"context": context, "question": q,
                         "answer": context, "all_sentences": sentences})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def load_model(pickle_path=PICKLE_PATH):
    model_data = {
        "model":          _get_st_model(),
        "gen_pipeline":   _get_gen_pipeline(),
        "mlp_model":      None,
        "numpy_weights":  None,
        "feedback_pairs": [],
        "trained":        False,
    }

    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                saved = pickle.load(f)

            # Load numpy weights (Vercel-compatible)
            nw = saved.get("numpy_weights")
            if nw:
                model_data["numpy_weights"] = nw
                model_data["trained"] = True

            # Load torch model (local only)
            if _TORCH_AVAILABLE:
                cfg = saved.get("mlp_config", {})
                if cfg and saved.get("mlp_state_dict"):
                    mlp = PersonaQAClassifier(**cfg)
                    mlp.load_state_dict(saved["mlp_state_dict"])
                    mlp.eval()
                    model_data["mlp_model"] = mlp
                    # Also derive numpy weights if not already present
                    if model_data["numpy_weights"] is None:
                        model_data["numpy_weights"] = _extract_numpy_weights(mlp)

            model_data["feedback_pairs"] = saved.get("feedback_pairs", [])
            model_data["trained"] = (
                model_data["trained"] or saved.get("trained", False)
            )
            size_kb = os.path.getsize(pickle_path) / 1024
            status  = "trained" if model_data["trained"] else "untrained"
            print(f"Loaded {pickle_path}  ({size_kb:.0f} KB,  {status},  "
                  f"{len(model_data['feedback_pairs'])} RL pairs)")
        except Exception as e:
            print(f"Warning: could not load {pickle_path}: {e}")
    else:
        print("No trained model found. Run:  python train_bart.py")

    # Register persona vocab with spell-checker
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_PATH)
    if os.path.exists(csv_path):
        try:
            _df = pd.read_csv(csv_path, encoding="latin1").dropna(subset=["Persona"])
            _register_persona_vocab(_df)
        except Exception:
            pass

    return model_data


def save_model(model_data, pickle_path=PICKLE_PATH):
    if _IS_VERCEL:
        return  # Vercel filesystem is read-only; nothing to persist between requests
    mlp          = model_data.get("mlp_model")
    numpy_weights = model_data.get("numpy_weights")

    # Always derive fresh numpy weights from torch model when available
    if mlp is not None and _TORCH_AVAILABLE:
        numpy_weights = _extract_numpy_weights(mlp)

    payload = {
        "mlp_state_dict": mlp.state_dict() if (mlp and _TORCH_AVAILABLE) else None,
        "numpy_weights":  numpy_weights,
        "mlp_config":     {
            "input_dim": INPUT_DIM, "hidden1": HIDDEN1,
            "hidden2": HIDDEN2, "num_labels": NUM_LABELS,
        },
        "feedback_pairs": model_data.get("feedback_pairs", []),
        "trained":        model_data.get("trained", False),
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    size_kb = os.path.getsize(pickle_path) / 1024
    print(f"Saved {pickle_path}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Training (local only)
# ---------------------------------------------------------------------------

def train_model(model_data, df,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                lr=LEARNING_RATE, verbose=True):
    if not _TORCH_AVAILABLE:
        print("torch not available — training skipped.")
        return model_data

    st = model_data["model"]
    _register_persona_vocab(df)

    if verbose:
        print(f"\nGenerating QA pairs from {len(df)} personas...")
    qa_df = build_persona_qa_df(df)
    if qa_df.empty:
        print("ERROR: No QA pairs generated. Check personality.csv.")
        return model_data

    if len(qa_df) > MAX_QA_PAIRS:
        qa_df = qa_df.sample(n=MAX_QA_PAIRS, random_state=42).reset_index(drop=True)

    if verbose:
        print(f"Total QA pairs: {len(qa_df)}  "
              f"(each becomes 1 positive + 1 negative example)")

    questions     = qa_df["question"].tolist()
    pos_sentences = qa_df["answer"].tolist()
    neg_sentences = []
    for _, row in qa_df.iterrows():
        others = [s for s in row["all_sentences"]
                  if _norm(s) != _norm(row["answer"])]
        if others:
            neg_sentences.append(random.choice(others))
        else:
            other_sents = [s.strip() for s in
                           re.split(r"[.]", str(df.sample(1).iloc[0]["Persona"]))
                           if len(s.strip()) > 2]
            neg_sentences.append(
                random.choice(other_sents) if other_sents else "I don't know.")

    if verbose:
        print("\nEncoding features with sentence-transformer...")
    all_texts = questions + pos_sentences + neg_sentences
    all_embs  = st.encode(all_texts, batch_size=128,
                          show_progress_bar=verbose, convert_to_numpy=True)
    n        = len(questions)
    q_embs   = all_embs[:n]
    pos_embs = all_embs[n:2*n]
    neg_embs = all_embs[2*n:3*n]

    X_pos = _build_feature_matrix(q_embs, pos_embs)
    X_neg = _build_feature_matrix(q_embs, neg_embs)
    X     = np.vstack([X_pos, X_neg])
    y     = np.array([1]*n + [0]*n, dtype=np.int64)

    idx  = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y)
    val_fraction = VAL_SPLIT / (1.0 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction, random_state=42, stratify=y_trainval)

    if verbose:
        print(f"\nData split:")
        print(f"  Train : {len(X_train):>6} samples  ({len(X_train)/len(X)*100:.0f}%)")
        print(f"  Val   : {len(X_val):>6} samples  ({len(X_val)/len(X)*100:.0f}%)")
        print(f"  Test  : {len(X_test):>6} samples  ({len(X_test)/len(X)*100:.0f}%)")
        print(f"\nModel: PersonaQAClassifier  "
              f"[{INPUT_DIM} -> {HIDDEN1} -> {HIDDEN2} -> {NUM_LABELS}]")
        print(f"Optimizer: Adam  lr={lr}")
        print(f"Epochs: {epochs}   Batch size: {batch_size}\n")

    X_tr_t  = torch.FloatTensor(X_train)
    y_tr_t  = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_tst_t = torch.FloatTensor(X_test)
    y_tst_t = torch.LongTensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True
    )

    mlp       = PersonaQAClassifier()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss = float("inf")
    best_state    = None

    header = (f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>9} | {'Val Acc':>8}")
    if verbose:
        print(header)
        print("-" * len(header))

    for epoch in range(1, epochs + 1):
        mlp.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = mlp(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(y_batch)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total   += len(y_batch)

        scheduler.step()
        avg_train_loss = train_loss / train_total
        train_acc      = train_correct / train_total * 100

        mlp.eval()
        with torch.no_grad():
            val_logits = mlp(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_acc    = (val_logits.argmax(1) == y_val_t).float().mean().item() * 100

        if verbose:
            print(f"{epoch:>6} | {avg_train_loss:>10.4f} | {train_acc:>8.2f}% | "
                  f"{val_loss:>9.4f} | {val_acc:>7.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in mlp.state_dict().items()}

    if best_state:
        mlp.load_state_dict(best_state)
    mlp.eval()

    with torch.no_grad():
        test_logits = mlp(X_tst_t)
        test_acc    = (test_logits.argmax(1) == y_tst_t).float().mean().item() * 100
        test_loss   = criterion(test_logits, y_tst_t).item()

    if verbose:
        print("-" * len(header))
        print(f"\nFinal TEST set  ->  Loss: {test_loss:.4f}  |  "
              f"Accuracy: {test_acc:.2f}%")

    model_data["mlp_model"]     = mlp
    model_data["numpy_weights"] = _extract_numpy_weights(mlp)
    model_data["trained"]       = True
    return model_data


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _find_persona_sentence(keyword: str, persona: str, model_data=None) -> str:
    sentences = _split_sentences(persona)
    if not sentences:
        return ""
    kw_lower = keyword.lower().rstrip(".")

    if _IS_VERCEL:
        # Use keyword overlap — no API call needed
        scores = _keyword_scores(keyword, sentences)
        best_idx = int(np.argmax(scores)) if scores.max() > 0 else -1
        if best_idx >= 0:
            return sentences[best_idx]
        # fallback: substring match
        for sent in sentences:
            if kw_lower in sent.lower():
                return sent
        return ""

    try:
        em       = _get_st_model()
        kw_emb   = em.encode(keyword,   convert_to_tensor=True)
        sent_emb = em.encode(sentences, convert_to_tensor=True)
        scores   = _st_util.cos_sim(kw_emb, sent_emb)[0].cpu().numpy()
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) > 0.30:
            return sentences[best_idx]
    except Exception:
        pass

    for sent in sentences:
        if kw_lower in sent.lower():
            return sent
    return ""


def _keyword_scores(question, sentences):
    """Reliable keyword-overlap scoring — used on Vercel when embeddings fail."""
    _stop = {
        'what','is','are','my','your','the','a','an','do','does','i','you',
        'me','we','he','she','it','to','of','and','or','in','on','at','be',
        'was','were','have','has','had','that','this','for','with','about',
        'tell','wt','did','can','could','would','will','not','no','so','if',
        'but','by','from','when','where','why','how','who',
    }
    q_kw = {w for w in re.sub(r"[^a-z\s]", "", question.lower()).split()
            if w not in _stop and len(w) > 2}
    scores = []
    for sent in sentences:
        s_kw  = {w for w in re.sub(r"[^a-z\s]", "", sent.lower()).split()
                 if w not in _stop and len(w) > 2}
        overlap = len(q_kw & s_kw)
        # bonus for partial word matches (e.g. "univ" matches "university")
        partial = sum(1 for qw in q_kw for sw in s_kw
                      if len(qw) > 3 and (qw in sw or sw in qw))
        scores.append(overlap + partial * 0.5)
    return np.array(scores, dtype=np.float32)


def _mlp_scores(model_data, persona, question):
    sentences = _split_sentences(persona)
    if not sentences:
        return sentences, np.array([])

    if _IS_VERCEL:
        # On Vercel: always use keyword scoring — no external API calls, fully reliable
        return sentences, _keyword_scores(question, sentences)
    else:
        st     = model_data.get("model")
        q_emb  = st.encode(question,  convert_to_numpy=True).reshape(1, -1)
        s_embs = st.encode(sentences, convert_to_numpy=True, batch_size=64)

    feats = _build_feature_matrix(np.tile(q_emb, (len(sentences), 1)), s_embs)

    nw  = model_data.get("numpy_weights")
    mlp = model_data.get("mlp_model")

    if nw is not None:
        scores = _numpy_mlp_forward(nw, feats)[:, 1]
    elif mlp is not None and _TORCH_AVAILABLE:
        mlp.eval()
        with torch.no_grad():
            logits = mlp(torch.FloatTensor(feats))
            scores = torch.softmax(logits, dim=1)[:, 1].numpy()
    else:
        q_n    = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        s_n    = s_embs / (np.linalg.norm(s_embs, axis=1, keepdims=True) + 1e-9)
        scores = (s_n @ q_n.T).flatten()

    # Feedback boost (local only — requires sentence-transformer)
    if not _IS_VERCEL:
        boost    = np.zeros(len(sentences))
        st_model = model_data.get("model")
        if st_model:
            q_emb_t = st_model.encode(question, convert_to_tensor=True)
            for fp in model_data.get("feedback_pairs", []):
                if fp.get("reward", 0) < 0.2:
                    continue
                fp_q  = st_model.encode(fp["question"], convert_to_tensor=True)
                q_sim = _st_util.cos_sim(q_emb_t, fp_q).item()
                if q_sim < 0.65:
                    continue
                ideal_sents = _split_sentences(fp.get("ideal", ""))
                if not ideal_sents:
                    continue
                ideal_embs = st_model.encode(ideal_sents, convert_to_tensor=True)
                s_embs_t   = st_model.encode(sentences,   convert_to_tensor=True)
                for i, s_emb_t in enumerate(s_embs_t):
                    match = _st_util.cos_sim(s_emb_t.unsqueeze(0), ideal_embs).max().item()
                    if match > 0.55:
                        boost[i] += fp["reward"] * q_sim * 0.2
        scores = scores + boost

    return sentences, scores


def _check_learned_memory(model_data, question: str, threshold: float = 0.82):
    pairs = model_data.get("feedback_pairs", [])
    if not pairs:
        return None

    # On Vercel: use text similarity (no HF API needed — 100% reliable)
    if _IS_VERCEL:
        norm_new   = _normalize_question(question).lower()
        best_sim   = 0.70          # text-similarity threshold
        best_ideal = None
        for pair in pairs:
            stored_norm = pair.get("norm_q") or pair.get("question", "")
            stored_norm = stored_norm.lower()
            sim = SequenceMatcher(None, norm_new, stored_norm).ratio()
            if sim > best_sim and pair.get("ideal"):
                best_sim   = sim
                best_ideal = pair["ideal"]
        return best_ideal if best_ideal else None

    # Local: use sentence-transformer cosine similarity
    st = model_data.get("model")
    if st is None:
        return None
    q_emb  = st.encode(question, convert_to_numpy=True)
    q_norm = np.linalg.norm(q_emb)
    if q_norm < 1e-9:
        return None

    best_sim   = threshold
    best_ideal = None
    for pair in pairs:
        stored = pair.get("q_emb")
        if stored is None:
            continue
        s_emb  = np.array(stored, dtype=np.float32)
        s_norm = np.linalg.norm(s_emb)
        if s_norm < 1e-9:
            continue
        sim = float(np.dot(q_emb, s_emb) / (q_norm * s_norm))
        if sim > best_sim:
            best_sim   = sim
            best_ideal = pair.get("ideal", "")

    return best_ideal if best_ideal else None


def _online_update_mlp(model_data, question: str, ideal_text: str,
                       reward: float, n_steps: int = 8):
    if _IS_VERCEL or not _TORCH_AVAILABLE:
        return False
    mlp = model_data.get("mlp_model")
    if mlp is None or not model_data.get("trained"):
        return False

    st        = model_data["model"]
    q_emb     = st.encode(question, convert_to_numpy=True).reshape(1, -1)
    sentences = _split_sentences(ideal_text) or [ideal_text.strip()]

    optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()
    loss_scale = max(0.5, 2.0 - reward * 2.0)

    mlp.train()
    for sent in sentences:
        s_emb = st.encode(sent, convert_to_numpy=True).reshape(1, -1)
        feat  = _build_feature_matrix(q_emb, s_emb)
        X     = torch.FloatTensor(feat)
        y     = torch.LongTensor([1])
        for _ in range(n_steps):
            optimizer.zero_grad()
            out  = mlp(X)
            loss = criterion(out, y) * loss_scale
            loss.backward()
            optimizer.step()
    mlp.eval()
    # Keep numpy weights in sync
    model_data["numpy_weights"] = _extract_numpy_weights(mlp)
    return True


# ---------------------------------------------------------------------------
# Main inference entry point
# ---------------------------------------------------------------------------

def generate_persona_response(model_data, persona, question, top_k=2, **kwargs):
    norm_q = _normalize_question(question)

    # Stage 0: RL memory
    memory = _check_learned_memory(model_data, norm_q)
    if memory:
        return _format_response(memory)

    # Stage 1: Text generation (local only — flan-t5 via HF API hallucinates without auth)
    if not _IS_VERCEL:
        try:
            gen = model_data.get("gen_pipeline") or _get_gen_pipeline()
            prompt = (
                f"Answer the following question in one complete, natural sentence "
                f"based only on the person's profile below.\n\n"
                f"Profile: {persona.strip()}\n\n"
                f"Question: {norm_q}\n\n"
                f"Answer:"
            )
            out = gen(prompt, max_new_tokens=120, do_sample=False)
            raw = (out[0].get("generated_text") or "").strip()

            word_count = len(raw.split()) if raw else 0

            if raw and word_count == 1:
                persona_sent = _find_persona_sentence(raw, persona, model_data)
                if persona_sent:
                    raw = persona_sent
                else:
                    expand_prompt = (
                        f"Rewrite the following short answer as one complete, "
                        f"natural sentence that answers the question.\n\n"
                        f"Question: {norm_q}\n"
                        f"Short answer: {raw}\n\n"
                        f"Complete sentence:"
                    )
                    exp     = gen(expand_prompt, max_new_tokens=80, do_sample=False)
                    exp_raw = (exp[0].get("generated_text") or "").strip()
                    if exp_raw and len(exp_raw) > len(raw):
                        raw = exp_raw

            elif raw and 2 <= word_count <= 5:
                expand_prompt = (
                    f"Rewrite the following short answer as one complete, natural "
                    f"sentence that answers the question. Use the exact words from "
                    f"the short answer - do not replace them with synonyms.\n\n"
                    f"Question: {norm_q}\n"
                    f"Short answer: {raw}\n\n"
                    f"Complete sentence:"
                )
                expanded     = gen(expand_prompt, max_new_tokens=80, do_sample=False)
                expanded_raw = (expanded[0].get("generated_text") or "").strip()
                if expanded_raw and len(expanded_raw) > len(raw):
                    raw = expanded_raw

            if raw and len(raw) > 1:
                return _format_response(raw)
        except Exception:
            pass

    # Stage 2: MLP / cosine retrieval (fallback)
    sentences, scores = _mlp_scores(model_data, persona, norm_q)
    if len(sentences) == 0:
        return "I don't have information about that."
    k       = min(top_k, len(sentences))
    top_idx = sorted(np.argsort(scores)[-k:].tolist())
    parts   = [sentences[i].rstrip(".") for i in top_idx]
    return _format_response(". ".join(parts))


# ---------------------------------------------------------------------------
# Scoring & reinforcement
# ---------------------------------------------------------------------------

def compute_reward(bleu_score, rouge1, rougeL):
    return 0.4 * (bleu_score / 100.0) + 0.3 * rouge1 + 0.3 * rougeL


def reinforcement_update(model_data, persona, question, generated_text,
                         ideal_text="", reward=0.0, **kwargs):
    if _IS_VERCEL:
        q_emb = np.zeros(384, dtype=np.float32)   # not used for Vercel memory lookup
    else:
        st    = model_data.get("model")
        q_emb = st.encode(question, convert_to_numpy=True)

    pair = {
        "question":  question,
        "norm_q":    _normalize_question(question),
        "generated": generated_text,
        "ideal":     ideal_text,
        "reward":    reward,
        "q_emb":     q_emb.tolist(),
        "timestamp": time.time(),
    }
    model_data.setdefault("feedback_pairs", []).append(pair)
    model_data["feedback_pairs"] = model_data["feedback_pairs"][-300:]
    model_data.setdefault("score_history", []).append(reward)
    model_data["score_history"] = model_data["score_history"][-50:]

    mlp_updated = _online_update_mlp(model_data, question, ideal_text, reward)

    history = model_data["score_history"]
    trend   = "stable"
    if len(history) >= 3:
        recent = sum(history[-3:]) / 3
        older  = sum(history[-6:-3]) / max(len(history[-6:-3]), 1)
        if recent > older + 0.05:
            trend = "improving"
        elif recent < older - 0.05:
            trend = "needs work"

    _stop = {
        'the','a','an','is','are','was','were','to','of','and','or',
        'in','on','at','for','with','this','that','it','they','their',
        'what','who','how','why','when','where','does','do','can','my',
        'this','person','people',
    }
    gen_kw   = {w for w in re.findall(r'\b[a-z]+\b', generated_text.lower())
                if w not in _stop and len(w) > 3}
    ideal_kw = {w for w in re.findall(r'\b[a-z]+\b', ideal_text.lower())
                if w not in _stop and len(w) > 3}
    missed     = ideal_kw - gen_kw
    missed_str = ", ".join(sorted(missed)[:5]) if missed else "none"

    if reward >= 0.6:
        quality = "Good response"
        action  = ("Memorised as ideal answer. Next time a similar question "
                   "is asked the model will respond with this directly.")
    elif reward >= 0.3:
        quality = "Partial match - key concepts were missing"
        action  = (f"Missed concepts: [{missed_str}]. "
                   f"Model updated - for similar questions it will now include "
                   f"these ideas in the answer.")
    else:
        quality = "Poor response - model learned a new pattern"
        action  = (f"Model's answer had very little overlap with the ideal. "
                   f"Missing concepts: [{missed_str}]. "
                   f"Stored as a strong correction: next time this question type "
                   f"appears the model returns the ideal answer directly.")

    analytics = {
        "reward":        round(reward, 4),
        "quality":       quality,
        "trend":         trend,
        "mlp_updated":   mlp_updated,
        "action":        action,
        "memory_stored": True,
        "total_learned": len(model_data["feedback_pairs"]),
    }

    save_model(model_data)
    return analytics


def score_response(generated, reference):
    if not _SCORING_AVAILABLE or not generated or not reference:
        return {"bleu": 0.0, "rouge1": 0.0, "rougeL": 0.0}
    bleu = BLEU(effective_order=True)
    b    = bleu.sentence_score(generated, [reference]).score
    scorer = _rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r    = scorer.score(reference, generated)
    return {"bleu": b,
            "rouge1": r["rouge1"].fmeasure,
            "rougeL": r["rougeL"].fmeasure}


# ---------------------------------------------------------------------------
# Entry point (local training)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_PATH)
    if not os.path.exists(csv_path):
        print(f"ERROR: {DATA_PATH} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path, encoding="latin1").dropna(subset=["Persona"])
    print(f"Loaded {len(df)} personas from {DATA_PATH}")

    md = load_model()
    md = train_model(md, df)
    save_model(md)
    print("Training complete. Model saved.")
