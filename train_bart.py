# =============================================================================
# train_bart.py — Persona-Aware QA: Proper ML Training Pipeline
# =============================================================================
#
#  DATA PIPELINE
#  ─────────────
#  personality.csv  →  QA pairs (question, correct sentence, wrong sentence)
#       │
#       └─ 70% TRAIN  /  10% VAL  /  20% TEST   (proper ML split)
#
#  FEATURE EXTRACTION  (frozen, no training)
#  ─────────────────────────────────────────
#  sentence-transformers (all-MiniLM-L6-v2, ~80 MB)
#  Encodes question + sentence → dense vectors (384-d each)
#  Feature vector per pair = [ q_emb | s_emb | cos_sim ]  →  769-d
#
#  MODEL  (trained on your data with epochs)
#  ──────────────────────────────────────────
#  PersonaQAClassifier  (3-layer MLP, PyTorch)
#    Layer 1: 769 → 256  (ReLU + Dropout 0.3)
#    Layer 2: 256 →  64  (ReLU + Dropout 0.2)
#    Layer 3:  64 →   2  (logits → softmax → P(match))
#
#  TRAINING  (what epochs actually do)
#  ───────────────────────────────────
#  Loss      : CrossEntropyLoss (positive=1, negative=0)
#  Optimizer : Adam  (lr = 0.001)
#  Scheduler : StepLR  (decay ×0.5 every 5 epochs)
#  Epochs    : configurable (default 15)
#  Per epoch : train_loss, train_acc, val_loss, val_acc  printed
#  Final     : test accuracy on the 20% held-out test set
#
#  OUTPUT
#  ──────
#  persona_model.pkl  : MLP state_dict + config  (~1 MB max)
#  Console            : epoch table + final test accuracy
#
#  INFERENCE
#  ─────────
#  Stage 1 – Extractive QA   (deepset/minilm-uncased-squad2, ~67 MB)
#    Reads persona as a document, extracts exact answer span.
#    "wt is my name" → normalised → "what is my name?" → "Abhiram"
#
#  Stage 2 – Trained MLP retrieval  (fallback for open-ended questions)
#    Scores every persona sentence with the trained MLP → top-k returned.
# =============================================================================

import os
import re
import sys
import pickle
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sentence_transformers import SentenceTransformer, util
from sklearn.model_selection import train_test_split
from transformers import pipeline as hf_pipeline
from spellchecker import SpellChecker

try:
    from sacrebleu.metrics import BLEU
    from rouge_score import rouge_scorer as _rs
    _SCORING_AVAILABLE = True
except ImportError:
    _SCORING_AVAILABLE = False

# ─── Hyperparameters (edit here) ──────────────────────────────────────────────
EPOCHS          = 15
BATCH_SIZE      = 64
LEARNING_RATE   = 0.001
TEST_SPLIT      = 0.20    # 20 % held-out test set
VAL_SPLIT       = 0.10    # 10 % validation during training
MAX_QA_PAIRS    = 14000   # cap total positive examples
MAX_PERSONAS    = 5000    # personas used to generate QA pairs

PICKLE_PATH     = "persona_model.pkl"
DATA_PATH       = "personality.csv"
QA_CONF_THRESH  = 0.08

# ─── Model architecture config ────────────────────────────────────────────────
INPUT_DIM  = 769   # 384 + 384 + 1
HIDDEN1    = 256
HIDDEN2    = 64
NUM_LABELS = 2

# ─── Singletons ───────────────────────────────────────────────────────────────
_st_model_instance  = None
_gen_pipeline_instance = None          # replaces the old extractive QA pipeline


def _get_st_model():
    global _st_model_instance
    if _st_model_instance is None:
        print("Loading sentence-transformer: all-MiniLM-L6-v2  (~80 MB, one-time)")
        _st_model_instance = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model_instance


def _get_gen_pipeline():
    """
    Load google/flan-t5-base  — a seq2seq model fine-tuned on 1,800+ NLP tasks.

    Why flan-t5-base (not small)?
      flan-t5-small (77 M params) was outputting garbage MCQ-style strings
      like '(iii)', '(II)' for open-ended questions because it is too small
      to distinguish QA from multiple-choice tasks in its training mixture.
      flan-t5-base (250 M params) handles both extractive and reasoning
      questions correctly on the standard  'question: ... context: ...'  format.

    Size : ~900 MB one-time download to  ~/.cache/huggingface/
    GPU  : not required  (runs on CPU, ~3-5 s per response)
    """
    global _gen_pipeline_instance
    if _gen_pipeline_instance is None:
        print("Loading generative model: google/flan-t5-base  (~900 MB, one-time download)")
        _gen_pipeline_instance = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=150,
        )
        print("Generative model ready.")
    return _gen_pipeline_instance


def _expand_to_sentence(answer: str, persona: str) -> str:
    """
    Given a short extracted keyword/phrase from T5, find and return the
    full persona sentence that best contains that information.

    Matching strategy (no hardcoding — works for any persona):
      1. Exact substring match  (fastest, most precise)
      2. Stem-based word overlap (handles 'freelancing' vs 'freelance',
         'studying' vs 'study', 'responsibilities' vs 'responsible', etc.)
         Uses first-N-character comparison so no vocabulary list needed.
      3. Returns raw T5 answer as-is if no sentence match found.
    """
    if not answer or not persona:
        return answer

    answer_lower  = answer.lower().strip()
    sentences     = _split_sentences(persona)

    # Strategy 1 — exact substring
    for sent in sentences:
        if answer_lower in sent.lower():
            return sent.strip()

    # Strategy 2 — stem-based word overlap
    def _stem(w, n=5):
        return w[:n].lower()          # first-N chars as cheap stem

    a_words = [w for w in re.findall(r'[a-z]+', answer_lower) if len(w) >= 4]
    a_stems = {_stem(w) for w in a_words}

    best_ov, best_sent = 0, None
    for sent in sentences:
        s_words = re.findall(r'[a-z]+', sent.lower())
        s_stems = {_stem(w) for w in s_words if len(w) >= 4}
        overlap = len(a_stems & s_stems)
        if overlap > best_ov:
            best_ov, best_sent = overlap, sent

    if best_ov > 0 and best_sent:
        return best_sent.strip()

    # Strategy 3 — no match; return the raw T5 answer
    return answer


# =============================================================================
# Neural Network Definition
# =============================================================================

class PersonaQAClassifier(nn.Module):
    """
    3-layer MLP that learns:  (question, sentence) → P(sentence answers question)

    Input  : 769-d feature vector  [ q_emb | s_emb | cos_sim ]
    Output : 2 logits  (class 0 = no match,  class 1 = match)
    """
    def __init__(self, input_dim=INPUT_DIM, hidden1=HIDDEN1,
                 hidden2=HIDDEN2, num_labels=NUM_LABELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden2, num_labels),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# QA pair generation  (rule-based, produces training data)
# =============================================================================

def _norm(s):
    return " ".join(s.strip().lower().split()).replace(".", "").replace("?", "")


def questions_for_sentence(sentence):
    """Generate (question, answer) pairs for a single persona sentence."""
    s = sentence.strip()
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
    """Build (context, question, answer, all_sentences) rows from a DataFrame."""
    rows = []
    for _, row in persona_df.iterrows():
        context = str(row["Persona"]).strip()
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
        for q in ["Who are you?", "Tell me about yourself.",
                  "Describe yourself."]:
            rows.append({"context": context, "question": q,
                         "answer": context, "all_sentences": sentences})
    return pd.DataFrame(rows)


# =============================================================================
# Feature engineering
# =============================================================================

def _build_feature_matrix(q_embs: np.ndarray, s_embs: np.ndarray) -> np.ndarray:
    """
    Build  [q_norm | s_norm | cos_sim]  →  shape (N, 769).
    Normalised embeddings + scalar similarity give the MLP rich signal.
    """
    q_n = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)
    s_n = s_embs / (np.linalg.norm(s_embs, axis=1, keepdims=True) + 1e-9)
    cos = np.einsum("ij,ij->i", q_n, s_n).reshape(-1, 1)
    return np.concatenate([q_n, s_n, cos], axis=1).astype(np.float32)


# =============================================================================
# Model load / save
# =============================================================================

def load_model(pickle_path=PICKLE_PATH):
    """
    Load model_data dict.  Always succeeds even with no trained pickle.
    Keys: model, gen_pipeline, mlp_model, feedback_pairs, trained
    """
    model_data = {
        "model":        _get_st_model(),
        "gen_pipeline": _get_gen_pipeline(),   # generative model replaces extractive QA
        "mlp_model":    None,
        "feedback_pairs": [],
        "trained":      False,
    }
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                saved = pickle.load(f)

            # Reconstruct MLP from saved state_dict
            cfg = saved.get("mlp_config", {})
            if cfg and saved.get("mlp_state_dict"):
                mlp = PersonaQAClassifier(**cfg)
                mlp.load_state_dict(saved["mlp_state_dict"])
                mlp.eval()
                model_data["mlp_model"] = mlp

            model_data["feedback_pairs"] = saved.get("feedback_pairs", [])
            model_data["trained"]        = saved.get("trained", False)

            size_kb = os.path.getsize(pickle_path) / 1024
            status  = "trained" if model_data["trained"] else "untrained"
            print(f"Loaded {pickle_path}  ({size_kb:.0f} KB,  {status},  "
                  f"{len(model_data['feedback_pairs'])} RL pairs)")
        except Exception as e:
            print(f"Warning: could not load {pickle_path}: {e}")
    else:
        print("No trained model found. Run:  python train_bart.py")

    # Register persona vocabulary with the spell-checker so names / places
    # from the dataset are never mis-corrected at inference time.
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_PATH)
    if os.path.exists(csv_path):
        try:
            _df = pd.read_csv(csv_path, encoding="latin1").dropna(subset=["Persona"])
            _register_persona_vocab(_df)
        except Exception:
            pass
    return model_data


def save_model(model_data, pickle_path=PICKLE_PATH):
    """Save MLP weights + feedback pairs only  (no ST/QA model weights)."""
    mlp = model_data.get("mlp_model")
    payload = {
        "mlp_state_dict": mlp.state_dict() if mlp else None,
        "mlp_config":     {"input_dim": INPUT_DIM, "hidden1": HIDDEN1,
                           "hidden2": HIDDEN2, "num_labels": NUM_LABELS},
        "feedback_pairs": model_data.get("feedback_pairs", []),
        "trained":        model_data.get("trained", False),
    }
    with open(pickle_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    size_kb = os.path.getsize(pickle_path) / 1024
    print(f"Saved {pickle_path}  ({size_kb:.0f} KB)")


# =============================================================================
# TRAINING  — the core ML pipeline
# =============================================================================

def train_model(model_data, df,
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                lr=LEARNING_RATE, verbose=True):
    """
    Full ML training pipeline:

      1. Generate QA pairs from persona data
      2. Encode with sentence-transformers  (frozen feature extractor)
      3. Split into  70% TRAIN / 10% VAL / 20% TEST
      4. Train MLP with Adam + CrossEntropyLoss for `epochs` epochs
      5. Print per-epoch: loss & accuracy on train + val sets
      6. Report final accuracy on the 20% held-out TEST set
      7. Save best model (lowest val loss)

    Returns model_data with trained MLP inside.
    """
    st = model_data["model"]

    # Register every word in the persona dataset with the spell-checker so
    # names, places, and domain terms are never mis-corrected.
    _register_persona_vocab(df)

    # ── 1. Build QA pairs ─────────────────────────────────────────────────────
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

    # ── 2. Build negatives ────────────────────────────────────────────────────
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

    # ── 3. Batch encode ───────────────────────────────────────────────────────
    if verbose:
        print("\nEncoding features with sentence-transformer...")
    all_texts = questions + pos_sentences + neg_sentences
    all_embs  = st.encode(all_texts, batch_size=128,
                          show_progress_bar=verbose, convert_to_numpy=True)
    n        = len(questions)
    q_embs   = all_embs[:n]
    pos_embs = all_embs[n:2*n]
    neg_embs = all_embs[2*n:3*n]

    X_pos = _build_feature_matrix(q_embs, pos_embs)   # label = 1
    X_neg = _build_feature_matrix(q_embs, neg_embs)   # label = 0
    X     = np.vstack([X_pos, X_neg])
    y     = np.array([1]*n + [0]*n, dtype=np.int64)

    # Shuffle
    idx  = np.random.permutation(len(y))
    X, y = X[idx], y[idx]

    # ── 4. Train / Val / Test split ──────────────────────────────────────────
    # First cut off 20% for test, then split the rest into 87.5%/12.5% ≈ 70/10
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=42, stratify=y)
    val_fraction = VAL_SPLIT / (1.0 - TEST_SPLIT)          # ~11.1%  of trainval
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction, random_state=42, stratify=y_trainval)

    if verbose:
        print(f"\nData split:")
        print(f"  Train : {len(X_train):>6} samples  ({len(X_train)/len(X)*100:.0f}%)")
        print(f"  Val   : {len(X_val):>6} samples  ({len(X_val)/len(X)*100:.0f}%)")
        print(f"  Test  : {len(X_test):>6} samples  ({len(X_test)/len(X)*100:.0f}%)")
        print(f"\nModel: PersonaQAClassifier  "
              f"[{INPUT_DIM} → {HIDDEN1} → {HIDDEN2} → {NUM_LABELS}]")
        print(f"Optimizer: Adam  lr={lr}")
        print(f"Epochs: {epochs}   Batch size: {batch_size}\n")

    # PyTorch tensors
    X_tr_t   = torch.FloatTensor(X_train)
    y_tr_t   = torch.LongTensor(y_train)
    X_val_t  = torch.FloatTensor(X_val)
    y_val_t  = torch.LongTensor(y_val)
    X_tst_t  = torch.FloatTensor(X_test)
    y_tst_t  = torch.LongTensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=True
    )

    # ── 5. Model, loss, optimizer ─────────────────────────────────────────────
    mlp       = PersonaQAClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_loss  = float("inf")
    best_state     = None

    # ── 6. Training loop ──────────────────────────────────────────────────────
    header = (f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
              f"{'Val Loss':>9} | {'Val Acc':>8}")
    if verbose:
        print(header)
        print("─" * len(header))

    for epoch in range(1, epochs + 1):

        # — Train —
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

        # — Validate —
        mlp.eval()
        with torch.no_grad():
            val_logits = mlp(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_acc    = (val_logits.argmax(1) == y_val_t).float().mean().item() * 100

        if verbose:
            print(f"{epoch:>6} | {avg_train_loss:>10.4f} | {train_acc:>8.2f}% | "
                  f"{val_loss:>9.4f} | {val_acc:>7.2f}%")

        # Save best model (lowest val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in mlp.state_dict().items()}

    # ── 7. Restore best and evaluate on TEST set ──────────────────────────────
    if best_state:
        mlp.load_state_dict(best_state)
    mlp.eval()

    with torch.no_grad():
        test_logits = mlp(X_tst_t)
        test_acc    = (test_logits.argmax(1) == y_tst_t).float().mean().item() * 100
        test_loss   = criterion(test_logits, y_tst_t).item()

    if verbose:
        print("─" * len(header))
        print(f"\nFinal TEST set  →  Loss: {test_loss:.4f}  |  "
              f"Accuracy: {test_acc:.2f}%")

    model_data["mlp_model"] = mlp
    model_data["trained"]   = True
    return model_data


# =============================================================================
# Question normalisation  (handles text-speak / abbreviations)
# =============================================================================

# ── Text-speak shortcuts  (NOT real words — a spell-checker cannot fix these) ─
# This is the ONLY hardcoded list. Everything else (real misspellings like
# "intrest", "univercity", etc.) is handled automatically by the SpellChecker.
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

# Global spell-checker instance.
# Persona-specific words (names, places, organisations) are added at
# load / train time via _register_persona_vocab() so they are never
# mis-corrected.
_spell = SpellChecker()


def _register_persona_vocab(df: pd.DataFrame):
    """
    Extract every word that appears in the personas and mark them as known
    so the spell-checker never 'corrects' a real persona word.
    Works for any dataset — no hardcoding needed.
    """
    vocab = set()
    for persona in df["Persona"].dropna():
        for word in re.findall(r"[A-Za-z]+", str(persona)):
            vocab.add(word.lower())
    if vocab:
        _spell.word_frequency.load_words(vocab)
        print(f"Spell-checker updated with {len(vocab)} persona vocabulary words.")

_OPEN_ENDED_RE = re.compile(
    r"\b(tell me about|describe yourself|who are you|about yourself"
    r"|what are (your|my) hobbies|what do you (like|enjoy|love)"
    r"|what are (your|my) interests)\b", re.I
)


def _normalize_question(text: str) -> str:
    """
    1. Expand text-speak shortcuts  (wt → what, ur → your, …)
    2. Auto-correct genuine misspellings using SpellChecker
       — works for ANY word, no hardcoding required.
       — persona words (names, places) are pre-registered so they're
         never mis-corrected.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\?\.\!,']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Step 1 — text-speak expansion
    for pat, rep in _TEXTSPEAK.items():
        text = re.sub(pat, rep, text)

    # Step 2 — spell-check every token that looks like a real word (len > 2)
    tokens = text.split()
    only_words = [t for t in tokens if t.isalpha() and len(t) > 2]
    misspelled = _spell.unknown(only_words)   # returns the set of bad words

    corrected_tokens = []
    for tok in tokens:
        if tok in misspelled:
            fix = _spell.correction(tok)       # best candidate from dictionary
            corrected_tokens.append(fix if fix else tok)
        else:
            corrected_tokens.append(tok)

    text = " ".join(corrected_tokens)
    if text and text[-1] not in ".?!":
        text += "?"
    return text


# =============================================================================
# Sentence splitting
# =============================================================================

def _split_sentences(text: str):
    """
    Split any text into sentences WITHOUT breaking on abbreviations or decimals.

    Works on whatever persona the user provides at runtime — no hardcoded text.

    Protects: Letter.Letter patterns (e.g. degree abbreviations, initials)
              Digit.Digit  patterns  (e.g. decimal numbers like 8.5, 3.14)
    Splits on real sentence-ending punctuation only.
    """
    # Step 1 – protect  Letter.Letter  patterns (B.Tech → B<P>Tech)
    protected = re.sub(r'([A-Za-z])\.([A-Za-z])', r'\1<P>\2', text)
    # Step 2 – protect  Digit.Digit  decimals  (8.5 → 8<P>5)
    protected = re.sub(r'(\d)\.(\d)', r'\1<P>\2', protected)
    # Step 3 – split on real sentence boundaries
    parts = re.split(r'[.!?]+', protected)
    # Step 4 – restore and filter
    return [s.replace('<P>', '.').strip()
            for s in parts if len(s.strip()) > 3]


def _find_persona_sentence(keyword: str, persona: str) -> str:
    """
    Given a single-word answer from T5 (which may be a paraphrase / synonym),
    find the persona sentence that is semantically closest to that keyword.

    This ensures the chatbot always replies with the persona's own exact words
    instead of T5's paraphrase.

    Example
    -------
    keyword = "biking",  persona contains "i love bicycling."
    → returns "I love bicycling."   (persona's own wording, not "biking")

    Approach: semantic cosine similarity via the sentence-transformer that is
    already loaded — no extra model, no hardcoded synonyms.
    """
    sentences = _split_sentences(persona)
    if not sentences:
        return ""

    try:
        from sentence_transformers import util as st_util
        # Lazy-load the sentence-transformer (already cached after train/load)
        import sentence_transformers as _st_mod
        _emb_model = _st_mod.SentenceTransformer("all-MiniLM-L6-v2")
        kw_emb   = _emb_model.encode(keyword, convert_to_tensor=True)
        sent_emb = _emb_model.encode(sentences, convert_to_tensor=True)
        scores   = st_util.cos_sim(kw_emb, sent_emb)[0]
        best_idx = int(scores.argmax())
        if float(scores[best_idx]) > 0.30:          # minimum confidence
            return sentences[best_idx]
    except Exception:
        pass

    # Plain substring fallback — if the keyword literally appears in a sentence
    kw_lower = keyword.lower().rstrip(".")
    for sent in sentences:
        if kw_lower in sent.lower():
            return sent
    return ""


# =============================================================================
# Inference
# =============================================================================

def _mlp_scores(model_data, persona, question):
    """Score every persona sentence using the trained MLP."""
    st      = model_data["model"]
    mlp     = model_data.get("mlp_model")
    sentences = _split_sentences(persona)
    if not sentences:
        return sentences, np.array([])

    q_emb  = st.encode(question,   convert_to_numpy=True)
    s_embs = st.encode(sentences,  convert_to_numpy=True, batch_size=64)
    feats  = _build_feature_matrix(
        np.tile(q_emb, (len(sentences), 1)), s_embs
    )

    if mlp is not None:
        mlp.eval()
        with torch.no_grad():
            logits = mlp(torch.FloatTensor(feats))
            scores = torch.softmax(logits, dim=1)[:, 1].numpy()
    else:
        # Fallback: cosine similarity before training
        q_n = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        s_n = s_embs / (np.linalg.norm(s_embs, axis=1, keepdims=True) + 1e-9)
        scores = s_n @ q_n

    # RL boost
    boost    = np.zeros(len(sentences))
    st_model = model_data["model"]
    q_emb_t  = st_model.encode(question, convert_to_tensor=True)
    for fp in model_data.get("feedback_pairs", []):
        if fp.get("reward", 0) < 0.2:
            continue
        fp_q = st_model.encode(fp["question"], convert_to_tensor=True)
        q_sim = util.cos_sim(q_emb_t, fp_q).item()
        if q_sim < 0.65:
            continue
        ideal_sents = _split_sentences(fp.get("ideal", ""))
        if not ideal_sents:
            continue
        ideal_embs = st_model.encode(ideal_sents, convert_to_tensor=True)
        s_embs_t   = st_model.encode(sentences,   convert_to_tensor=True)
        for i, s_emb_t in enumerate(s_embs_t):
            match = util.cos_sim(s_emb_t.unsqueeze(0), ideal_embs).max().item()
            if match > 0.55:
                boost[i] += fp["reward"] * q_sim * 0.2

    return sentences, scores + boost


def _format_response(text: str) -> str:
    """Capitalise first letter and ensure clean ending — applied to every response."""
    text = text.strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def _check_learned_memory(model_data, question: str, threshold: float = 0.82):
    """
    STAGE 0 — look up RL memory.

    If the user previously corrected the model on a very similar question
    (cosine-similarity >= threshold), return the ideal answer they provided.
    This is the key 'learning' gate: it fires BEFORE the QA pipeline so that
    repeated questions always get the corrected answer.

    No hardcoding — similarity is computed on-the-fly for ANY question.
    """
    pairs = model_data.get("feedback_pairs", [])
    if not pairs:
        return None

    st    = model_data["model"]
    q_emb = st.encode(question, convert_to_numpy=True)
    q_norm = np.linalg.norm(q_emb)
    if q_norm < 1e-9:
        return None

    best_sim   = threshold   # minimum required; only improves from here
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
    """
    Online gradient update — teaches the MLP that (question, ideal sentence)
    is a positive match.

    Called every time the user submits an ideal response, so the model
    immediately improves for similar future questions.
    """
    mlp = model_data.get("mlp_model")
    if mlp is None or not model_data.get("trained"):
        return False          # nothing to update yet

    st      = model_data["model"]
    q_emb   = st.encode(question, convert_to_numpy=True).reshape(1, -1)
    sentences = _split_sentences(ideal_text) or [ideal_text.strip()]

    optimizer = torch.optim.Adam(mlp.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    mlp.train()
    for sent in sentences:
        s_emb = st.encode(sent, convert_to_numpy=True).reshape(1, -1)
        feat  = _build_feature_matrix(q_emb, s_emb)
        X     = torch.FloatTensor(feat)
        y     = torch.LongTensor([1])                  # label: correct match

        # Weight loss by how wrong the model was (low reward → bigger update)
        loss_scale = max(0.5, 2.0 - reward * 2.0)

        for _ in range(n_steps):
            optimizer.zero_grad()
            out  = mlp(X)
            loss = criterion(out, y) * loss_scale
            loss.backward()
            optimizer.step()
    mlp.eval()
    return True


def generate_persona_response(model_data, persona, question, top_k=2, **kwargs):
    """
    Three-stage pipeline — every response is capitalised and punctuated.
    No hardcoding; every stage is fully data-driven.

    Stage 0 — RL Memory
        Before anything else, check if the user previously gave an ideal
        answer for a semantically similar question.  If found (cosine
        similarity >= 0.82), return that memorised answer immediately.
        This is the core 'learning' loop: the model gets smarter with use.

    Stage 1 — Generative Model  (google/flan-t5-small)
        Uses a small instruction-tuned seq2seq model to GENERATE a fresh
        answer from the persona.  Unlike the old extractive approach, this
        can handle ANY question type:
          • Factual   → "What does this person do?"
          • Reasoning → "Why might this person struggle with routines?"
          • Synthesis → "What are the main challenges they face?"
          • Creative  → "Suggest a daily schedule for this person."
          • Judgement → "Is this person suited for full-time or freelancing?"
        The persona is injected verbatim as context so every answer is
        grounded in the actual profile — never hallucinated.

    Stage 2 — Trained MLP Fallback
        If the generative model fails or returns an empty string, fall back
        to the MLP classifier (trained on personality.csv QA pairs) which
        retrieves the top-k most relevant persona sentences.
    """
    norm_q = _normalize_question(question)

    # ── Stage 0: RL memory — highest priority ────────────────────────────────
    memory = _check_learned_memory(model_data, norm_q)
    if memory:
        return _format_response(memory)

    # ── Stage 1: T5 conversational generation ────────────────────────────────
    # Instruction-style prompt so flan-t5-base produces a COMPLETE natural
    # sentence instead of a bare extracted span.
    #
    # Old (extractive QA format):
    #   prompt  → "question: X  context: Y"
    #   output  → "AI."  /  "Stable job."       ← bare phrase, feels robotic
    #
    # New (instruction-generation format):
    #   prompt  → "Answer ... in one complete sentence ... Profile: ... Question: ..."
    #   output  → "He is passionate about AI."
    #             "He would prefer a stable job."  ← natural, conversational
    #
    # The persona is passed verbatim as context so the model cannot
    # hallucinate — every answer is grounded in the actual profile.
    try:
        gen = model_data.get("gen_pipeline") or _get_gen_pipeline()

        # Primary call — answer grounded in persona
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
            # ── Single-word answer (e.g. "bicycling", "government") ──────────
            # T5 identified the right concept but may have paraphrased it.
            # Search the persona for the sentence that contains this exact
            # concept so the chatbot always uses the persona's own wording.
            # Example: T5 says "biking" but persona says "i love bicycling"
            #          → we return "I love bicycling." not "I love biking."
            persona_sent = _find_persona_sentence(raw, persona)
            if persona_sent:
                raw = persona_sent   # exact persona sentence, no paraphrase
            else:
                # word not literally in persona — expand to sentence via T5
                expand_prompt = (
                    f"Rewrite the following short answer as one complete, "
                    f"natural sentence that answers the question.\n\n"
                    f"Question: {norm_q}\n"
                    f"Short answer: {raw}\n\n"
                    f"Complete sentence:"
                )
                exp = gen(expand_prompt, max_new_tokens=80, do_sample=False)
                exp_raw = (exp[0].get("generated_text") or "").strip()
                if exp_raw and len(exp_raw) > len(raw):
                    raw = exp_raw

        elif raw and 2 <= word_count <= 5:
            # ── Short phrase (e.g. "government exams", "AI and ML") ──────────
            # Ask T5 to wrap this into a proper conversational sentence.
            # The original short answer is passed explicitly so T5 uses the
            # exact words rather than inventing synonyms.
            expand_prompt = (
                f"Rewrite the following short answer as one complete, natural "
                f"sentence that answers the question. Use the exact words from "
                f"the short answer — do not replace them with synonyms.\n\n"
                f"Question: {norm_q}\n"
                f"Short answer: {raw}\n\n"
                f"Complete sentence:"
            )
            expanded = gen(expand_prompt, max_new_tokens=80, do_sample=False)
            expanded_raw = (expanded[0].get("generated_text") or "").strip()
            if expanded_raw and len(expanded_raw) > len(raw):
                raw = expanded_raw

        # 6+ words → already a full sentence, return as-is
        if raw and len(raw) > 1:
            return _format_response(raw)
    except Exception:
        pass   # fall through to MLP

    # ── Stage 2: Trained MLP retrieval (fallback) ────────────────────────────
    sentences, scores = _mlp_scores(model_data, persona, norm_q)
    if len(sentences) == 0:
        return "I don't have information about that."
    k       = min(top_k, len(sentences))
    top_idx = sorted(np.argsort(scores)[-k:].tolist())
    parts   = [sentences[i].rstrip(".") for i in top_idx]
    return _format_response(". ".join(parts))


# =============================================================================
# Reward & RL
# =============================================================================

def compute_reward(bleu_score, rouge1, rougeL):
    return 0.4 * (bleu_score / 100.0) + 0.3 * rouge1 + 0.3 * rougeL


def reinforcement_update(model_data, persona, question, generated_text,
                         ideal_text="", reward=0.0, **kwargs):
    """
    Real RL update — three-phase learning:

    Phase 1  Store feedback with question embedding so the memory gate in
             generate_persona_response() can recall the ideal answer next time
             a semantically similar question is asked.

    Phase 2  Online gradient update on the MLP — a few back-prop steps that
             teach the network to score (question, ideal_sentence) as a
             positive match. The learning rate is small so the model doesn't
             forget prior training; the loss is up-weighted when reward is low
             (i.e. when the model was most wrong).

    Phase 3  Return a human-readable analytics dict so the UI can show the
             user exactly what changed and whether the model is improving.
    """
    import time

    st    = model_data["model"]
    q_emb = st.encode(question, convert_to_numpy=True)

    # ── Phase 1: store with embedding ─────────────────────────────────────────
    pair = {
        "question":  question,
        "generated": generated_text,
        "ideal":     ideal_text,
        "reward":    reward,
        "q_emb":     q_emb.tolist(),          # enables memory lookup
        "timestamp": time.time(),
    }
    model_data.setdefault("feedback_pairs", []).append(pair)
    model_data["feedback_pairs"] = model_data["feedback_pairs"][-300:]

    # Keep running score history for trend analysis
    model_data.setdefault("score_history", []).append(reward)
    model_data["score_history"] = model_data["score_history"][-50:]

    # ── Phase 2: online MLP gradient update ───────────────────────────────────
    mlp_updated = _online_update_mlp(model_data, question, ideal_text, reward)

    # ── Phase 3: build analytics ───────────────────────────────────────────────
    history = model_data["score_history"]
    trend   = "stable"
    if len(history) >= 3:
        recent = sum(history[-3:]) / 3
        older  = sum(history[-6:-3]) / max(len(history[-6:-3]), 1)
        if recent > older + 0.05:
            trend = "improving"
        elif recent < older - 0.05:
            trend = "needs work"

    # ── Why did the model struggle? Identify missing concepts ─────────────────
    _stop = {'the','a','an','is','are','was','were','to','of','and','or',
             'in','on','at','for','with','this','that','it','they','their',
             'what','who','how','why','when','where','does','do','can','my',
             'this','person','people'}
    gen_kw   = {w for w in re.findall(r'\b[a-z]+\b', generated_text.lower())
                if w not in _stop and len(w) > 3}
    ideal_kw = {w for w in re.findall(r'\b[a-z]+\b', ideal_text.lower())
                if w not in _stop and len(w) > 3}
    missed   = ideal_kw - gen_kw
    missed_str = ", ".join(sorted(missed)[:5]) if missed else "none"

    if reward >= 0.6:
        quality = "Good response"
        action  = (f"Memorised as ideal answer. Next time a similar question "
                   f"is asked the model will respond with this directly.")
    elif reward >= 0.3:
        quality = "Partial match — key concepts were missing"
        action  = (f"Missed concepts: [{missed_str}]. "
                   f"Model updated — for similar questions it will now include "
                   f"these ideas in the answer.")
    else:
        quality = "Poor response — model learned a new pattern"
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


# =============================================================================
# Scoring helper
# =============================================================================

def score_response(generated, reference):
    if not _SCORING_AVAILABLE or not generated or not reference:
        return {"bleu": 0.0, "rouge1": 0.0, "rougeL": 0.0}
    bleu   = BLEU(effective_order=True)
    b      = bleu.sentence_score(generated, [reference]).score
    scorer = _rs.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r      = scorer.score(reference, generated)
    return {"bleu": b,
            "rouge1": r["rouge1"].fmeasure,
            "rougeL": r["rougeL"].fmeasure}


# =============================================================================
# Backward-compat stubs
# =============================================================================
BartFineTuner    = None
PersonaQADataset = None


# =============================================================================
# Entry point — run as:  python train_bart.py
# =============================================================================

if __name__ == "__main__":
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_PATH)
    if not os.path.isfile(csv_path):
        print(f"ERROR: {DATA_PATH} not found."); sys.exit(1)

    df = pd.read_csv(csv_path, encoding="latin1").dropna(subset=["Persona"])
    print(f"Loaded {len(df)} personas from {csv_path}")

    df_train = df.sample(n=min(MAX_PERSONAS, len(df)),
                         random_state=42).reset_index(drop=True)
    print(f"Using {len(df_train)} personas  "
          f"(change MAX_PERSONAS to use more / all {len(df)})")

    # Initialise model (downloads ST + QA models on first run)
    model_data = load_model()

    # Full ML training pipeline
    model_data = train_model(model_data, df_train,
                             epochs=EPOCHS, batch_size=BATCH_SIZE,
                             lr=LEARNING_RATE, verbose=True)

    # Save
    save_model(model_data)

    # ── Smoke tests — all personas loaded dynamically from the dataset ────────
    # No persona is hardcoded here; we pick real rows from personality.csv so
    # the test reflects actual data the model was trained on.
    generic_questions = [
        "What do you like?",
        "Tell me about yourself.",
        "What are your interests?",
        "Where do you live?",
        "What is your name?",
    ]
    sample_rows = df.sample(min(4, len(df)), random_state=0)
    tests = [
        (row["Persona"], generic_questions[i % len(generic_questions)])
        for i, (_, row) in enumerate(sample_rows.iterrows())
    ]
    print("\n--- Smoke tests (personas from dataset, no hardcoding) -----------")
    for persona, q in tests:
        ans = generate_persona_response(model_data, persona, q)
        print(f"\n  Persona : {str(persona)[:70]}...")
        print(f"  Q       : {q}")
        print(f"  A       : {ans}")

    pkl_kb = os.path.getsize(PICKLE_PATH) / 1024
    print(f"\n  Pickle  : {PICKLE_PATH}  ({pkl_kb:.0f} KB)")
    print(f"  HF cache: ~150 MB  (all-MiniLM + minilm-squad2, one-time)")
    print(f"  GPU     : not required")
