# =============================================================================
# SCRIPT: chat_bart.py
# PURPOSE: Persona-Based Chat with Reinforcement Learning (BLEU + ROUGE Reward)
# Uses lightweight sentence-transformers instead of BART (~80 MB vs ~16 GB).
# =============================================================================

import sys
import os
import textwrap
import pandas as pd

import sacrebleu
from rouge_score import rouge_scorer

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from train_bart import (
    load_model,
    generate_persona_response,
    compute_reward,
    reinforcement_update,
)


# =============================================================================
# Helpers
# =============================================================================

def select_persona(df):
    while True:
        print("\n" + "=" * 40)
        print("PERSONA SELECTION")
        print("1. Pick a Random Persona")
        print("2. Enter a Custom Persona")
        print("=" * 40)
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1" and df is not None:
            persona = df.sample(1).iloc[0].Persona
            print(f"\nPersona Selected:\n{textwrap.fill(persona, 70)}")
            return persona

        elif choice == "2":
            persona = input("\nEnter Persona: ").strip()
            if persona:
                print(f"\nPersona Set:\n{textwrap.fill(persona, 70)}")
                return persona

        print("Invalid input. Try again.")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Load model (always succeeds — no checkpoint required)
    print("Loading model...")
    model_data = load_model()

    try:
        df = pd.read_csv(os.path.join(current_dir, "personality.csv"), encoding="latin1")
    except Exception:
        df = None
        print("Warning: personality.csv not found.")

    print("\n--- Persona-Based Chatbot with Reinforcement Learning ---")
    print("Commands: quit | exit | change persona")
    print("-" * 70)

    current_persona = select_persona(df)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ["quit", "exit"]:
            break

        if user_input.lower() == "change persona":
            current_persona = select_persona(df)
            continue

        if not user_input:
            continue

        # Step 1: Generate response
        bot_reply = generate_persona_response(model_data, current_persona, user_input)
        print(f"\nBot: {bot_reply}")

        # Step 2: Optional ideal response for RL
        ideal_response = input("\nEnter IDEAL Response (or type 'skip'): ").strip()

        if ideal_response.lower() == "skip" or not ideal_response:
            print("Reinforcement skipped.")
            continue

        # Step 3: Evaluate
        bleu   = sacrebleu.corpus_bleu([bot_reply], [[ideal_response]])
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        scores = scorer.score(ideal_response, bot_reply)

        bleu_score = bleu.score
        rouge1     = scores["rouge1"].fmeasure
        rougeL     = scores["rougeL"].fmeasure
        reward     = compute_reward(bleu_score, rouge1, rougeL)

        print("\nEvaluation Scores")
        print(f"  BLEU     : {bleu_score:.2f}")
        print(f"  ROUGE-1  : {rouge1:.2f}")
        print(f"  ROUGE-L  : {rougeL:.2f}")
        print(f"  Reward   : {reward:.4f}")

        # Step 4: RL update (stores feedback, no gradient computation)
        reinforcement_update(
            model_data, current_persona, user_input,
            bot_reply, ideal_text=ideal_response, reward=reward,
        )
        print("Feedback stored. Model will use this for future similar questions.")
        print("-" * 70)
