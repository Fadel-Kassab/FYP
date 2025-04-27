import os, openai

# ─── Configuration ─────────────────────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY") or "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"
# Either paste your fine-tuned model here:
FT_MODEL = "ft:gpt-4o-mini:abcd1234-ef56-7890-gh12-ijklmnopqrstuv"
# …or read it from the file written by fine_tune.py:
# FT_MODEL = open("ft_model_name.txt").read().strip()
# ────────────────────────────────────────────────────────────────────────────────

def chat(prompt: str):
    resp = openai.ChatCompletion.create(
        model=FT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )  # Inference :contentReference[oaicite:4]{index=4}
    return resp.choices[0].message.content

if __name__ == "__main__":
    question = input("You: ")
    answer   = chat(question)
    print("Bot:", answer)
