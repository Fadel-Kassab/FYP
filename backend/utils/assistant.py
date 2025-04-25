import re
import openai
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

def clean_cypher(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text

def format_records(records: list[dict]) -> str:
    """
    Turn a list of dicts into a simple bullet-list of key:value pairs,
    so the LLM sees human-readable text rather than raw JSON.
    """
    if not records:
        return "• No matching data found."
    lines = []
    for row in records:
        parts = []
        for k, v in row.items():
            parts.append(f"{k} = {v}")
        lines.append("• " + "; ".join(parts))
    return "\n".join(lines)

def chat_with_kg(
    model: str = "gpt-4o-mini"
):
    openai_api_key = "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"
    neo4j_uri = "neo4j+s://c703fa4d.databases.neo4j.io"
    neo4j_user = "neo4j" 
    neo4j_password = "JuhfBiYU-pgzF9CpGVhg5AhpkdHWtMPOCWnwpZxX09o" 
    """
    Interactive loop: user asks natural-language questions about patients,
    we RAG against Neo4j and answer with high accuracy and no raw JSON dumps.
    """
    openai.api_key = openai_api_key
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    system_prompt = (
        "You are MedAssist, a medical-graph assistant. Follow this protocol:\n"
        "1) User asks about a patient or medication.\n"
        "2) You generate exactly one Cypher query (no fences) to fetch required data.\n"
        "3) You do NOT return any raw JSON or code to the user.\n"
        "4) You translate the query results into a brief, accurate summary in natural language.\n"
        "5) If data is missing or ambiguous, you clearly say so and suggest consulting a professional."
    )

    print("🩺 MedAssist ready. Ask your question (type 'exit' to quit).\n")
    history = [{"role": "system", "content": system_prompt}]

    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in ("exit", "quit"):
            print("MedAssist: Goodbye!")
            break

        # 1) Generate Cypher
        resp1 = openai.chat.completions.create(
            model=model,
            messages=history + [
                {"role": "user",
                 "content": f"Generate a Cypher query to answer: “{user_q}”. "
                            "Output only the Cypher statement, no fences, no explanation."}
            ]
        )
        cypher_q = clean_cypher(resp1.choices[0].message.content)

        # 2) Execute, swallow errors
        try:
            with driver.session() as session:
                records = session.run(cypher_q).data()
        except Neo4jError:
            records = []

        # 3) Format records into bullet list
        formatted = format_records(records)

        # 4) Summarize for the user
        resp2 = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                    "You are a concise summarizer. "
                    "Given a bullet list of database rows, produce 2–3 sentences "
                    "that capture the key facts in plain English. "
                    "Do not include any JSON or code."},
                {"role": "user", "content": formatted}
            ]
        )
        summary = resp2.choices[0].message.content.strip()

        # 5) Final answer: tie back to the question
        resp3 = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":
                    f"User question: {user_q}\n\n"
                    f"Key facts from the graph:\n{summary}\n\n"
                    "Answer the question clearly, based only on these facts. "
                    "If there is no relevant data, say so and recommend a professional consult."
                }
            ]
        )
        answer = resp3.choices[0].message.content.strip()

        print("MedAssist:", answer, "\n")

        # archive for follow-ups
        history.append({"role": "user",      "content": user_q})
        history.append({"role": "assistant", "content": answer})

    driver.close()