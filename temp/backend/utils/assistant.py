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
    so it's readable in both raw-query and inference modes.
    """
    if not records:
        return "• No matching data found."
    lines = []
    for row in records:
        parts = [f"{k} = {v}" for k, v in row.items()]
        lines.append("• " + "; ".join(parts))
    return "\n".join(lines)

def get_schema(session):
    labels = [r["label"] for r in session.run("CALL db.labels()")]
    rels   = [r["relationshipType"] for r in session.run("CALL db.relationshipTypes()")]
    return labels, rels

def chat_with_kg(
    model: str = "gpt-4o-mini"
):
    openai.api_key = "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"
    driver = GraphDatabase.driver(
        "neo4j+s://c703fa4d.databases.neo4j.io",
        auth=("neo4j", "JuhfBiYU-pgzF9CpGVhg5AhpkdHWtMPOCWnwpZxX09o")
    )

    # Introspect schema once
    with driver.session() as session:
        labels, rels = get_schema(session)

    system_prompt = (
        "You are MedAssist, a medical-graph assistant.\n"
        f"Node labels: {', '.join(labels)}.\n"
        f"Relationship types: {', '.join(rels)}.\n"
        "Protocol:\n"
        "  • For natural-language questions, generate & run Cypher, summarize, then answer.\n"
        "  • For raw Cypher queries, prefix with 'cypher:' and I'll run them directly.\n"
        "  • Always avoid returning raw JSON; format as bullets or plain English.\n"
    )

    print("🩺 MedAssist ready.")
    print(" • To run a raw Cypher:  cypher: MATCH (p:Patient) RETURN p LIMIT 5")
    print(" • To ask naturally: just type your question\n")

    history = [{"role": "system", "content": system_prompt}]

    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in ("exit", "quit"):
            print("MedAssist: Goodbye!")
            break

        # RAW QUERY MODE
        if user_q.lower().startswith("cypher:"):
            cypher_q = user_q[len("cypher:"):].strip()
            try:
                with driver.session() as session:
                    records = session.run(cypher_q).data()
                print(format_records(records), "\n")
            except Neo4jError as e:
                print(f"❗ Cypher error: {e.message}\n")
            continue

        # INFERENCE MODE
        # 1) Ask LLM to generate Cypher
        resp1 = openai.chat.completions.create(
            model=model,
            messages=history + [{
                "role": "user",
                "content": (
                    f"Generate a Cypher query to answer: “{user_q}”.\n"
                    "Output only the statement, no fences or explanation."
                )
            }]
        )
        cypher_q = clean_cypher(resp1.choices[0].message.content)

        # 2) Execute, with automatic schema-based retry
        try:
            with driver.session() as session:
                records = session.run(cypher_q).data()
        except Neo4jError as e:
            # refresh schema
            with driver.session() as session:
                labels, rels = get_schema(session)
            fix_prompt = (
                f"Your Cypher failed: {e.message}\n"
                f"Labels: {', '.join(labels)}.\n"
                f"Rels: {', '.join(rels)}.\n"
                "Please correct it: only the Cypher, no fences."
            )
            resp_fix = openai.chat.completions.create(
                model=model,
                messages=history + [{"role": "user", "content": fix_prompt}]
            )
            cypher_q = clean_cypher(resp_fix.choices[0].message.content)
            with driver.session() as session:
                records = session.run(cypher_q).data()

        # 3) Format & summarize
        bullets = format_records(records)
        resp2 = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                    "You are a concise summarizer. Given bullets of rows, write 2–3 plain-English sentences."},
                {"role": "user", "content": bullets}
            ]
        )
        summary = resp2.choices[0].message.content.strip()

        # 4) Final answer
        resp3 = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":
                    f"Question: {user_q}\n\nFacts:\n{summary}\n\n"
                    "Answer using only these facts; if none, say so and recommend consulting a professional."
                }
            ]
        )
        answer = resp3.choices[0].message.content.strip()
        print("MedAssist:", answer, "\n")

        history.append({"role": "user",    "content": user_q})
        history.append({"role": "assistant", "content": answer})

    driver.close()

if __name__ == "__main__":
    chat_with_kg()
