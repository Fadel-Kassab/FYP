import os
import openai
import json
import textwrap
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from typing import List, Dict, Tuple

openai.api_key   = "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"
NEO4J_URI        = "neo4j+s://c703fa4d.databases.neo4j.io"
NEO4J_USERNAME   = "neo4j"
NEO4J_PASSWORD   = "JuhfBiYU-pgzF9CpGVhg5AhpkdHWtMPOCWnwpZxX09o"

def send_to_neo4j(input_file_path: str = "backend/send_files/input_neo4j.txt") -> str:
    """
    1) Reads an unstructured medical record from `input_file_path`.
    2) Uses GPT-4o-mini to parse it into a fixed JSON schema.
    3) Ensures a patient_id field (derived from filename if missing).
    4) Builds & executes one parameterized Cypher that MERGEs Patient and
       all related nodes via UNWIND/WITH blocks.
    5) Returns the actual Cypher text that was run.
    """

    with open(input_file_path, 'r', encoding='utf-8') as f:
        record_text = f.read().strip()

    system_prompt = """
    You are a medical-record parser. Given an unstructured clinical note, output
    a single JSON object (no markdown fences!) with these keys:

    {
    "patient": {
    "patient_id": str, "name": str, "age": int, "gender": str,
    "dob": "YYYY-MM-DD", "date_of_visit": "YYYY-MM-DD",
    "complaints": str
    },
    "conditions":   [ { "name": str } ],
    "allergies":    [ { "name": str, "reaction": str } ],
    "medications":  [ { "name": str, "dosage": str, "frequency": str } ],
    "vitals":       {
    "date": "YYYY-MM-DD", "blood_pressure": str,
    "heart_rate": int, "respiratory_rate": int,
    "temperature": float, "weight": float, "height": float
    },
    "lab_results":  {
    "date": "YYYY-MM-DD", "HbA1c": float,
    "fasting_glucose": int, "creatinine": float,
    "wbc_count": str, "serum_sodium": int
    }
    }
    """
    
    user_prompt = f"Parse this record:\n\n\"\"\"\n{record_text}\n\"\"\""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content": system_prompt},
            {"role":"user",   "content": user_prompt}
        ],
        temperature=0
    )
    
    data = json.loads(resp.choices[0].message.content)

    filename_id = os.path.splitext(os.path.basename(input_file_path))[0]
    data["patient"]["patient_id"] = data["patient"].get("patient_id") or filename_id

    cypher = """
    MERGE (p:Patient {patient_id: $patient.patient_id})
    SET p += $patient

    WITH p
    UNWIND $conditions AS cond
    MERGE (c:Condition {name: cond.name})
    MERGE (p)-[:HAS_CONDITION]->(c)

    WITH p
    UNWIND $allergies AS alg
    MERGE (a:Allergy {name: alg.name})
    SET a.reaction = alg.reaction
    MERGE (p)-[:HAS_ALLERGY]->(a)

    WITH p
    UNWIND $medications AS med
    MERGE (m:Medication {name: med.name})
    SET m.dosage = med.dosage, m.frequency = med.frequency
    MERGE (p)-[:TAKES_MEDICATION]->(m)

    WITH p
    MERGE (v:VitalSign {
        patient_id: p.patient_id,
        date: date($vitals.date)
    })
    SET v += apoc.map.removeKeys($vitals, ["date"])
    MERGE (p)-[:HAS_VITAL_SIGNS]->(v)

    WITH p
    MERGE (l:LabResult {
        patient_id: p.patient_id,
        date: date($lab_results.date)
    })
    SET l += apoc.map.removeKeys($lab_results, ["date"])
    MERGE (p)-[:HAS_LAB_RESULT]->(l)
    """
    
    params = {
        "patient":     data["patient"],
        "conditions":  data["conditions"],
        "allergies":   data["allergies"],
        "medications": data["medications"],
        "vitals":      data["vitals"],
        "lab_results": data["lab_results"]
    }

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        with driver.session() as session:
            session.run(cypher, params)
    finally:
        driver.close()

    return cypher

driver = GraphDatabase.driver(
    "neo4j+s://c703fa4d.databases.neo4j.io",
    auth=("neo4j", "JuhfBiYU-pgzF9CpGVhg5AhpkdHWtMPOCWnwpZxX09o")
)
with driver.session() as sess:
    initial_labels = [r["label"] for r in sess.run("CALL db.labels()")]
    initial_rels   = [r["relationshipType"] for r in sess.run("CALL db.relationshipTypes()")]

# a more permissive system prompt
system_prompt = (
    "You are MedAssist, a friendly yet expert medical-graph assistant.\n"
    f"Your database has these node labels: {', '.join(initial_labels)};\n"
    f"relationship types: {', '.join(initial_rels)}.\n"
    "When asked, you:\n"
    " 1) translate natural questions into Cypher and run them;\n"
    " 2) summarize the raw results;\n"
    " 3) provide an answer that not only reports the facts but also draws inferences,\n"
    "    offers practical recommendations, and engages in a conversational style.\n"
    "Always avoid dumping raw JSON; format as plain English or friendly bullets.\n"
)

# seed history
history: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

def chat_step(
    user_q: str,
    history: List[Dict[str, str]],
    model: str = "gpt-4o-mini"
) -> Tuple[str, List[Dict[str, str]]]:
    """One-shot handler: returns (assistant_reply, updated_history)."""

    # nested helpers
    def clean_cypher(t: str) -> str:
        txt = t.strip()
        if txt.startswith("```"):
            lines = txt.splitlines()[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()
        return txt

    def format_records(rows: List[Dict]) -> str:
        if not rows:
            return "• No matching data found."
        out = []
        for r in rows:
            parts = [f"{k} = {v}" for k, v in r.items()]
            out.append("• " + "; ".join(parts))
        return "\n".join(out)

    def refresh_schema():
        with driver.session() as s:
            lbls = [r["label"] for r in s.run("CALL db.labels()")]
            rels = [r["relationshipType"] for r in s.run("CALL db.relationshipTypes()")]
        return lbls, rels

    # 1) raw-Cypher branch
    if user_q.lower().startswith("cypher:"):
        q = user_q[len("cypher:"):].strip()
        try:
            with driver.session() as s:
                recs = s.run(q).data()
            response = format_records(recs)
        except Neo4jError as e:
            response = f"❗ Cypher error: {e.message}"

    else:
        # 2) generate Cypher
        gen = openai.chat.completions.create(
            model=model,
            messages=history + [{
                "role": "user",
                "content": (
                    f"Generate a Cypher query to answer: “{user_q}”.\n"
                    "Respond with the raw Cypher only, no fences or explanation."
                )
            }]
        )
        cypher_q = clean_cypher(gen.choices[0].message.content)

        # 3) execute (with automatic retry on schema errors)
        try:
            with driver.session() as s:
                recs = s.run(cypher_q).data()
        except Neo4jError as e:
            # refresh schema and ask for fix
            lbls, rels = refresh_schema()
            fix_prompt = (
                f"Your Cypher failed: {e.message}\n"
                f"Current labels: {', '.join(lbls)}; rels: {', '.join(rels)}.\n"
                "Please correct only the Cypher statement."
            )
            fix = openai.chat.completions.create(
                model=model,
                messages=history + [{"role": "user", "content": fix_prompt}]
            )
            cypher_q = clean_cypher(fix.choices[0].message.content)
            with driver.session() as s:
                recs = s.run(cypher_q).data()

        # 4) summarize raw rows
        bullets = format_records(recs)
        summary = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                    "You are a concise summarizer. Turn these bullets into 2–3 plain-English sentences."},
                {"role": "user", "content": bullets}
            ]
        ).choices[0].message.content.strip()

        # 5) final, “loose” conversational answer
        final = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":
                    f"Question: {user_q}\n\nFacts:\n{summary}\n\n"
                    "Now, please answer conversationally—use these facts to draw any inferences,\n"
                    "offer practical recommendations or next steps, and feel free to ask clarifying questions."
                }
            ]
        )
        response = final.choices[0].message.content.strip()

    # 6) update history
    history.append({"role": "user",      "content": user_q})
    history.append({"role": "assistant", "content": response})
    return response, history
