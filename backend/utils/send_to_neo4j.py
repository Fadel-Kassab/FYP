import os
import json
import openai
from neo4j import GraphDatabase

def process_medical_record(
    file_path: str
) -> str:
    
    openai_api_key = "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"
    neo4j_uri = "neo4j+s://c703fa4d.databases.neo4j.io"
    neo4j_user = "neo4j" 
    neo4j_password = "JuhfBiYU-pgzF9CpGVhg5AhpkdHWtMPOCWnwpZxX09o" 
    
    """
    1) Reads an unstructured medical-record from `file_path`.
    2) Uses GPT-4o-mini to extract it into a JSON schema.
    3) Ensures a `patient_id` field (derived from the filename if missing).
    4) Builds & runs a single parameterized Cypher query that MERGEs the Patient
       and all related nodes (Condition, Medication, Allergy, VitalSign, LabResult)
       with WITH/UNWIND for a unified schema.
    5) Returns the Cypher that was executed.
    """
    # --- 1. Read text ---
    with open(file_path, 'r', encoding='utf-8') as f:
        record_text = f.read()

    # --- 2. Ask GPT for structured JSON ---
    openai.api_key = openai_api_key
    system_prompt = """
You are a medical-record parser. Given an unstructured clinical note, output
a single JSON object (no markdown fences!) with these top-level keys:

{
  "patient": { "patient_id": str, "name": str, "age": int, "gender": str,
               "dob": "YYYY-MM-DD", "date_of_visit": "YYYY-MM-DD",
               "complaints": str },
  "conditions": [ { "name": str } ],
  "allergies":   [ { "name": str, "reaction": str } ],
  "medications":[ { "name": str, "dosage": str, "frequency": str } ],
  "vitals":     { "date": "YYYY-MM-DD",
                   "blood_pressure": str,
                   "heart_rate": int,
                   "respiratory_rate": int,
                   "temperature": float,
                   "weight": float,
                   "height": float },
  "lab_results": { "date": "YYYY-MM-DD",
                   "HbA1c": float,
                   "fasting_glucose": int,
                   "creatinine": float,
                   "wbc_count": str,
                   "serum_sodium": int }
}
"""
    user_prompt = f"Parse this record:\n\n\"\"\"\n{record_text}\n\"\"\""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content": system_prompt},
            {"role":"user","content": user_prompt}
        ]
    )
    data = json.loads(resp.choices[0].message.content)

    # --- 3. Ensure patient_id exists (use filename if missing) ---
    filename_id = os.path.splitext(os.path.basename(file_path))[0]
    data["patient"]["patient_id"] = data["patient"].get("patient_id") or filename_id

    # --- 4. Build unified Cypher with WITH between each section ---
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

    # --- 5. Execute on Neo4j ---
    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    try:
        with driver.session() as session:
            session.run(cypher, params)
    finally:
        driver.close()

    return cypher

# Example usage
if __name__ == "__main__":
    q = process_medical_record(
        file_path="patient.txt"
    )
    print("Ran unified Cypher:\n", q)
