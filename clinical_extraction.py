"""Omni Urdu transcript -> structured record -> SQLite. Stdlib only; no network."""
import hashlib, json, re, sqlite3
from datetime import datetime, timezone

SCHEMA_VERSION = 1

# English terms already restored by english_terms.py -> entity type.
TERM_TYPES = {
    "blockage": "finding", "infection": "finding", "fracture": "finding", "allergy": "finding",
    "heart attack": "diagnosis", "diabetes": "diagnosis", "hepatitis B": "diagnosis",
    "hepatitis C": "diagnosis", "pneumonia": "diagnosis", "dengue": "diagnosis",
    "malaria": "diagnosis", "typhoid": "diagnosis", "TB": "diagnosis", "cancer": "diagnosis",
    "angiography": "procedure", "angioplasty": "procedure", "PCI": "procedure",
    "CABG": "procedure", "bypass": "procedure", "surgery": "procedure", "operation": "procedure",
    "dialysis": "procedure", "biopsy": "procedure",
    "ECG": "test", "CT scan": "test", "MRI": "test", "x-ray": "test", "ultrasound": "test",
    "CBC": "test", "echo": "test", "troponin": "test",
    "aspirin": "medication", "heparin": "medication", "insulin": "medication",
    "antibiotic": "medication",
}
# Urdu symptom words (Omni Urdu output) -> English label.
SYMPTOMS = {"کمر درد": "back pain", "سینے میں درد": "chest pain", "سر درد": "headache",
            "بخار": "fever", "کھانسی": "cough", "چکر": "dizziness", "الٹی": "vomiting",
            "سانس": "breathlessness", "درد": "pain"}
VITALS = {"BP": "blood_pressure", "blood pressure": "blood_pressure", "sugar": "blood_sugar"}
QUALIFIERS = {"ٹھیک": "normal", "نارمل": "normal", "زیادہ": "high", "ہائی": "high",
              "کم": "low", "لو": "low"}
NEGATION = ("نہیں", "نہ ")
QUESTION = ("کیا ", "کیوں ", "کب ", "کتنا ", "کتنی ")  # clause opens with a question word
CONTEXTS = {"affirmed", "negated", "asked"}
TYPES = {"finding", "diagnosis", "procedure", "test", "medication", "symptom"}


def _clauses(text):
    # Omni Urdu has no punctuation; split on Urdu/Latin punctuation and common conjunctions.
    return [c.strip() for c in re.split(r"[۔،؟.?!]|\sلیکن\s|\sاور\s|\sمگر\s", text) if c.strip()]


def extract(text):
    """Deterministic extraction: every value is grounded in a span of `text`."""
    entities, vitals, seen = [], {}, set()
    for clause in _clauses(text):
        context = ("asked" if clause.startswith(QUESTION) or clause.endswith("؟")
                   else "negated" if any(n in clause for n in NEGATION) else "affirmed")
        for term, etype in TERM_TYPES.items():
            if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", clause, re.I) and term not in seen:
                seen.add(term)
                entities.append({"type": etype, "value": term, "context": context, "evidence": clause})
        for urdu, label in SYMPTOMS.items():
            if urdu in clause and label not in seen and not (label == "pain" and seen & {"back pain", "chest pain", "headache"}):
                seen.add(label)
                entities.append({"type": "symptom", "value": label, "context": context, "evidence": clause})
        for term, key in VITALS.items():
            if re.search(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", clause) and key not in vitals:
                q = next((v for u, v in QUALIFIERS.items() if u in clause.split() or u in clause), None)
                if q:
                    vitals[key] = {"status": q, "evidence": clause}
    return {"schema_version": SCHEMA_VERSION, "entities": entities, "vitals": vitals}


def _check(ok, msg):
    if not ok:  # explicit raise, not assert: python -O strips asserts
        raise ValueError(msg)


def validate(record, transcript):
    """Reject malformed or ungrounded records (also used on SLM output)."""
    _check(record.get("schema_version") == SCHEMA_VERSION, "schema_version")
    _check(isinstance(record.get("entities"), list) and isinstance(record.get("vitals"), dict), "shape")
    for e in record["entities"]:
        _check(e.get("type") in TYPES, f"bad type {e.get('type')!r}")
        _check(isinstance(e.get("value"), str) and e["value"], "empty value")
        _check(e.get("context") in CONTEXTS, "bad context")
        _check(e.get("evidence") and e["evidence"] in transcript, f"ungrounded: {e.get('value')}")
    for k, v in record["vitals"].items():
        _check(v.get("status") in {"normal", "high", "low"} and v.get("evidence") in transcript, f"vital {k}")
    return record


DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY, audio_sha256 TEXT NOT NULL UNIQUE, audio_path TEXT NOT NULL,
    duration_s REAL NOT NULL, created_at TEXT NOT NULL) STRICT;
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY, recording_id INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    asr_model TEXT NOT NULL, text TEXT NOT NULL, rtf REAL, created_at TEXT NOT NULL) STRICT;
CREATE TABLE IF NOT EXISTS extractions (
    id INTEGER PRIMARY KEY, transcript_id INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    extractor TEXT NOT NULL, schema_version INTEGER NOT NULL,
    record_json TEXT NOT NULL CHECK (json_valid(record_json)), created_at TEXT NOT NULL) STRICT;
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY, extraction_id INTEGER NOT NULL REFERENCES extractions(id) ON DELETE CASCADE,
    type TEXT NOT NULL CHECK (type IN ('finding','diagnosis','procedure','test','medication','symptom')),
    value TEXT NOT NULL, context TEXT NOT NULL CHECK (context IN ('affirmed','negated','asked')),
    evidence TEXT NOT NULL) STRICT;
CREATE INDEX IF NOT EXISTS idx_entities_type_value ON entities(type, value);
"""


def connect(path="pipeline.db"):
    db = sqlite3.connect(path)
    db.executescript(DDL)
    return db


def store(db, audio_path, audio_bytes, duration_s, asr_model, text, rtf, record, extractor="rules"):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sha = hashlib.sha256(audio_bytes).hexdigest()
    with db:  # one transaction: all rows or none
        db.execute("INSERT OR IGNORE INTO recordings (audio_sha256, audio_path, duration_s, created_at) "
                   "VALUES (?,?,?,?)", (sha, str(audio_path), duration_s, now))
        rec_id = db.execute("SELECT id FROM recordings WHERE audio_sha256 = ?", (sha,)).fetchone()[0]
        t_id = db.execute("INSERT INTO transcripts (recording_id, asr_model, text, rtf, created_at) "
                          "VALUES (?,?,?,?,?)", (rec_id, asr_model, text, rtf, now)).lastrowid
        x_id = db.execute("INSERT INTO extractions (transcript_id, extractor, schema_version, record_json, created_at) "
                          "VALUES (?,?,?,?,?)", (t_id, extractor, record["schema_version"],
                                                 json.dumps(record, ensure_ascii=False), now)).lastrowid
        db.executemany("INSERT INTO entities (extraction_id, type, value, context, evidence) VALUES (?,?,?,?,?)",
                       [(x_id, e["type"], e["value"], e["context"], e["evidence"]) for e in record["entities"]])
    return x_id
