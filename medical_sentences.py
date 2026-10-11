"""Synthetic doctor-patient sentences (Urdu + English medical terms) for training data.

Each sentence comes in two forms:
  label  what the model should write: Urdu in Urdu script, English terms in English letters
  tts    what the Urdu text-to-speech voice reads: the same sentence with every English term
         written in Urdu script, so it is pronounced the way Pakistani doctors say it

    label: مریض کو diabetes ہے اور وہ metformin لے رہا ہے۔
    tts:   مریض کو ڈائبیٹیز ہے اور وہ میٹفارمن لے رہا ہے۔

In templates, {SLOT} picks a random entry from that slot's list, and [english|اردو] is a fixed
English word with its Urdu-script pronunciation.
"""

import random
import re

# (label, Urdu-script pronunciation). Urdu-only entries use the same text twice.
SLOTS = {
    "SYM": [
        ("chest pain", "چیسٹ پین"), ("palpitations", "پیلپیٹیشنز"),
        ("shortness of breath", "شارٹنس آف بریتھ"), ("fever", "فیور"), ("cough", "کف"),
        ("headache", "ہیڈیک"), ("dizziness", "ڈیزینس"), ("nausea", "نازیا"),
        ("vomiting", "وومٹنگ"), ("diarrhea", "ڈائریا"), ("constipation", "کانسٹیپیشن"),
        ("acidity", "ایسڈٹی"), ("heartburn", "ہارٹ برن"), ("back pain", "بیک پین"),
        ("joint pain", "جوائنٹ پین"), ("fatigue", "فٹیگ"), ("weakness", "ویکنس"),
        ("swelling", "سویلنگ"), ("numbness", "نمبنس"), ("anxiety", "اینگزائٹی"),
        ("insomnia", "انسومنیا"), ("allergy", "الرجی"), ("sore throat", "سور تھروٹ"),
        ("بخار", "بخار"), ("کھانسی", "کھانسی"), ("سر درد", "سر درد"), ("سینے میں درد", "سینے میں درد"),
        ("چکر", "چکر"), ("متلی", "متلی"), ("پیٹ درد", "پیٹ درد"), ("کمزوری", "کمزوری"),
        ("سانس پھولنے", "سانس پھولنے"), ("جلن", "جلن"), ("بے چینی", "بے چینی"),
    ],
    "DIS": [
        ("diabetes", "ڈائبیٹیز"), ("hypertension", "ہائپرٹینشن"), ("asthma", "استھما"),
        ("heart failure", "ہارٹ فیلیئر"), ("heart attack", "ہارٹ اٹیک"), ("angina", "اینجائنا"),
        ("arrhythmia", "اریتھمیا"), ("atrial fibrillation", "ایٹریل فبریلیشن"),
        ("stroke", "سٹروک"), ("kidney failure", "کڈنی فیلیئر"), ("hepatitis", "ہیپاٹائٹس"),
        ("tuberculosis", "ٹیوبرکلوسس"), ("pneumonia", "نمونیا"), ("typhoid", "ٹائیفائیڈ"),
        ("dengue", "ڈینگی"), ("malaria", "ملیریا"), ("thyroid", "تھائیرائیڈ"),
        ("anemia", "انیمیا"), ("cholesterol", "کولیسٹرول"), ("arthritis", "آرتھرائٹس"),
        ("migraine", "مائیگرین"), ("gastritis", "گیسٹرائٹس"), ("ulcer", "السر"),
        ("copd", "سی او پی ڈی"), ("cancer", "کینسر"), ("depression", "ڈپریشن"),
        ("obesity", "اوبیسٹی"), ("شوگر", "شوگر"), ("دمہ", "دمہ"), ("یرقان", "یرقان"),
    ],
    "MED": [
        ("aspirin", "اسپرین"), ("clopidogrel", "کلوپیڈوگریل"), ("atorvastatin", "اٹورواسٹیٹن"),
        ("rosuvastatin", "روزوواسٹیٹن"), ("metformin", "میٹفارمن"), ("insulin", "انسولین"),
        ("amlodipine", "املوڈیپین"), ("losartan", "لوسارٹن"), ("bisoprolol", "بائیسوپرولول"),
        ("metoprolol", "میٹوپرولول"), ("furosemide", "فیوروسیمائیڈ"), ("warfarin", "وارفرین"),
        ("heparin", "ہیپارن"), ("nitroglycerin", "نائٹروگلیسرین"), ("paracetamol", "پیراسیٹامول"),
        ("ibuprofen", "آئبوپروفین"), ("omeprazole", "اومیپرازول"), ("amoxicillin", "اموکسیسلن"),
        ("azithromycin", "ازیتھرومائسن"), ("ciprofloxacin", "سپروفلوکساسن"),
        ("salbutamol", "سالبیوٹامول"), ("levothyroxine", "لیوتھائراکسن"),
        ("antibiotic", "اینٹی بائیوٹک"), ("painkiller", "پین کلر"), ("multivitamin", "ملٹی وٹامن"),
    ],
    "FORM": [
        ("tablet", "ٹیبلٹ"), ("capsule", "کیپسول"), ("syrup", "سیرپ"), ("injection", "انجکشن"),
        ("drip", "ڈرپ"), ("inhaler", "انہیلر"),
    ],
    "TEST": [
        ("ecg", "ای سی جی"), ("echo", "ایکو"), ("cbc", "سی بی سی"), ("blood test", "بلڈ ٹیسٹ"),
        ("urine test", "یورین ٹیسٹ"), ("chest x ray", "چیسٹ ایکسرے"), ("ultrasound", "الٹراساؤنڈ"),
        ("ct scan", "سی ٹی اسکین"), ("mri", "ایم آر آئی"), ("troponin", "ٹروپونن"),
        ("lipid profile", "لپڈ پروفائل"), ("sugar test", "شوگر ٹیسٹ"),
        ("liver function test", "لیور فنکشن ٹیسٹ"), ("kidney function test", "کڈنی فنکشن ٹیسٹ"),
        ("stress test", "سٹریس ٹیسٹ"), ("angiography", "اینجیوگرافی"), ("holter", "ہولٹر"),
    ],
    "PROC": [
        ("angioplasty", "اینجیوپلاسٹی"), ("stent", "سٹینٹ"), ("bypass surgery", "بائی پاس سرجری"),
        ("pacemaker", "پیس میکر"), ("dialysis", "ڈائیلیسس"), ("endoscopy", "اینڈوسکوپی"),
        ("colonoscopy", "کولونوسکوپی"), ("biopsy", "بائیوپسی"), ("operation", "آپریشن"),
        ("surgery", "سرجری"), ("nebulizer", "نیبولائزر"), ("angiography", "اینجیوگرافی"),
    ],
    "FIND": [
        ("blockage", "بلاکیج"), ("infection", "انفیکشن"), ("low hemoglobin", "لو ہیموگلوبن"),
        ("high cholesterol", "ہائی کولیسٹرول"), ("fatty liver", "فیٹی لیور"),
        ("kidney stone", "کڈنی سٹون"), ("fracture", "فریکچر"), ("enlarged heart", "انلارجڈ ہارٹ"),
        ("normal", "نارمل"),
    ],
    "VITAL": [
        ("blood pressure", "بلڈ پریشر"), ("heart rate", "ہارٹ ریٹ"), ("pulse", "پلس"),
        ("oxygen saturation", "آکسیجن سیچوریشن"), ("temperature", "ٹمپریچر"),
        ("sugar level", "شوگر لیول"),
    ],
    "STATE": [
        ("high", "ہائی"), ("low", "لو"), ("normal", "نارمل"), ("stable", "سٹیبل"),
        ("کنٹرول میں", "کنٹرول میں"), ("بہت زیادہ", "بہت زیادہ"),
    ],
    "DUR": [(d, d) for d in ["دو دن", "تین دن", "چار دن", "ایک ہفتے", "دو ہفتوں", "ایک مہینے",
                             "کافی عرصے", "کل رات", "آج صبح", "پچھلے سال", "کئی دنوں"]],
    "FREQ": [(f, f) for f in ["ایک بار", "دو بار", "تین بار"]],
    "TIMING": [(t, t) for t in ["کھانے کے بعد", "کھانے سے پہلے", "رات کو سونے سے پہلے",
                                "صبح خالی پیٹ", "ناشتے کے ساتھ"]],
    "BODY": [(b, b) for b in ["سینے", "پیٹ", "کمر", "سر", "ٹانگوں", "بازو", "گردن", "گھٹنوں"]],
    "PERSON": [(p, p) for p in ["میرے والد صاحب", "میری والدہ", "میرے بھائی", "میری بیوی",
                                "مریض", "بچے", "میرے دادا"]],
}

TEMPLATES = [
    "ڈاکٹر صاحب، مجھے {DUR} سے {SYM} کی شکایت ہے۔",
    "مجھے {DUR} سے {SYM} اور {SYM} کا مسئلہ ہے۔",
    "کیا آپ کو {SYM} کی شکایت ہے؟",
    "{SYM} کب سے ہے اور کیا رات کو زیادہ ہوتا ہے؟",
    "{PERSON} کو {DIS} ہے اور وہ {MED} لے رہے ہیں۔",
    "مریض کو {DIS} ہے اور وہ {MED} لے رہا ہے۔",
    "کیا آپ کو {DIS} یا {DIS} ہے؟",
    "{PERSON} کو {DIS} کی [history|ہسٹری] ہے۔",
    "یہ {MED} دن میں {FREQ} {TIMING} لیں۔",
    "{MED} کی ایک {FORM} {TIMING} لینی ہے۔",
    "آپ کا {TEST} کروانا ضروری ہے۔",
    "پہلے {TEST} اور {TEST} کروائیں، پھر [report|رپورٹ] لے کر آئیں۔",
    "{TEST} کی [report|رپورٹ] میں {FIND} آیا ہے۔",
    "آپ کا {VITAL} {STATE} ہے۔",
    "مریض کا {VITAL} چیک کریں اور ہر گھنٹے بعد [monitor|مانیٹر] کریں۔",
    "ہمیں {PROC} کا فیصلہ جلد کرنا ہوگا۔",
    "{PROC} کے بعد مریض کی [condition|کنڈیشن] [stable|سٹیبل] ہے۔",
    "اگر {SYM} دوبارہ ہو تو فوراً [emergency|ایمرجنسی] آئیں۔",
    "[please|پلیز] {MED} بند نہ کریں، یہ آپ کے {DIS} کے لیے ضروری ہے۔",
    "{SYM} کی وجہ {DIS} ہو سکتی ہے۔",
    "آپ کو {DIS} کا [risk|رسک] ہے، اس لیے نمک کم کھائیں اور روزانہ [walk|واک] کریں۔",
    "[patient|پیشنٹ] کو کل [admit|ایڈمٹ] کیا تھا اور آج [discharge|ڈسچارج] ہو جائے گا۔",
    "[the patient is stable|دی پیشنٹ از سٹیبل]، لیکن {VITAL} پر نظر رکھیں۔",
    "[please check his|پلیز چیک ہز] {VITAL} [and|اینڈ] {VITAL}۔",
    "[patient has a history of|پیشنٹ ہیز اے ہسٹری آف] {DIS} [and|اینڈ] {DIS}۔",
    "[start|سٹارٹ] {MED} [and continue|اینڈ کنٹینیو] {MED}۔",
    "{BODY} میں درد اور {SYM} کی شکایت ہے۔",
    "سیڑھیاں چڑھتے ہوئے {SYM} کا مسئلہ ہوتا ہے۔",
    "میں {DUR} سے {MED} لے رہا ہوں لیکن کوئی فرق نہیں پڑا۔",
    "کیا آپ کو کسی دوائی سے [allergy|الرجی] ہے؟",
    "آپ کا {TEST} [normal|نارمل] ہے، پریشانی کی کوئی بات نہیں۔",
    "[follow up|فالو اپ] کے لیے دو ہفتے بعد آئیں اور {TEST} کی [report|رپورٹ] ساتھ لائیں۔",
    "{MED} سے معدے میں جلن ہو سکتی ہے، اس لیے {TIMING} لیں۔",
    "بچے کو {DUR} سے {SYM} ہے اور وہ کچھ کھا نہیں رہا۔",
    "ڈاکٹر صاحب، کیا مجھے {PROC} کی ضرورت ہے؟",
    "{VITAL} بہت {STATE} ہے، فوراً {MED} کی {FORM} دیں۔",
    "[icu|آئی سی یو] میں مریض کا {VITAL} {STATE} ہے۔",
    "[nurse|نرس]، مریض کو {MED} کا [injection|انجکشن] لگا دیں۔",
    "{DIS} کے مریضوں میں {SYM} عام ہے۔",
    "{FIND} کی وجہ سے {PROC} کرنا پڑے گا۔",
    "{TEST} سے پتہ چلا کہ {FIND} ہے۔",
    "[angiography|اینجیوگرافی] میں دل کی نالی میں [blockage|بلاکیج] نظر آئی ہے۔",
    "[doctor|ڈاکٹر] نے {DIS} کے لیے {MED} اور {MED} لکھی ہیں۔",
    "{PERSON} کا {PROC} پچھلے مہینے ہوا تھا۔",
    "رات کو لیٹتے ہی {SYM} شروع ہو جاتا ہے۔",
    "آپ کا {VITAL} {STATE} آیا ہے، {TEST} دوبارہ کروائیں۔",
]

_FIXED = re.compile(r"\[([^|\]]+)\|([^\]]+)\]")
_SLOT = re.compile(r"\{([A-Z]+)\}")
_LATIN = re.compile(r"[a-z]")


def _fill(template, rng):
    label, tts, terms = template, template, []
    for m in _FIXED.finditer(template):
        label = label.replace(m.group(0), m.group(1), 1)
        tts = tts.replace(m.group(0), m.group(2), 1)
    for m in _SLOT.finditer(template):
        lab, say = rng.choice(SLOTS[m.group(1)])
        label = label.replace(m.group(0), lab, 1)
        tts = tts.replace(m.group(0), say, 1)
        if _LATIN.search(lab):
            terms.append(lab)
    return {"label": label, "tts": tts, "terms": terms}


def generate(n, seed=0, exclude=()):
    """n distinct sentences (deterministic for a seed), skipping labels in `exclude`."""
    rng, seen, out = random.Random(seed), set(exclude), []
    for _ in range(n * 50):
        s = _fill(rng.choice(TEMPLATES), rng)
        if s["label"] not in seen:
            seen.add(s["label"])
            out.append(s)
            if len(out) == n:
                break
    return out


if __name__ == "__main__":
    for s in generate(8, seed=1):
        print(s["label"], "\n   ", s["tts"])
