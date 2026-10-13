"""Synthetic doctor-patient sentences (Urdu + English medical terms) for training data.

Each sentence has:
  label  what the model should write: Urdu in Urdu script, English terms in English letters
  tts    what the text-to-speech voice reads. For the Urdu voice ("urd") every English term is
         written in Urdu script, so it is pronounced the way Pakistani doctors say it; for the
         English voice ("eng") it is the English sentence itself.

    label: مریض کو diabetes ہے اور وہ metformin لے رہا ہے۔
    tts:   مریض کو ڈائبیٹیز ہے اور وہ میٹفارمن لے رہا ہے۔

In templates, {SLOT} picks a random entry from that slot's list, and [english|اردو] is a fixed
English word with its Urdu-script pronunciation.

HELD_OUT terms never appear in training sentences; generate(..., held_out="only") builds test
sentences from them to measure how well the model writes medical words it has never heard.
"""

import random
import re


def _terms(block):
    """'english = اردو' lines -> [(english, urdu)]; a line without '=' is an Urdu-only entry."""
    out = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        en, _, ur = line.partition("=")
        out.append((en.strip(), ur.strip() or en.strip()))
    return out


SYMPTOMS = _terms("""
chest pain = چیسٹ پین
palpitations = پیلپیٹیشنز
shortness of breath = شارٹنس آف بریتھ
breathlessness = بریتھ لیسنس
fever = فیور
cough = کف
dry cough = ڈرائی کف
headache = ہیڈیک
migraine = مائیگرین
dizziness = ڈیزینس
vertigo = ورٹیگو
fainting = فینٹنگ
nausea = نازیا
vomiting = وومٹنگ
diarrhea = ڈائریا
constipation = کانسٹیپیشن
acidity = ایسڈٹی
heartburn = ہارٹ برن
indigestion = انڈائجیسچن
bloating = بلوٹنگ
abdominal pain = ایبڈامنل پین
back pain = بیک پین
neck pain = نیک پین
joint pain = جوائنٹ پین
knee pain = نی پین
muscle pain = مسل پین
fatigue = فٹیگ
weakness = ویکنس
swelling = سویلنگ
numbness = نمبنس
tingling = ٹنگلنگ
anxiety = اینگزائٹی
insomnia = انسومنیا
allergy = الرجی
sore throat = سور تھروٹ
runny nose = رنی نوز
sneezing = سنیزنگ
rash = ریش
itching = اچنگ
blurred vision = بلرڈ وژن
weight loss = ویٹ لاس
loss of appetite = لاس آف ایپیٹائٹ
burning urine = برننگ یورین
frequent urination = فریکوئنٹ یوری نیشن
chills = چلز
sweating = سویٹنگ
wheezing = وہیزنگ
""") + _terms("""
بخار
کھانسی
سر درد
سینے میں درد
چکر
متلی
پیٹ درد
کمزوری
سانس پھولنے
جلن
بے چینی
الٹی
""")

DISEASES = _terms("""
diabetes = ڈائبیٹیز
hypertension = ہائپرٹینشن
asthma = استھما
heart failure = ہارٹ فیلیئر
heart attack = ہارٹ اٹیک
myocardial infarction = مائیوکارڈیل انفارکشن
angina = اینجائنا
arrhythmia = اریتھمیا
atrial fibrillation = ایٹریل فبریلیشن
pericarditis = پیریکارڈائٹس
stroke = سٹروک
kidney failure = کڈنی فیلیئر
hepatitis = ہیپاٹائٹس
tuberculosis = ٹیوبرکلوسس
pneumonia = نمونیا
bronchitis = برونکائٹس
typhoid = ٹائیفائیڈ
dengue = ڈینگی
malaria = ملیریا
thyroid = تھائیرائیڈ
hypothyroidism = ہائپوتھائیرائیڈزم
hyperthyroidism = ہائپرتھائیرائیڈزم
anemia = انیمیا
cholesterol = کولیسٹرول
arthritis = آرتھرائٹس
osteoporosis = آسٹیوپوروسس
gout = گاؤٹ
gastritis = گیسٹرائٹس
ulcer = السر
copd = سی او پی ڈی
cancer = کینسر
leukemia = لیوکیمیا
depression = ڈپریشن
epilepsy = ایپی لیپسی
obesity = اوبیسٹی
covid = کووڈ
influenza = انفلوئنزا
urinary tract infection = یورینری ٹریکٹ انفیکشن
kidney stones = کڈنی سٹونز
fatty liver = فیٹی لیور
cirrhosis = سروسس
appendicitis = اپینڈیسائٹس
eczema = ایگزیما
psoriasis = سورائسس
glaucoma = گلوکوما
cataract = کیٹریکٹ
dementia = ڈیمینشیا
parkinson = پارکنسن
sinusitis = سائنوسائٹس
deep vein thrombosis = ڈیپ وین تھرومبوسس
cardiomyopathy = کارڈیومائیوپیتھی
endocarditis = اینڈوکارڈائٹس
""") + _terms("""
شوگر
دمہ
یرقان
""")

MEDICINES = _terms("""
aspirin = اسپرین
clopidogrel = کلوپیڈوگریل
ticagrelor = ٹکاگریلر
atorvastatin = اٹورواسٹیٹن
rosuvastatin = روزوواسٹیٹن
metformin = میٹفارمن
glimepiride = گلیمیپرائیڈ
sitagliptin = سیٹاگلپٹن
empagliflozin = ایمپاگلیفلوزن
insulin = انسولین
amlodipine = املوڈیپین
losartan = لوسارٹن
valsartan = ولسارٹن
ramipril = رامی پرل
captopril = کیپٹوپرل
bisoprolol = بائیسوپرولول
metoprolol = میٹوپرولول
carvedilol = کارویڈیلول
propranolol = پروپرانولول
furosemide = فیوروسیمائیڈ
spironolactone = سپائرونولیکٹون
hydrochlorothiazide = ہائیڈروکلوروتھائیازائیڈ
digoxin = ڈائیگوکسن
amiodarone = امیوڈیرون
warfarin = وارفرین
rivaroxaban = ریواروکسابان
apixaban = اپیکسابان
heparin = ہیپارن
enoxaparin = اینوکساپرن
nitroglycerin = نائٹروگلیسرین
isosorbide = آئسوسوربائیڈ
colchicine = کولچیسین
paracetamol = پیراسیٹامول
ibuprofen = آئبوپروفین
diclofenac = ڈائیکلوفینک
naproxen = نیپروکسن
tramadol = ٹرامادول
omeprazole = اومیپرازول
esomeprazole = ایسومیپرازول
pantoprazole = پینٹوپرازول
ranitidine = رینیٹیڈین
domperidone = ڈومپیریڈون
ondansetron = اونڈانسیٹرون
metoclopramide = میٹوکلوپرامائیڈ
loperamide = لوپیرامائیڈ
lactulose = لیکٹولوز
amoxicillin = اموکسیسلن
augmentin = آگمینٹن
azithromycin = ازیتھرومائسن
clarithromycin = کلیریتھرومائسن
ciprofloxacin = سپروفلوکساسن
levofloxacin = لیووفلوکساسن
ceftriaxone = سیفٹرائیکسون
cefixime = سیفکسیم
metronidazole = میٹرونیڈازول
doxycycline = ڈوکسی سائکلین
vancomycin = وینکومائسن
salbutamol = سالبیوٹامول
montelukast = مونٹیلوکاسٹ
prednisolone = پریڈنیسولون
hydrocortisone = ہائیڈروکارٹیسون
dexamethasone = ڈیکسامیتھاسون
cetirizine = سیٹیریزین
loratadine = لوراٹاڈین
levothyroxine = لیوتھائراکسن
carbimazole = کاربیمازول
sertraline = سرٹرالین
escitalopram = ایسیٹالوپرام
alprazolam = الپرازولام
gabapentin = گیباپینٹن
pregabalin = پریگابالن
vitamin d = وٹامن ڈی
calcium = کیلشیم
folic acid = فولک ایسڈ
iron = آئرن
antibiotic = اینٹی بائیوٹک
painkiller = پین کلر
multivitamin = ملٹی وٹامن
antacid = اینٹاسڈ
""")

FORMS = _terms("""
tablet = ٹیبلٹ
capsule = کیپسول
syrup = سیرپ
injection = انجکشن
drip = ڈرپ
inhaler = انہیلر
drops = ڈراپس
cream = کریم
sachet = سیشے
""")

TESTS = _terms("""
ecg = ای سی جی
echo = ایکو
cbc = سی بی سی
blood test = بلڈ ٹیسٹ
urine test = یورین ٹیسٹ
chest x ray = چیسٹ ایکسرے
x ray = ایکسرے
ultrasound = الٹراساؤنڈ
ct scan = سی ٹی اسکین
mri = ایم آر آئی
troponin = ٹروپونن
lipid profile = لپڈ پروفائل
sugar test = شوگر ٹیسٹ
liver function test = لیور فنکشن ٹیسٹ
kidney function test = کڈنی فنکشن ٹیسٹ
thyroid profile = تھائیرائیڈ پروفائل
stress test = سٹریس ٹیسٹ
angiography = اینجیوگرافی
holter = ہولٹر
spirometry = سپائرومیٹری
creatinine = کریٹینین
hemoglobin = ہیموگلوبن
electrolytes = الیکٹرولائٹس
blood culture = بلڈ کلچر
biopsy report = بائیوپسی رپورٹ
pet scan = پیٹ اسکین
doppler = ڈوپلر
endoscopy = اینڈوسکوپی
eeg = ای ای جی
ferritin = فیریٹن
""")

PROCEDURES = _terms("""
angioplasty = اینجیوپلاسٹی
stent = سٹینٹ
bypass surgery = بائی پاس سرجری
pacemaker = پیس میکر
dialysis = ڈائیلیسس
endoscopy = اینڈوسکوپی
colonoscopy = کولونوسکوپی
biopsy = بائیوپسی
operation = آپریشن
surgery = سرجری
nebulizer = نیبولائزر
angiography = اینجیوگرافی
knee replacement surgery = نی ری پلیسمنٹ سرجری
hip replacement = ہپ ری پلیسمنٹ
cataract surgery = کیٹریکٹ سرجری
appendectomy = اپینڈیکٹومی
cholecystectomy = کولی سسٹیکٹومی
tonsillectomy = ٹانسلیکٹومی
cesarean section = سیزیرین سیکشن
blood transfusion = بلڈ ٹرانسفیوژن
physiotherapy = فزیوتھراپی
chemotherapy = کیموتھراپی
radiotherapy = ریڈیوتھراپی
cardioversion = کارڈیوورژن
lumbar puncture = لمبر پنکچر
""")

FINDINGS = _terms("""
blockage = بلاکیج
infection = انفیکشن
low hemoglobin = لو ہیموگلوبن
high cholesterol = ہائی کولیسٹرول
fatty liver = فیٹی لیور
kidney stone = کڈنی سٹون
fracture = فریکچر
enlarged heart = انلارجڈ ہارٹ
normal = نارمل
inflammation = انفلیمیشن
tumor = ٹیومر
cyst = سسٹ
clot = کلاٹ
fluid = فلوئڈ
""")

VITALS = _terms("""
blood pressure = بلڈ پریشر
heart rate = ہارٹ ریٹ
pulse = پلس
oxygen saturation = آکسیجن سیچوریشن
temperature = ٹمپریچر
sugar level = شوگر لیول
respiratory rate = ریسپیریٹری ریٹ
weight = ویٹ
""")

# Terms used only in test sentences, never in training (to measure unseen-word spelling).
HELD_OUT = {
    "endocarditis", "cardiomyopathy", "sinusitis", "glaucoma", "psoriasis", "gout", "leukemia",
    "spironolactone", "ramipril", "pantoprazole", "cetirizine", "prednisolone", "ceftriaxone",
    "doxycycline", "montelukast", "ferritin", "doppler", "creatinine", "cholecystectomy",
    "tonsillectomy", "cardioversion", "lumbar puncture", "radiotherapy", "cyst", "wheezing",
    "chills", "vertigo", "bloating",
}

SLOTS = {
    "SYM": SYMPTOMS, "DIS": DISEASES, "MED": MEDICINES, "FORM": FORMS, "TEST": TESTS,
    "PROC": PROCEDURES, "FIND": FINDINGS, "VITAL": VITALS,
    "STATE": _terms("""
        high = ہائی
        low = لو
        normal = نارمل
        stable = سٹیبل
        کنٹرول میں
        بہت زیادہ
    """),
    "DUR": _terms("دو دن\nتین دن\nچار دن\nایک ہفتے\nدو ہفتوں\nایک مہینے\nکافی عرصے\nکل رات\nآج صبح\nپچھلے سال\nکئی دنوں"),
    "FREQ": _terms("ایک بار\nدو بار\nتین بار"),
    "TIMING": _terms("کھانے کے بعد\nکھانے سے پہلے\nرات کو سونے سے پہلے\nصبح خالی پیٹ\nناشتے کے ساتھ"),
    "BODY": _terms("سینے\nپیٹ\nکمر\nسر\nٹانگوں\nبازو\nگردن\nگھٹنوں"),
    "PERSON": _terms("میرے والد صاحب\nمیری والدہ\nمیرے بھائی\nمیری بیوی\nمریض\nبچے\nمیرے دادا"),
}

# English-only slots for the English voice (English words only, plus English time phrases).
EN_SLOTS = {
    **{k: [t for t in v if re.search("[a-z]", t[0])] for k, v in SLOTS.items()},
    "STATE": [(s, s) for s in ["high", "low", "normal", "stable", "very high"]],
    "DUR": [(d, d) for d in ["two days", "three days", "a week", "two weeks", "a month", "last night",
                             "this morning", "several months"]],
    "FREQ": [(f, f) for f in ["once", "twice", "three times"]],
    "TIMING": [(t, t) for t in ["after meals", "before meals", "at bedtime", "on an empty stomach",
                                "with breakfast"]],
    "PERSON": [(p, p) for p in ["the patient", "my father", "my mother", "my son", "my wife"]],
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
    "مریض کو {DIS} ہے اور اسے {MED} دی جا رہی ہے۔",
    "ہمیں اس کا {TEST} اور {TEST} کروانا ہے۔",
    "اس کی {PROC} اگلے ہفتے ہے۔",
    "{MED} اور {MED} ایک ساتھ نہ لیں۔",
    "{MED} کا [dose|ڈوز] کم کر دیں کیونکہ {SYM} ہو رہا ہے۔",
    "{SYM} کے ساتھ {SYM} بھی ہو تو یہ {DIS} ہو سکتا ہے۔",
]

EN_TEMPLATES = [
    "please start {MED} and continue {MED}",
    "the patient has a history of {DIS} and {DIS}",
    "the patient has {DIS} and is on {MED}",
    "she has had {SYM} for {DUR}",
    "he complains of {SYM} and {SYM} since {DUR}",
    "please check his {VITAL} and {VITAL}",
    "the {TEST} shows {FIND}",
    "we need a {TEST} and a {TEST} today",
    "take {MED} {FREQ} a day {TIMING}",
    "{PERSON} is scheduled for {PROC} next week",
    "the patient was admitted with {SYM} and {DIS}",
    "her {VITAL} is {STATE} so give {MED} now",
    "stop {MED} and switch to {MED}",
    "{SYM} may be a sign of {DIS}",
    "after {PROC} the patient is stable",
    "repeat the {TEST} and send a {TEST}",
    "give one {FORM} of {MED} {TIMING}",
    "is there any family history of {DIS}",
]

_FIXED = re.compile(r"\[([^|\]]+)\|([^\]]+)\]")
_SLOT = re.compile(r"\{([A-Z]+)\}")
_LATIN = re.compile(r"[a-z]")


def _pools(slots, held_out):
    """Slot lists restricted to: 'exclude' -> no held-out terms, 'only' -> medical slots held-out only."""
    pools = {}
    for name, items in slots.items():
        if held_out == "exclude":
            pools[name] = [t for t in items if t[0] not in HELD_OUT]
        elif held_out == "only" and name in ("SYM", "DIS", "MED", "TEST", "PROC", "FIND"):
            pools[name] = [t for t in items if t[0] in HELD_OUT]
        else:
            pools[name] = items
    return pools


def _fill(template, rng, pools):
    label, tts, terms = template, template, []
    for m in _FIXED.finditer(template):
        label = label.replace(m.group(0), m.group(1), 1)
        tts = tts.replace(m.group(0), m.group(2), 1)
    for m in _SLOT.finditer(template):
        lab, say = rng.choice(pools[m.group(1)] or SLOTS[m.group(1)])
        label = label.replace(m.group(0), lab, 1)
        tts = tts.replace(m.group(0), say, 1)
        if _LATIN.search(lab):
            terms.append(lab)
    return label, tts, terms


def generate(n, seed=0, exclude=(), held_out="exclude", voice="urd"):
    """n distinct sentences (deterministic for a seed), skipping labels in `exclude`.

    held_out="exclude" for training, "only" for a test set built around HELD_OUT terms.
    voice="urd" (Urdu sentences, Urdu-script TTS text) or "eng" (English sentences).
    """
    templates = TEMPLATES if voice == "urd" else EN_TEMPLATES
    pools = _pools(SLOTS if voice == "urd" else EN_SLOTS, held_out)
    if held_out == "only":  # every test sentence must contain at least one held-out term
        templates = [t for t in templates if any(s in t for s in ("{SYM}", "{DIS}", "{MED}", "{TEST}", "{PROC}"))]
    rng, seen, out = random.Random(seed), set(exclude), []
    for _ in range(n * 50):
        label, tts, terms = _fill(rng.choice(templates), rng, pools)
        if voice == "eng":
            tts = label
        if held_out == "only" and not HELD_OUT.intersection(terms):
            continue
        if label not in seen:
            seen.add(label)
            out.append({"label": label, "tts": tts, "terms": terms, "voice": voice})
            if len(out) == n:
                break
    return out


def all_terms():
    return sorted({t[0] for items in SLOTS.values() for t in items if _LATIN.search(t[0])})


if __name__ == "__main__":
    print(len(all_terms()), "English terms,", len(HELD_OUT), "held out")
    for s in generate(4, seed=1) + generate(2, seed=1, voice="eng") + generate(2, seed=3, held_out="only"):
        print(s["voice"], s["label"], "\n     ", s["tts"])
