"""Write English medical terms in English letters inside Urdu transcripts.

Terms and their usual Urdu-script spellings live in vocab/english_terms.txt (editable, no code
changes needed). A transcript word, or a 2-3 word phrase, is replaced when it matches one of those
spellings after normalising letters the model often confuses, allowing one letter difference for
longer words. Exact spellings (not consonant skeletons) keep real Urdu words safe: پریشان
("worried") must never become "operation" (آپریشن).
"""

from functools import lru_cache
from pathlib import Path

TERMS_FILE = Path(__file__).parent / "vocab" / "english_terms.txt"
MAX_PHRASE_WORDS = 3
FUZZY_MIN_LEN = 5   # normalised length from which one letter difference is allowed
FUZZY2_MIN_LEN = 9  # ... and two: long foreign terms drift more and rarely collide with Urdu words

# Letters the model often confuses -> one representative; ھ is dropped (aspiration is unreliable).
CONFUSABLE = str.maketrans({
    "پ": "ب", "ٹ": "ت", "ط": "ت", "ص": "س", "ث": "س", "ذ": "ز", "ض": "ز", "ظ": "ز", "ج": "ز",
    "ق": "ک", "ح": "ہ", "آ": "ا", "ے": "ی", "ئ": "ی", "ھ": None,
})


def normalise(text):
    return text.translate(CONFUSABLE).replace(" ", "")


def edit_distance_within(a, b, limit):
    """True if Levenshtein distance(a, b) <= limit (strings here are short, so plain DP)."""
    if abs(len(a) - len(b)) > limit:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > limit:
            return False
        prev = cur
    return prev[-1] <= limit


@lru_cache(maxsize=1)
def load_terms(path=TERMS_FILE):
    """-> list of (spelling words, English), longest spellings first."""
    entries = []
    if not Path(path).exists():
        return entries
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        english, spellings = (part.strip() for part in line.split("=", 1))
        for spelling in spellings.split(","):
            if spelling.strip():
                entries.append((tuple(spelling.split()), english))
    return sorted(entries, key=lambda e: -len("".join(e[0])))


def word_matches(heard, listed):
    norm_listed = normalise(listed)
    if len(norm_listed) < FUZZY_MIN_LEN:
        # Short words (acronym letters, بی/سی) must match exactly: with confusable letters merged,
        # the address "بی بی" would become BP (بی پی), and hepatitis C would become B.
        return heard == listed
    limit = 2 if len(norm_listed) >= FUZZY2_MIN_LEN else 1
    return edit_distance_within(normalise(heard), norm_listed, limit)


def match(candidate):
    """candidate: one word or a space-separated phrase. Compared word by word with a listed
    spelling of the same length, so fuzzy matching can never swallow a neighbouring word."""
    words = candidate.split()
    for spelling, english in load_terms():
        if len(spelling) == len(words) and all(map(word_matches, words, spelling)):
            return english
    return None


# Short Urdu words the model often glues onto a neighbour ("کی آرٹریز" -> "کیارٹریز").
GLUE_WORDS = ["میں", "سے", "کو", "نے", "پر", "کی", "کے", "کا", "ہے", "ہیں", "اور"]
MIN_TERM_LEN = 4  # the remaining part must be at least this long to count as a term


def split_glued(word):
    """'کیارٹریز' -> ['کی', 'arteries']; None if the word isn't a glue word + a known term."""
    for glue in GLUE_WORDS:
        if word.startswith(glue) and len(word) - len(glue) >= MIN_TERM_LEN:
            english = match(word[len(glue):])
            if english:
                return [glue, english]
        if word.endswith(glue) and len(word) - len(glue) >= MIN_TERM_LEN:
            english = match(word[:-len(glue)])
            if english:
                return [english, glue]
    return None


def apply_english_terms(text):
    words, out, i = text.split(), [], 0
    while i < len(words):
        for n in range(min(MAX_PHRASE_WORDS, len(words) - i), 0, -1):  # longest phrase first
            english = match(" ".join(words[i:i + n]))
            if english:
                out.append(english)
                i += n
                break
        else:
            glued_phrase = split_glued_phrase(words, i)
            if glued_phrase:
                parts, used = glued_phrase
                out.extend(parts)
                i += used
                continue
            out.extend(split_glued(words[i]) or [words[i]])
            i += 1
    return " ".join(out)


def split_glued_phrase(words, i):
    """Glue word stuck to the first word of a phrase: 'کاای سی جی' -> ['کا', 'ECG'] (3 words used)."""
    for glue in GLUE_WORDS:
        rest = words[i][len(glue):]
        if not (words[i].startswith(glue) and rest):
            continue
        for n in range(min(MAX_PHRASE_WORDS, len(words) - i), 1, -1):
            english = match(" ".join([rest] + words[i + 1:i + n]))
            if english:
                return [glue, english], n
    return None
