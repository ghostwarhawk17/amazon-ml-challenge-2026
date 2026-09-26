"""Indic scripts -> Latin letters, so "श्री सॉफ्टवेयर लिमिटेड" can be compared with "Sree Software Limited".

The nine Brahmic scripts in the data (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu,
Kannada, Malayalam) share one Unicode layout: every block is 128 code points and the same offset is
the same letter (0x15 is KA in all of them). One offset table therefore covers every script, plus a
few script-specific overrides. Standard library only.

Two layers:
  1. rules: consonant + inherent "a" / vowel sign / virama, nasal signs, digits, and (for the northern
     scripts) Hindi-style schwa deletion, so मार्केटिंग -> "marketing", not "maarakeitinga".
  2. a word dictionary learned from the training pairs (`learn_dictionary`): frequent Indic words are
     mapped to the Latin word they are matched with most often (लिमिटेड -> "limited",
     प्राइवेट -> "private"). Only words seen in many different entities are kept, so the dictionary
     holds common vocabulary, not the names of individual training businesses.
"""
import re
import unicodedata
from collections import Counter, defaultdict

_BLOCKS = {  # block start -> script
    0x0900: "deva", 0x0980: "beng", 0x0A00: "guru", 0x0A80: "gujr", 0x0B00: "orya",
    0x0B80: "taml", 0x0C00: "telu", 0x0C80: "knda", 0x0D00: "mlym",
}
# Hindi-like scripts drop the inherent vowel at the end of a word (and in VC_CV); the southern
# scripts write dead consonants explicitly (virama / chillu), so their inherent vowel is kept.
_SCHWA_DELETION = {"deva", "beng", "guru", "gujr", "orya"}

_CONS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "n",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # precomposed nukta letters (NFC decomposes most of them, but keep them anyway)
    0x58: "q", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r", 0x5D: "rh", 0x5E: "f", 0x5F: "y",
}
_NUKTA_OF = {"k": "q", "kh": "kh", "g": "g", "j": "z", "d": "r", "dh": "rh", "f": "f", "p": "f", "y": "y"}
_VOWELS = {  # independent vowels
    0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
    0x60: "ri", 0x61: "li",
}
_MATRAS = {  # dependent vowel signs
    0x3A: "e", 0x3B: "o", 0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
    0x4E: "e", 0x4F: "aw", 0x55: "", 0x56: "", 0x57: "", 0x62: "li", 0x63: "li",
}
_NASAL = {0x01, 0x02}  # candrabindu, anusvara
_VISARGA, _NUKTA, _VIRAMA = 0x03, 0x3C, 0x4D
# script-specific letters that break the shared layout
_OVERRIDES = {
    ("beng", 0x4E): ("dead", "t"),     # khanda ta
    ("beng", 0x70): ("cons", "r"), ("beng", 0x71): ("cons", "w"),
    ("orya", 0x71): ("cons", "w"),
    ("guru", 0x70): ("nasal", "n"),    # tippi
    ("guru", 0x71): ("skip", ""),      # addak (doubles the next consonant)
    ("guru", 0x72): ("skip", ""), ("guru", 0x73): ("skip", ""),  # vowel carriers
    ("mlym", 0x02): ("dead", "m"),     # Malayalam anusvara is a plain "m" (കേരളം -> keralam)
    ("mlym", 0x4E): ("dead", "r"),     # dot reph
    ("mlym", 0x7A): ("dead", "n"), ("mlym", 0x7B): ("dead", "n"), ("mlym", 0x7C): ("dead", "r"),
    ("mlym", 0x7D): ("dead", "l"), ("mlym", 0x7E): ("dead", "l"), ("mlym", 0x7F): ("dead", "k"),
    ("deva", 0x50): ("word", "om"), ("taml", 0x50): ("word", "om"),
    ("deva", 0x72): ("vowel", "a"),
}
_LABIAL = ("p", "b", "m", "f")
_INDIC = re.compile(r"[ऀ-ൿ]")
_INDIC_RUN = re.compile(r"[ऀ-ൿ‌‍]+")


def _script(ch):
    cp = ord(ch)
    base = cp & ~0x7F
    return _BLOCKS.get(base), cp - base


def has_indic(s: str) -> bool:
    return bool(_INDIC.search(s))


def _word(run: str) -> str:
    """Transliterate one run of Indic characters (one word)."""
    units = []  # [consonant, vowel, inherent]
    script = None
    for ch in run:
        sc, off = _script(ch)
        if sc is None:  # ZWJ / ZWNJ
            continue
        script = script or sc
        kind, val = _OVERRIDES.get((sc, off), (None, None))
        if kind == "skip":
            continue
        if kind == "cons" or (kind is None and off in _CONS):
            units.append([val if kind else _CONS[off], "a", True, False])
        elif kind == "dead":
            units.append([val, "", False, False])
        elif kind == "word":
            units.append(["", val, False, False])
        elif kind == "vowel" or (kind is None and off in _VOWELS):
            units.append(["", val if kind else _VOWELS[off], False, False])
        elif kind == "nasal" or (kind is None and off in _NASAL):
            units.append(["n", "", False, False])
        elif off == _VISARGA:
            units.append(["h", "", False, False])
        elif off == _NUKTA:
            if units and units[-1][0]:
                units[-1][0] = _NUKTA_OF.get(units[-1][0], units[-1][0])
        elif off == _VIRAMA:
            if units and units[-1][2]:
                units[-1][1:] = ["", False, True]  # half consonant: first part of a conjunct
        elif off in _MATRAS:
            if units and units[-1][2]:
                units[-1][1], units[-1][2] = _MATRAS[off], False
            else:
                units.append(["", _MATRAS[off], False, False])
        elif 0x66 <= off <= 0x6F:
            units.append([str(off - 0x66), "", False, False])
        # danda, accents, stress marks, avagraha ...: dropped
    if script in _SCHWA_DELETION and len(units) > 1:
        # final schwa goes, except after a conjunct: महाराष्ट्र -> maharashtra, आदित्य -> aditya
        if units[-1][2] and not units[-2][3]:
            units[-1][1] = ""
        # Ohala's rule, right to left: V C [a] C V  ->  V C C V
        for i in range(len(units) - 2, 0, -1):
            u, prev, nxt = units[i], units[i - 1], units[i + 1]
            if u[2] and u[1] == "a" and prev[1] and nxt[0] and nxt[1]:
                u[1] = ""
    out = []
    for i, (c, v, _, _) in enumerate(units):
        if c == "n" and not v and i + 1 < len(units) and units[i + 1][0].startswith(_LABIAL):
            c = "m"  # anusvara before a labial
        out.append(c + v)
    return "".join(out)


class Transliterator:
    """Rules + optional learned word dictionary. Call it on any string; Latin text passes through."""

    def __init__(self, dictionary: dict = None):
        self.dictionary = dictionary or {}
        self._cache = {}

    def word(self, w: str) -> str:
        r = self._cache.get(w)
        if r is None:
            r = self.dictionary.get(w)
            if r is None:
                r = _word(w)
            if len(self._cache) < 2_000_000:
                self._cache[w] = r
        return r

    def __call__(self, s: str) -> str:
        if not s or not _INDIC.search(s):
            return s
        s = unicodedata.normalize("NFC", s)
        return _INDIC_RUN.sub(lambda m: self.word(m.group(0)), s)


def _latin_tokens(s: str):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return [t for t in re.split(r"[^a-z0-9]+", s) if len(t) > 1]


def learn_dictionary(pairs, min_entities: int = 20, min_share: float = 0.5, min_sim: float = 0.45) -> dict:
    """Learn {Indic word: Latin word} from (Latin text, text with Indic words) pairs of true matches.

    Each Indic word is aligned with the most similar Latin token of its matched text (similarity to the
    rule-based transliteration), and the dictionary keeps a word only when it was seen with >= min_entities
    different partners' texts and one Latin form wins >= min_share of them. That keeps shared vocabulary
    (limited, private, software, state names) and drops one-off business names.
    """
    from rapidfuzz import fuzz
    from rapidfuzz.process import extractOne

    votes = defaultdict(Counter)
    for latin, indic in pairs:
        if not latin or not indic or not _INDIC.search(indic):
            continue
        toks = _latin_tokens(latin)
        if not toks:
            continue
        for w in {m.group(0) for m in _INDIC_RUN.finditer(unicodedata.normalize("NFC", indic))}:
            best = extractOne(_word(w), toks, scorer=fuzz.ratio)
            if best and best[1] >= 100 * min_sim:
                votes[w][best[0]] += 1
    out = {}
    for w, c in votes.items():
        n = sum(c.values())
        tok, k = c.most_common(1)[0]
        if n >= min_entities and k / n >= min_share:
            out[w] = tok
    return out


def _self_test():
    t = Transliterator()
    assert t("मार्केटिंग") == "marketing", t("मार्केटिंग")
    assert t("लिमिटेड") == "limited", t("लिमिटेड")
    assert t("प्राइवेट") == "praivet", t("प्राइवेट")
    assert t("उत्तर प्रदेश") == "uttar pradesh", t("उत्तर प्रदेश")
    assert t("Om हेल्थकेयर") == "Om helthakeyar", t("Om हेल्थकेयर")
    assert t("కర్ణాటక") == "karnataka", t("కర్ణాటక")       # Telugu keeps the final vowel
    assert t("லிமிடெட்") == "limitet", t("லிமிடெட்")        # Tamil has no separate voiced stops
    assert t("ગુજરાત") == "gujrat", t("ગુજરાત")            # rules say "gujrat"; the dictionary learns "gujarat"
    assert t("राजकुमार नमकीन") == "rajkumar namkin", t("राजकुमार नमकीन")
    assert t("महाराष्ट्र आदित्य") == "maharashtra aditya", t("महाराष्ट्र आदित्य")
    assert t("കേരളം") == "keralam", t("കേരളം")
    assert t("Sree Software") == "Sree Software"


_self_test()
