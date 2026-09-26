"""Text normalisation for business names and addresses.

Everything here is rule-based and works on countries that are not in the training data (France in
the test set): Indic scripts are transliterated (src/translit.py), accents are stripped with
unicodedata, and every country-specific table falls back to "no change" for an unknown country.

Noise seen in the official data and handled here:
  names     : Indic scripts, digits used as letters ("C0mpanies", "5ervices"), tags such as
              "(ID: 81759)", "#92781" or phone numbers, "@handle" / "#hashtag" / "domain.com" names,
              "<made-up name> formerly|aka|DBA <real name>", bracketed or repeated legal words
  addresses : Indic state names, literal "null" / "<NULL>", zero-padded house numbers, state
              names vs. codes ("Connecticut" / "CT", "Uttar Pradesh" / "UP"), "City of ..."
The data has no postal codes: 5-6 digit numbers are house numbers ("12718 Braewood Glen Lane").
"""
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .translit import Transliterator

# ---------------------------------------------------------------- name rules
# Variants -> one canonical token. Canonical forms are short so that
# "Corporation" and "Corp" become the same token.
_NAME_CANON = {
    "and": "and", "et": "and", "und": "and",
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "incorporated": "inc", "inc": "inc",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prv": "pvt",
    "company": "co", "co": "co", "cos": "co", "compagnie": "co", "cie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",
    "international": "intl", "intl": "intl", "internatl": "intl",
    "services": "svc", "service": "svc", "svcs": "svc", "svc": "svc", "srvcs": "svc",
    "technologies": "tech", "technology": "tech", "tech": "tech", "techs": "tech", "technol": "tech",
    "enterprises": "ent", "enterprise": "ent", "ent": "ent", "entp": "ent",
    "industries": "ind", "industry": "ind", "inds": "ind", "ind": "ind", "industrial": "ind",
    "manufacturing": "mfg", "mfg": "mfg", "manufacturers": "mfg", "manufacturer": "mfg",
    "management": "mgmt", "mgmt": "mgmt", "mgt": "mgmt",
    "associates": "assoc", "associate": "assoc", "assoc": "assoc", "assocs": "assoc", "association": "assoc",
    "brothers": "bros", "bros": "bros", "bro": "bros",
    "solutions": "soln", "solution": "soln", "soln": "soln", "solns": "soln",
    "systems": "sys", "system": "sys", "sys": "sys",
    "group": "grp", "grp": "grp", "groupe": "grp",
    "holdings": "hldg", "holding": "hldg", "hldgs": "hldg", "hldg": "hldg",
    "laboratories": "lab", "laboratory": "lab", "labs": "lab", "lab": "lab", "laboratoire": "lab", "laboratoires": "lab",
    "pharmaceuticals": "pharma", "pharmaceutical": "pharma", "pharma": "pharma", "pharmacie": "pharma", "pharmacy": "pharma",
    "trading": "trdg", "traders": "trdg", "trader": "trdg", "trdg": "trdg",
    "restaurant": "rest", "restaurants": "rest", "rest": "rest",
    "hospital": "hosp", "hospitals": "hosp", "hosp": "hosp",
    "center": "ctr", "centre": "ctr", "ctr": "ctr", "cntr": "ctr",
    "department": "dept", "dept": "dept",
    "national": "natl", "natl": "natl",
    "american": "amer", "amer": "amer",
    "consulting": "consult", "consultants": "consult", "consultancy": "consult", "consultant": "consult",
    "engineering": "engg", "engineers": "engg", "engg": "engg", "eng": "engg",
    "construction": "constr", "constructions": "constr", "constr": "constr",
    "society": "soc", "societe": "soc", "soc": "soc", "ste": "soc",
    "saint": "st", "st": "st", "sainte": "st",
    "mount": "mt", "mt": "mt",
    "the": "the", "le": "the", "la": "the", "les": "the",
}
# Tokens that describe the legal form, not the business -> removed in the "core" name.
_LEGAL = {
    "corp", "inc", "ltd", "pvt", "co", "llc", "llp", "lp", "plc", "pllc", "pc", "pa",
    "gmbh", "ag", "sa", "sas", "sasu", "sarl", "eurl", "snc", "sci", "sca", "bv", "nv",
    "opc", "the", "and",
}
_HONORIFIC = {"m", "s", "ms", "mss"}  # "M/s ABC Traders" (India)
# "<new name> formerly <old name>": the part after the marker is the name the other sources use
_MARKER = re.compile(r"\b(?:f\s*/\s*k\s*/\s*a|fka|formerly(?:\s+known\s+as)?|previously(?:\s+known\s+as)?|now\s+known\s+as"
                     r"|a\s*/\s*k\s*/\s*a|aka|d\s*/\s*b\s*/\s*a|dba|doing\s+business\s+as|t\s*/\s*a|trading\s+as)\b\s*:?")
_TAG_NOISE = re.compile(r"\(\s*id\s*:?\s*\d+\s*\)|#\s*\d+|(?<![a-z0-9])\d{7,}(?![a-z0-9])")  # (ID: 81759), #92781, phones
_HANDLE = re.compile(r"(?:^|(?<=\s))[@#](?=[a-z0-9])")                                          # @handle, #hashtag
_TLD = re.compile(r"(?<=[a-z0-9])\.(?:co\.in|com|net|org|in|co|fr|biz|info|io|us)\b")             # globalcardiology.com
_LEET = str.maketrans("01345678", "oleasgtb")
_ORDINAL_TOK = re.compile(r"^\d+(?:st|nd|rd|th|er|e|eme)$")

# ------------------------------------------------------------- address rules
_ADDR_CANON = {
    "street": "st", "st": "st", "str": "st", "strt": "st",
    "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave", "aven": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "boul": "blvd", "bld": "blvd",
    "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr", "drv": "dr",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "square": "sq", "sq": "sq",
    "parkway": "pkwy", "pkwy": "pkwy",
    "highway": "hwy", "hwy": "hwy",
    "circle": "cir", "cir": "cir",
    "terrace": "ter", "ter": "ter",
    "route": "rte", "rte": "rte",
    "rue": "rue", "r": "rue",
    "allee": "all", "all": "all", "allees": "all",
    "chemin": "ch", "ch": "ch", "chem": "ch",
    "impasse": "imp", "imp": "imp",
    "faubourg": "fbg", "fbg": "fbg", "fg": "fbg",
    "bis": "", "ter": "",  # "91 bis Rue" vs "91 RUE"
    "saint": "st", "sainte": "ste",  # "ST.-NAZAIRE" vs "Saint-Nazaire" (in the US: "St Louis" vs "Saint Louis")
    "suite": "ste", "ste": "ste",
    "floor": "fl", "flr": "fl", "fl": "fl", "etage": "fl",
    "building": "bldg", "bldg": "bldg", "bldng": "bldg",
    "apartment": "apt", "apt": "apt", "appt": "apt",
    "north": "n", "n": "n", "south": "s", "s": "s", "east": "e", "e": "e", "west": "w", "w": "w",
    "northeast": "ne", "ne": "ne", "northwest": "nw", "nw": "nw",
    "southeast": "se", "se": "se", "southwest": "sw", "sw": "sw",
    "near": "near", "nr": "near", "opposite": "opp", "opp": "opp", "behind": "behind", "bhd": "behind",
    "nagar": "nagar", "ngr": "nagar",
    "colony": "colony", "col": "colony",
    "sector": "sector", "sec": "sector", "sect": "sector",
    "cross": "cross", "crs": "cross",
    "main": "main", "mn": "main",
    "marg": "marg",
    "extension": "extn", "extn": "extn", "ext": "extn",
    "phase": "ph", "ph": "ph",
    "block": "blk", "blk": "blk",
    "plot": "plot", "plt": "plot",
    "number": "", "no": "", "num": "", "nr.": "", "door": "", "hno": "", "h": "",
    "mount": "mt", "mt": "mt",
    "fort": "ft", "ft": "ft",
    "point": "pt", "pt": "pt",
    "district": "dist", "dist": "dist", "distt": "dist",
    "post": "po", "po": "po",
    "pin": "", "pincode": "", "zip": "", "zipcode": "", "cedex": "",
    "township": "", "twp": "",
}
_LANDMARK = re.compile(r"\b(?:near|nr|opp|opposite|behind|next to|beside|adjacent to|in front of|landmark)\b[^,;]*")
_NULL = re.compile(r"<\s*null\s*>|\bnull\b")
_CITY_OF = re.compile(r"\b(?:city|town|village|township|borough)\s+of\b")
_ORDINAL = re.compile(r"(\d+)(?:st|nd|rd|th|er|e|eme|ème)\b")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")

# State / province names and codes -> one code, applied to whole comma-separated address components only
# ("Fawn Ct" stays a court, a lone "CT" component is Connecticut). Keyed by normalised country label;
# an unknown country (France) is left as is.
_US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california", "co": "colorado",
    "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas", "ky": "kentucky", "la": "louisiana",
    "me": "maine", "md": "maryland", "ma": "massachusetts", "mi": "michigan", "mn": "minnesota",
    "ms": "mississippi", "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia", "pr": "puerto rico",
}
_IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh", "jh": "jharkhand",
    "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur",
    "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland", "od": "odisha", "pb": "punjab", "rj": "rajasthan",
    "sk": "sikkim", "tn": "tamil nadu", "tg": "telangana", "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand", "wb": "west bengal", "an": "andaman and nicobar islands", "ch": "chandigarh",
    "dn": "dadra and nagar haveli and daman and diu", "dl": "delhi", "jk": "jammu and kashmir", "la": "ladakh",
    "ld": "lakshadweep", "py": "puducherry",
}
_IN_ALIASES = {"ct": "cg", "chattisgarh": "cg", "or": "od", "orissa": "od", "ts": "tg", "ua": "uk",
               "uttaranchal": "uk", "pondicherry": "py", "nct of delhi": "dl", "dilli": "dl",
               "andaman and nicobar": "an", "jammu kashmir": "jk", "daman and diu": "dn", "dadra and nagar haveli": "dn"}


def _state_table(codes: dict, aliases: dict = None) -> dict:
    """code / name / alias -> full name (not the code: a lone "CT" must not become the token for "Court")."""
    t = {}
    for code, name in codes.items():
        t[code] = t[name] = t[name.replace(" and ", " ")] = name
    t.update({a: codes[c] for a, c in (aliases or {}).items()})
    return t


_STATES = {"us": _state_table(_US_STATES), "usa": _state_table(_US_STATES), "united states": _state_table(_US_STATES),
           "india": _state_table(_IN_STATES, _IN_ALIASES)}


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _basic(s: str) -> str:
    s = strip_accents(str(s)).lower()
    s = s.replace("&", " and ").replace(" @ ", " at ")
    s = re.sub(r"['’`]", "", s)  # O'Brien -> obrien, L'Oreal -> loreal
    # dotted initialisms: "U.S.A." / "i.b.m" -> "usa" / "ibm"; other dots become separators later
    s = re.sub(r"\b[a-z](?:\.[a-z])+\b\.?", lambda m: m.group(0).replace(".", ""), s)
    return s


def _tokens(s: str):
    s = _NON_ALNUM.sub(" ", s)
    return [t for t in _SPACES.split(s) if t]


def _canon(tokens, table):
    out = []
    for t in tokens:
        c = table.get(t, t)
        if c and (not out or out[-1] != c):  # also drops repeats: "private private" -> "pvt"
            out.append(c)
    return out


def _unleet(tok: str) -> str:
    """Digits used as letters inside a word: c0mpanies -> companies, 5ervices -> services, deve1opers -> developers."""
    nd = sum(c.isdigit() for c in tok)
    if 1 <= nd <= 2 and len(tok) - nd >= 3 and not _ORDINAL_TOK.match(tok):
        return tok.translate(_LEET)
    return tok


# ------------------------------------------------------------------ public API
def clean_name(raw: str, tr: Transliterator) -> str:
    """Transliterated, lower-case, noise-free name with the words kept as written (input for text models)."""
    s = _basic(tr(str(raw)))
    s = _TAG_NOISE.sub(" ", s)
    s = _HANDLE.sub("", s)
    s = _TLD.sub(" ", s.replace("www.", " "))
    s = re.sub(r"\bpra\.?\s*li\b\.?", " pvt ltd ", s)  # Hindi "प्रा. लि." = Pvt. Ltd.
    s = " ".join(_unleet(t) for t in s.split())
    return _SPACES.sub(" ", s).strip(" -*>.<[]()")


def normalize_name(raw: str, tr: Transliterator):
    """Return (clean, full_norm, core_norm, core_alt).

    clean     : clean_name() -- words as written, for the text models
    full_norm : canonicalised name, legal words kept
    core_norm : legal suffixes / honorifics / 'the' / 'and' removed
    core_alt  : the other half of a "X formerly/aka/DBA Y" name, else == core_norm
    """
    clean = clean_name(raw, tr)
    s = re.sub(r"\bm\s*/\s*s\b", " ", clean)  # M/s prefix
    s = re.sub(r"\bpvt\s*\.?\s*ltd\b", " pvt ltd ", s)
    m = _MARKER.search(s)
    if m:
        before, after = s[:m.start()], s[m.end():]
        parts = [p for p in (after, before) if _tokens(p)] or [s]
    else:
        parts = [s]
    cores, fulls = [], []
    for p in parts[:2]:
        toks = _canon(_tokens(p), _NAME_CANON)
        fulls.append(" ".join(toks))
        core = [t for t in toks if t not in _LEGAL and t not in _HONORIFIC]
        cores.append(" ".join(core) if core else " ".join(toks))
    full = " ".join(fulls)
    core = cores[0]
    alt = cores[1] if len(cores) > 1 else core
    return clean, full, core, alt


def clean_address(raw: str, country_norm: str, tr: Transliterator) -> str:
    """Transliterated, lower-case address without null markers; state names/codes unified per component."""
    s = strip_accents(tr(str(raw))).lower()
    s = _NULL.sub(" ", s).replace("n°", " ").replace("nº", " ")  # French "N° 44" = number 44
    states = _STATES.get(country_norm)
    comps = []
    for c in s.split(","):
        c = _SPACES.sub(" ", c).strip(" .;:-")
        if not c:
            continue
        if states:
            key = _SPACES.sub(" ", _NON_ALNUM.sub(" ", c.replace("&", " and "))).strip()
            c = states.get(key, c)
        comps.append(c)
    return ", ".join(comps)


def normalize_address(raw: str, country_norm: str, tr: Transliterator):
    """Return (clean, addr_norm, addr_core, numbers, has_landmark).

    clean     : clean_address() -- for the text models
    addr_norm : canonicalised address tokens
    addr_core : addr_norm without landmark phrases ("near SBI ATM")
    numbers   : house / plot / unit numbers, leading zeros removed ("0100" -> "100")
    """
    clean = clean_address(raw, country_norm, tr)
    s = _basic(clean)
    s = _CITY_OF.sub(" ", s)
    s = _ORDINAL.sub(r"\1", s)              # "5th cross" -> "5 cross"
    s = re.sub(r"[/\-#]", " ", s)          # "12/3-A" -> "12 3 a"
    has_landmark = bool(_LANDMARK.search(s))
    no_landmark = _LANDMARK.sub(" ", s)
    strip0 = lambda toks: [t.lstrip("0") or "0" if t.isdigit() else t for t in toks]
    toks = strip0(_canon(_tokens(s), _ADDR_CANON))
    core_toks = strip0(_canon(_tokens(no_landmark), _ADDR_CANON))
    numbers = [t for t in core_toks if t.isdigit()]
    return clean, " ".join(toks), " ".join(core_toks), " ".join(numbers), has_landmark


def normalize_country(raw: str) -> str:
    s = strip_accents(str(raw)).lower().strip()
    return _SPACES.sub(" ", _NON_ALNUM.sub(" ", s)).strip()


def acronym(core: str) -> str:
    toks = core.split()
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


_COLS = ["name_clean", "name_norm", "name_core", "name_alt", "addr_clean", "addr_norm", "addr_core", "addr_nums",
         "has_landmark"]
_WORKER_TR = None


def _init_worker(dictionary):
    global _WORKER_TR
    _WORKER_TR = Transliterator(dictionary)


def _normalize_chunk(args):
    names, addrs, countries = args
    tr = _WORKER_TR or Transliterator()
    out = {c: [] for c in _COLS}
    for n, a, c in zip(names, addrs, countries):
        nc, nf, nco, nal = normalize_name(n, tr)
        ac, an, aco, anum, lm = normalize_address(a, c, tr)
        for k, v in zip(_COLS, (nc, nf, nco, nal, ac, an, aco, anum, lm)):
            out[k].append(v)
    return out


def prepare(df: pd.DataFrame, dictionary: dict = None, workers: int = None, chunk: int = 50_000) -> pd.DataFrame:
    """Add all normalised columns to a source dataframe (does not modify input). Runs on all CPU cores."""
    df = df.copy()
    for c in ("entity_id", "business_name", "business_address", "country"):
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].fillna("").astype(str)
    df["country_norm"] = [normalize_country(x) for x in df["country"].tolist()]
    names, addrs, ctry = df["business_name"].tolist(), df["business_address"].tolist(), df["country_norm"].tolist()
    jobs = [(names[i:i + chunk], addrs[i:i + chunk], ctry[i:i + chunk]) for i in range(0, len(df), chunk)]
    workers = workers or os.cpu_count() or 1
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), initializer=_init_worker,
                                 initargs=(dictionary or {},)) as ex:
            parts = list(ex.map(_normalize_chunk, jobs))
    else:
        _init_worker(dictionary or {})
        parts = [_normalize_chunk(j) for j in jobs]
    for c in _COLS:
        df[c] = [v for p in parts for v in p[c]]
    df["has_landmark"] = np.asarray(df["has_landmark"], dtype=bool)
    # transliterated text is a noisier match (the model should trust name similarity a bit less)
    df["name_indic"] = df["business_name"].str.contains(r"[ऀ-ൿ]", regex=True).to_numpy()
    df["addr_indic"] = df["business_address"].str.contains(r"[ऀ-ൿ]", regex=True).to_numpy()
    df["name_acr"] = [acronym(x) for x in df["name_core"].tolist()]
    df["name_first"] = [x.split()[0] if x else "" for x in df["name_core"].tolist()]
    df["full_text"] = (df["name_core"] + " " + df["addr_core"]).str.strip()
    df["source"] = df["entity_id"].str.slice(0, 2)
    return df.reset_index(drop=True)


def _self_test():
    tr = Transliterator()
    assert normalize_name("Fayelum formerly: Gardea Terra Inc", tr)[2] == "gardea terra"
    assert normalize_name("Keefe Charter LLC 5ervices", tr)[2] == "keefe charter svc"
    assert normalize_name("Marsh C0mpanies Ltd (ID: 81759)", tr)[2] == "marsh companies"
    assert normalize_name("GLOBALCARDIOLOGY.COM", tr)[2] == "globalcardiology"
    assert normalize_name("#globalcardiology", tr)[2] == "globalcardiology"
    assert normalize_name("Monarch (India) Private Private Limited Service - 6421593784", tr)[1] == "monarch india pvt ltd svc"
    assert normalize_address("CT, STAMFORD, 0100 WEDGEMERE RD", "us", tr)[2] == "connecticut stamford 100 wedgemere rd"
    assert normalize_address("100 Wedgemere Road, Stamford, Connecticut", "us", tr)[2] == "100 wedgemere rd stamford connecticut"
    assert normalize_address("12718 Braewood Glen Lane, TX, Houston", "us", tr)[3] == "12718"
    assert normalize_address("Vps-21, Ssi Noida, UP", "india", tr)[0].endswith("uttar pradesh")
    assert normalize_address("A-70, Delhi, NULL, DL", "india", tr)[0] == "a-70, delhi, delhi"
    assert normalize_address("Fawn Ct, Fort Worth, Texas", "us", tr)[2] == "fawn ct ft worth texas"
    # French: abbreviated street types, "bis", "St.-" for Saint-
    assert (normalize_address("13 - ALL. ODETTE DU PUIGAUDEAU, ST.-NAZAIRE", "france", tr)[2]
            == normalize_address("13 Allée Odette du Puigaudeau, Saint-Nazaire", "france", tr)[2])
    assert normalize_address("91 bis Rue Bellus", "france", tr)[2] == normalize_address("91 R. BELLUS", "france", tr)[2]


_self_test()
