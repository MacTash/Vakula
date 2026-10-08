"""Deterministic extraction from source text.

Everything here is a pure function of the string it is given: no model, no
network, no randomness, no learning. That is deliberate. A 0.6B model asked to
pull out entities invents them, and an invented entity quietly becomes evidence
downstream. So Vakula only recognises what :mod:`vakula.gazetteer` already knows
and reports nothing at all otherwise. An unrecognised name yields UNKNOWN, which
is a safe answer; a hallucinated one is not.

The stored source text is never modified. HTML is stripped only in a local
copy used for matching, so what the user reads and what an assessment cites stay
byte-for-byte what the source returned.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone

from vakula import gazetteer, storage

# Spans this long are still matched in full; the cap exists only so a runaway
# document cannot dominate the event loop.
MAX_TEXT = 2_000_000

NUMBER_WORDS = {
    # "zero" is a quantity, not a denial: "zero vessels departed" asserts zero
    # rather than negating something. "no" and "none" are deliberately NOT here
    # even though they read as zero, because "none of the two vessels" would
    # then become a confident 0 the source never actually stated.
    "zero": 0,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "hundred": 100,
}

# Curated units. A bare number is not a quantity until it is next to one of
# these, which keeps "three sources" and "2026" out of the results.
UNIT_NOUNS = {
    "vessel": ("vessel", "vessels", "warship", "warships", "ship", "ships", "frigate", "frigates",
               "destroyer", "destroyers", "corvette", "corvettes", "naval ship", "naval ships"),
    "aircraft": ("aircraft", "plane", "planes", "jet", "jets", "fighter", "fighters",
                 "bomber", "bombers", "helicopter", "helicopters", "drone", "drones"),
    "soldier": ("soldier", "soldiers", "troop", "troops", "trooper", "troopers", "personnel",
                "personnel", "marine", "marines", "infantryman", "infantrymen"),
    "vehicle": ("vehicle", "vehicles", "truck", "trucks", "armoured vehicle", "armoured vehicles",
                "armored vehicle", "armored vehicles", "tank", "tanks", "convoy", "convoys"),
    "missile": ("missile", "missiles", "rocket", "rockets", "torpedo", "torpedoes"),
    "strike": ("strike", "strikes", "airstrike", "airstrikes", "sortie", "sorties"),
    "port": ("port", "ports", "harbour", "harbours", "harbor", "harbors", "terminal", "terminals"),
    "facility": ("facility", "facilities", "plant", "plants", "refinery", "refineries",
                 "pipeline", "pipelines", "depot", "depots", "base", "bases"),
}

_NEGATION_CUES = (
    "no", "not", "none", "never", "without", "denied", "denies", "deny",
    "refuted", "refutes", "rejected", "rejects", "dismissed", "dismisses",
    "contradicted", "retracted", "unconfirmed", "no additional", "did not", "didn't",
    "was not", "were not", "is not", "are not", "cannot", "unable",
)

# Script blocks mapped to the language they most likely indicate. A shared block
# (Arabic covers Arabic, Persian and Urdu) resolves to the macrolanguage rather
# than claiming a specific one.
_SCRIPTS = (
    ("he", re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]")),
    ("ar", re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")),
    ("hi", re.compile(r"[\u0900-\u097F]")),
    ("bn", re.compile(r"[\u0980-\u09FF]")),
    ("th", re.compile(r"[\u0E00-\u0E7F]")),
    ("ja", re.compile(r"[\u3040-\u309F\u30A0-\u30FF]")),
    ("ko", re.compile(r"[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]")),
    ("zh", re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")),
    ("ru", re.compile(r"[\u0400-\u04FF]")),
    ("el", re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")),
)

_TAG = re.compile(r"<[^>]{0,4096}>")
_HANDLE = re.compile(r"(?<![\w@])@([A-Za-z0-9_]{1,15})\b")
_WHITESPACE = re.compile(r"[ \t  - \r\f\v]+")


@dataclass(frozen=True)
class Mention:
    """One curated name found in the text, with every occurrence collapsed."""

    kind: str          # LOCATION or ENTITY
    key: str           # gazetteer key, e.g. loc:red-sea
    label: str         # canonical name
    matched: str       # the surface form that matched
    count: int         # occurrences in this document
    start: int = -1
    end: int = -1

    @property
    def mention_count(self) -> int:
        return self.count


@dataclass(frozen=True)
class Quantity:
    """A number next to a curated unit noun, with polarity."""

    value: int
    unit: str
    matched: str
    negated: bool = False
    start: int = -1
    end: int = -1
    context: str = ""

    @property
    def is_disputed(self) -> bool:
        """A quantity that a source denies is the interesting kind to keep."""
        return self.negated


@dataclass(frozen=True)
class Extraction:
    """Everything Vakula could determine from a document without guessing."""

    language: str = ""
    locations: tuple[Mention, ...] = ()
    entities: tuple[Mention, ...] = ()
    handles: tuple[str, ...] = ()
    quantities: tuple[Quantity, ...] = ()
    timestamp: str = ""

    @property
    def has_recognised_entities(self) -> bool:
        return bool(self.locations or self.entities or self.handles)

    def location_keys(self) -> tuple[str, ...]:
        return tuple(mention.key for mention in self.locations)

    def entity_keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys([*(m.key for m in self.entities), *self.handles]))


def clean_text(text: str) -> str:
    """A local copy for matching. The stored text is never modified."""
    if not isinstance(text, str):
        return ""
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT]
    plain = _TAG.sub(" ", text)
    plain = html.unescape(plain)
    plain = unicodedata.normalize("NFC", plain)
    return _WHITESPACE.sub(" ", plain)


def detect_language(text: str) -> str:
    """Guess a language from its script, or return "" when unsure.

    Latin script returns "" on purpose. Distinguishing English from Dutch,
    German or Norwegian needs a lexicon, and a wrong confident answer is worse
    than admitting ignorance.
    """
    plain = clean_text(text)
    if not plain.strip():
        return ""
    best, best_hits = "", 0
    for code, pattern in _SCRIPTS:
        hits = len(pattern.findall(plain))
        if hits > best_hits:
            best, best_hits = code, hits
    # A couple of stray characters from a quoted post is not a language.
    return best if best_hits >= 3 else ""


def parse_timestamp(value: str) -> datetime | None:
    """Parse the timestamp shapes real backends emit, or return None.

    Ambiguous input returns None rather than a guess, because a wrong timestamp
    silently corrupts every proximity calculation in the fusion stage.
    """
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d{10}", text):
        return datetime.fromtimestamp(int(text), tz=timezone.utc)
    if re.fullmatch(r"\d{13}", text):
        return datetime.fromtimestamp(int(text) / 1000, tz=timezone.utc)
    cleaned = text.replace("Z", "+00:00").replace("z", "+00:00")
    for candidate in (cleaned, cleaned.replace(" ", "T", 1)):
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    # X and friends: "Wed Oct 08 12:00:00 +0000 2026"
    for pattern in ("%a %b %d %H:%M:%S %z %Y", "%a %b %d %H:%M:%S %Y",
                    "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%d %b %Y %H:%M:%S %z"):
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def normalise_timestamp(value: str) -> str:
    """ISO-8601 UTC form, or "" when the input cannot be read."""
    parsed = parse_timestamp(value)
    return parsed.astimezone(timezone.utc).isoformat() if parsed else ""


def _alias_index() -> tuple[tuple[re.Pattern, str, str, str], ...]:
    compiled = []
    for kind, key, phrase in gazetteer.all_phrases():
        pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.IGNORECASE)
        compiled.append((pattern, kind, key, phrase))
    return tuple(compiled)


_ALIAS_INDEX: tuple[tuple[re.Pattern, str, str, str], ...] | None = None


def _index() -> tuple[tuple[re.Pattern, str, str, str], ...]:
    global _ALIAS_INDEX
    if _ALIAS_INDEX is None:
        _ALIAS_INDEX = _alias_index()
    return _ALIAS_INDEX


def find_mentions(text: str) -> tuple[tuple[Mention, ...], tuple[Mention, ...]]:
    """Find curated locations and entities. Unknown names yield nothing."""
    plain = clean_text(text)
    if not plain:
        return (), ()
    taken: list[tuple[int, int]] = []
    found: dict[tuple[str, str], Mention] = {}
    labels = {entry["key"]: entry["name"]
              for entry in (*gazetteer.LOCATION_SEED, *gazetteer.ENTITY_SEED)}
    # The index is sorted longest-first, so "Eastern Mediterranean" is claimed
    # before "Mediterranean" if both ever appear.
    for pattern, kind, key, phrase in _index():
        for match in pattern.finditer(plain):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in taken):
                continue
            taken.append(span)
            existing = found.get((kind, key))
            if existing:
                found[(kind, key)] = Mention(
                    kind=existing.kind, key=existing.key, label=existing.label,
                    matched=existing.matched, count=existing.count + 1,
                    start=existing.start, end=existing.end)
            else:
                found[(kind, key)] = Mention(
                    kind=kind, key=key, label=labels.get(key, phrase), matched=phrase,
                    count=1, start=span[0], end=span[1])
    locations = tuple(m for (kind, _), m in found.items() if kind == "LOCATION")
    entities = tuple(m for (kind, _), m in found.items() if kind == "ENTITY")
    return locations, entities


def find_handles(text: str) -> tuple[str, ...]:
    """Account handles, deduplicated. Recorded as OTHER, never as a person."""
    seen, ordered = set(), []
    for match in _HANDLE.finditer(clean_text(text)):
        handle = f"@{match.group(1)}"
        if handle.lower() not in seen:
            seen.add(handle.lower())
            ordered.append(handle)
    return tuple(ordered)


def _negated_before(text: str, start: int, window: int = 70) -> bool:
    """True when a negation cue sits just before the span.

    The cue has to end near the quantity. "No additional vessels departed" is
    negated; an unrelated "no" in the previous sentence is not, which is why
    this looks backwards a short distance rather than at the whole document.
    """
    prefix = text[max(0, start - window):start].lower()
    for cue in _NEGATION_CUES:
        index = prefix.rfind(cue)
        if index < 0:
            continue
        before = prefix[:index].rstrip()
        # The cue must be its own word, so "know" never counts as "no".
        if cue == "no" and before and before[-1].isalpha():
            continue
        tail = prefix[index + len(cue):].strip()
        if len(tail) <= 40:
            return True
    return False


def find_quantities(text: str) -> tuple[Quantity, ...]:
    """Numbers attached to curated unit nouns, each with its polarity."""
    plain = clean_text(text)
    if not plain:
        return ()
    units = sorted({noun for nouns in UNIT_NOUNS.values() for noun in nouns},
                   key=len, reverse=True)
    number_alternatives = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
    unit_alternatives = "|".join(re.escape(noun) for noun in units)
    pattern = re.compile(
        rf"(?<!\w)(?P<number>\d{{1,4}}|{number_alternatives})"
        rf"(?:\s+(?:[A-Za-z][\w-]*)){{0,2}}"
        rf"\s+(?P<unit>{unit_alternatives})(?!\w)",
        re.IGNORECASE)
    quantities = []
    for match in pattern.finditer(plain):
        token = match.group("number").lower()
        value = int(token) if token.isdigit() else NUMBER_WORDS[token]
        quantities.append(Quantity(
            value=value,
            unit=_canonical_unit(match.group("unit")),
            matched=match.group(0),
            negated=_negated_before(plain, match.start()),
            start=match.start(),
            end=match.end(),
            context=plain[max(0, match.start() - 60):match.end() + 40].strip(),
        ))
    return tuple(quantities)


def _canonical_unit(matched: str) -> str:
    lowered = matched.lower()
    for canonical, nouns in UNIT_NOUNS.items():
        if lowered in nouns:
            return canonical
    return "other"


def extract(text: str, *, timestamp: str = "") -> Extraction:
    """Run every deterministic pass over one document."""
    locations, entities = find_mentions(text)
    return Extraction(
        language=detect_language(text),
        locations=locations,
        entities=entities,
        handles=find_handles(text),
        quantities=find_quantities(text),
        timestamp=normalise_timestamp(timestamp) if timestamp else "",
    )


def enrich_observation(observation_id: int) -> Extraction | None:
    """Extract from a stored observation and link what was recognised.

    Reads the stored text, never rewrites it. Nothing is linked that the
    gazetteer did not already know, so a link in the database is always a fact
    about a curated entry rather than a guess.
    """
    record = storage.get_observation(observation_id)
    if record is None:
        return None
    extraction = extract(record.get("content", ""), timestamp=record.get("timestamp", ""))
    storage.link_mentions(observation_id, extraction)
    return extraction
