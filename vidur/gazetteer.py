"""Curated knowledge about places and actors, kept deliberately small.

This is hand-maintained and offline. Vidur does not guess: if a name is not in
here, extraction returns nothing for it rather than inventing an entity. A
gazetteer that grows by guessing stops being evidence and starts being
hallucination, so the list stays short and every entry is something a person
chose on purpose.

Aliases exist so that "the Taiwan Strait", "Taiwan Strait" and "Taiwan Straits"
resolve to one place. Longest aliases are matched first so a specific name wins
over a shorter one it contains.

Coordinates are approximate centroids with a radius, never a precise fix.
Omitting coordinates entirely is valid; a location does not have to be mappable
to be real.
"""

from __future__ import annotations

# key, name, aliases, country, region, location_type, latitude, longitude, radius_km
LOCATION_SEED: tuple[dict, ...] = (
    {
        "key": "loc:taiwan-strait",
        "name": "Taiwan Strait",
        "aliases": ("the Taiwan Strait", "Taiwan Straits", "Formosa Strait"),
        "country": "TW",
        "region": "East Asia",
        "location_type": "WATERBODY",
        "latitude": 24.5,
        "longitude": 119.5,
        "radius_km": 200.0,
    },
    {
        "key": "loc:eastern-mediterranean",
        "name": "Eastern Mediterranean",
        "aliases": ("the Eastern Mediterranean", "East Mediterranean",
                    "Eastern Mediterranean basin", "eastern Mediterranean"),
        # A basin spans several states, so no single country is asserted.
        "country": "",
        "region": "Eastern Mediterranean",
        "location_type": "REGION",
        "latitude": 34.5,
        "longitude": 33.0,
        "radius_km": 1200.0,
    },
    {
        "key": "loc:gaza",
        "name": "Gaza",
        "aliases": ("Gaza City", "Gaza Strip", "the Strip of Gaza", "Strip of Gaza"),
        # Country is left empty on purpose. Vidur records where something was
        # reported from and does not adjudicate sovereignty.
        "country": "",
        "region": "Eastern Mediterranean",
        "location_type": "CITY",
        "latitude": 31.5,
        "longitude": 34.47,
        "radius_km": 30.0,
    },
    {
        "key": "loc:delhi",
        "name": "Delhi",
        "aliases": ("New Delhi", "National Capital Territory of Delhi",
                    "NCT of Delhi", "Delhi NCR"),
        "country": "IN",
        "region": "South Asia",
        "location_type": "CITY",
        "latitude": 28.61,
        "longitude": 77.21,
        "radius_km": 50.0,
    },
    {
        "key": "loc:fujian",
        "name": "Fujian",
        "aliases": ("Fujian Province", "Fujian province", "Fujian coastal region"),
        "country": "CN",
        "region": "East China",
        "location_type": "REGION",
        "latitude": 26.0,
        "longitude": 118.0,
        "radius_km": 400.0,
    },
    {
        "key": "loc:red-sea",
        "name": "Red Sea",
        "aliases": ("the Red Sea", "Southern Red Sea", "Red Sea corridor",
                    "Red Sea shipping corridor"),
        "country": "",
        "region": "Red Sea",
        "location_type": "WATERBODY",
        "latitude": 20.0,
        "longitude": 38.5,
        "radius_km": 1800.0,
    },
)

# key, name, entity_type, country, aliases
ENTITY_SEED: tuple[dict, ...] = (
    {"key": "ent:country:cn", "name": "China", "entity_type": "COUNTRY", "country": "CN",
     "aliases": ("PRC", "People's Republic of China", "Mainland China")},
    {"key": "ent:country:in", "name": "India", "entity_type": "COUNTRY", "country": "IN",
     "aliases": ("Republic of India",)},
    {"key": "ent:country:il", "name": "Israel", "entity_type": "COUNTRY", "country": "IL",
     "aliases": (),
     },
    {"key": "ent:country:eg", "name": "Egypt", "entity_type": "COUNTRY", "country": "EG",
     "aliases": ("Arab Republic of Egypt",)},
    {"key": "ent:country:gr", "name": "Greece", "entity_type": "COUNTRY", "country": "GR",
     "aliases": ("Hellenic Republic",)},
    {"key": "ent:country:tr", "name": "Turkey", "entity_type": "COUNTRY", "country": "TR",
     "aliases": ("Turkiye", "Republic of Turkey")},
    {"key": "ent:country:sy", "name": "Syria", "entity_type": "COUNTRY", "country": "SY",
     "aliases": ("Syrian Arab Republic",)},
    {"key": "ent:country:lb", "name": "Lebanon", "entity_type": "COUNTRY", "country": "LB",
     "aliases": ("Lebanese Republic",)},
    {"key": "ent:country:jo", "name": "Jordan", "entity_type": "COUNTRY", "country": "JO",
     "aliases": ("Hashemite Kingdom of Jordan",)},
    {"key": "ent:country:sa", "name": "Saudi Arabia", "entity_type": "COUNTRY", "country": "SA",
     "aliases": ("KSA", "Kingdom of Saudi Arabia")},
    {"key": "ent:country:ye", "name": "Yemen", "entity_type": "COUNTRY", "country": "YE",
     "aliases": ("Republic of Yemen",)},
    {"key": "ent:country:sd", "name": "Sudan", "entity_type": "COUNTRY", "country": "SD",
     "aliases": ("Republic of the Sudan",)},
    {"key": "ent:country:us", "name": "United States", "entity_type": "COUNTRY", "country": "US",
     "aliases": ("USA", "United States of America", "US")},
    {"key": "ent:country:ru", "name": "Russia", "entity_type": "COUNTRY", "country": "RU",
     "aliases": ("Russian Federation",)},
    {"key": "ent:country:ua", "name": "Ukraine", "entity_type": "COUNTRY", "country": "UA",
     "aliases": (),
     },
    {"key": "ent:country:tw", "name": "Taiwan", "entity_type": "COUNTRY", "country": "TW",
     "aliases": ("Republic of China", "Chinese Taipei")},
    {"key": "ent:org:un", "name": "United Nations", "entity_type": "ORGANIZATION",
     "country": "", "aliases": ("UN",)},
    {"key": "ent:org:nato", "name": "NATO", "entity_type": "MILITARY_UNIT",
     "country": "", "aliases": ("North Atlantic Treaty Organization",)},
    {"key": "ent:org:eu", "name": "European Union", "entity_type": "ORGANIZATION",
     "country": "", "aliases": ("EU",)},
    {"key": "ent:org:iaea", "name": "IAEA", "entity_type": "ORGANIZATION",
     "country": "", "aliases": ("International Atomic Energy Agency",)},
    {"key": "ent:org:opec", "name": "OPEC", "entity_type": "ORGANIZATION",
     "country": "", "aliases": ("Organization of the Petroleum Exporting Countries",)},
)

# Handles are accounts, which may be a person, an outlet or a bot. Vidur records
# them as OTHER rather than guessing which.
HANDLE_ALIASES = ("OTHER",)


def alias_phrases(entry: dict) -> tuple[str, ...]:
    """Every surface form that should resolve to this entry, name included."""
    return tuple(dict.fromkeys((entry["name"], *entry.get("aliases", ()))))


def all_phrases() -> tuple[tuple[str, str, str], ...]:
    """(kind, key, phrase) for every location and entity alias, name included."""
    phrases: list[tuple[str, str, str]] = []
    for entry in LOCATION_SEED:
        for phrase in alias_phrases(entry):
            phrases.append(("LOCATION", entry["key"], phrase))
    for entry in ENTITY_SEED:
        for phrase in alias_phrases(entry):
            phrases.append(("ENTITY", entry["key"], phrase))
    return tuple(sorted(phrases, key=lambda item: len(item[2]), reverse=True))


def seed(store=None) -> dict[str, int]:
    """Insert the curated entries. Idempotent, and safe to run on every launch.

    Existing rows are refreshed from the gazetteer but keep their ids and any
    columns Vidur manages elsewhere, so re-seeding never duplicates a place or
    resets what has been recorded about it.
    """
    if store is None:
        from vidur import storage as store
    seeded = {"locations": 0, "entities": 0}
    for entry in LOCATION_SEED:
        if store.upsert_location(entry):
            seeded["locations"] += 1
    for entry in ENTITY_SEED:
        if store.upsert_entity(entry):
            seeded["entities"] += 1
    return seeded
