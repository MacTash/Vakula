import time

from vidur import extract, gazetteer, storage
from vidur.extract import _clean, find_handles, find_mentions, find_quantities


# --- the curated locations -------------------------------------------------

def _keys(mentions):
    return sorted(mention.key for mention in mentions)


def test_every_curated_location_is_recognised():
    cases = {
        "Vessel traffic in the Taiwan Strait has doubled.": "loc:taiwan-strait",
        "Strikes reported across the Eastern Mediterranean.": "loc:eastern-mediterranean",
        "Airstrikes reported in Gaza City overnight.": "loc:gaza",
        "Protests continue in New Delhi this week.": "loc:delhi",
        "Fujian Province ordered new restrictions.": "loc:fujian",
        "Ships are rerouting away from the Red Sea.": "loc:red-sea",
    }
    for text, expected in cases.items():
        locations, _ = find_mentions(text)
        assert expected in _keys(locations), (text, _keys(locations))


def test_location_aliases_resolve_to_the_same_place():
    aliases = {
        "loc:gaza": ("Gaza", "Gaza City", "Gaza Strip", "the Strip of Gaza", "Strip of Gaza"),
        "loc:delhi": ("Delhi", "New Delhi", "NCT of Delhi", "Delhi NCR"),
        "loc:eastern-mediterranean": ("Eastern Mediterranean", "East Mediterranean",
                                      "the Eastern Mediterranean", "eastern Mediterranean"),
        "loc:red-sea": ("Red Sea", "the Red Sea", "Southern Red Sea", "Red Sea corridor"),
        "loc:fujian": ("Fujian", "Fujian Province", "Fujian province"),
        "loc:taiwan-strait": ("Taiwan Strait", "the Taiwan Strait", "Taiwan Straits"),
    }
    for key, forms in aliases.items():
        for form in forms:
            locations, _ = find_mentions(f"Report from {form} this morning.")
            assert _keys(locations) == [key], (form, _keys(locations))


def test_duplicate_mentions_collapse_into_one_entry():
    locations, entities = find_mentions(
        "Gaza. Gaza again. More from Gaza City, and Gaza.")
    gaza = next(mention for mention in locations if mention.key == "loc:gaza")
    assert gaza.count == 4, gaza.count
    assert len([mention for mention in locations if mention.key == "loc:gaza"]) == 1
    assert len(locations) == 1


def test_longest_alias_wins_over_a_shorter_one():
    # "Eastern Mediterranean" must not be swallowed by any shorter alias.
    locations, _ = find_mentions("The Eastern Mediterranean basin is tense.")
    assert _keys(locations) == ["loc:eastern-mediterranean"]


def test_unknown_places_and_actors_yield_nothing():
    # Conservative by design: an unfamiliar proper noun is UNKNOWN, never invented.
    locations, entities = find_mentions(
        "Zephyria reported activity near Brant Hollow while the Kestrel Accord was discussed.")
    assert locations == ()
    assert entities == ()


def test_entity_matching_recognises_curated_actors():
    _, entities = find_mentions("NATO and the United Nations met; China and India were briefed.")
    keys = set(_keys(entities))
    assert {"ent:org:nato", "ent:org:un", "ent:country:cn", "ent:country:in"} <= keys


def test_handles_are_recorded_as_other_never_as_a_person():
    handles = find_handles("cc: @naval_watch and @IAF_ official, plus @naval_watch again.")
    assert handles == ("@naval_watch", "@IAF_")
    assert find_handles("email me at know@example.com") == ()


# --- quantities ------------------------------------------------------------

def test_three_naval_vessels_is_extracted_with_its_unit():
    quantities = find_quantities("Country A deployed three naval vessels.")
    assert len(quantities) == 1
    assert quantities[0].value == 3
    assert quantities[0].unit == "vessel"
    assert quantities[0].negated is False


def test_digit_and_alternative_unit_forms():
    quantities = find_quantities("2 ships and four fighter jets departed.")
    by_unit = {quantity.unit: quantity.value for quantity in quantities}
    assert by_unit == {"vessel": 2, "aircraft": 4}


def test_negation_is_detected_and_kept():
    denied = find_quantities("Officials deny 12 vessels departed the port.")
    assert denied and denied[0].negated is True
    assert denied[0].value == 12

    affirmed = find_quantities("Three vessels departed the port.")
    assert affirmed[0].negated is False

    assert find_quantities("Officials denied that three vessels departed.")[0].negated is True


def test_zero_is_a_quantity_and_not_a_denial():
    quantities = find_quantities("Zero vessels departed the port.")
    assert quantities and quantities[0].value == 0
    assert quantities[0].negated is False


def test_a_denial_carrying_no_number_yields_no_quantity():
    # "No additional vessels" denies a proposition without stating a number.
    # Batch 2 records nothing rather than inventing one; proposition-level
    # denial is the fusion layer's job in Batch 3.
    assert find_quantities("No additional vessels departed the port.") == ()
    assert find_quantities("None of the two vessels were counted.")[0].value == 2


def test_negation_does_not_fire_on_a_substring():
    # "know" contains "no"; that is not a denial.
    quantities = find_quantities("I know three vessels departed.")
    assert quantities[0].negated is False


def test_bare_numbers_are_not_quantities():
    assert find_quantities("There were 2026 reports and 3 opinions.") == ()


# --- timestamps ------------------------------------------------------------

def test_timestamp_formats_real_backends_emit():
    assert extract.normalise_timestamp("2026-10-08T09:37:42Z").startswith("2026-10-08T09:37:42")
    assert extract.normalise_timestamp("Wed Oct 08 09:37:42 +0000 2026").startswith("2026-10-08T09:37:42")
    assert extract.normalise_timestamp("2026-10-08 09:37:42").startswith("2026-10-08T09:37:42")
    assert extract.normalise_timestamp("1791449862").startswith("2026-10-08")
    assert extract.normalise_timestamp("2026-10-08") == "2026-10-08T00:00:00+00:00"


def test_unreadable_timestamps_are_not_guessed():
    for value in ("", "yesterday", "sometime last week", "2026-13-45", "not a date"):
        assert extract.parse_timestamp(value) is None, value


# --- hostile and awkward text ----------------------------------------------

def test_html_is_stripped_for_matching_but_never_mutates_the_source():
    body = '<div class="post"><p>Three vessels left <b>Port X</b></p><script>alert(1)</script></div>'
    result = extract.extract(body)
    assert "script" not in _clean(body).lower()
    assert result.quantities and result.quantities[0].value == 3
    assert body.startswith("<div") and "<script>" in body


def test_emoji_do_not_break_matching():
    result = extract.extract("🚢🌊 Three naval vessels departed Gaza 🇵🇸🚢 — reports say so 🛰️")
    assert any(quantity.value == 3 for quantity in result.quantities)
    assert "loc:gaza" in result.location_keys()
    assert result.locations[0].count == 1


def test_right_to_left_text_is_handled():
    arabic = extract.extract(" sle drei vessels ?")
    hebrew = extract.extract("שלוש שלושה ספינות")
    assert arabic.language == ""
    assert hebrew.language == "he"
    # A Latin-script document stays UNKNOWN rather than being called English.
    assert extract.detect_language("three naval vessels departed the port") == ""


def test_language_detection_only_claims_what_it_can_see():
    assert extract.detect_language("三艘军舰离开了港口，另外还有两艘") == "zh"
    assert extract.detect_language("ثلاث سفن غادرت الميناء اليوم") == "ar"
    assert extract.detect_language("Yunuslar ülkeyi terk etti") == ""
    assert extract.detect_language("") == ""


def test_two_hundred_kilobyte_body_stays_fast_and_correct():
    filler = "Background commentary without recognisable names. " * 2500
    body = (filler + "Red Sea shipping update. ") * 4
    assert len(body) > 200_000
    started = time.monotonic()
    result = extract.extract(body)
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, elapsed
    assert "loc:red-sea" in result.location_keys()
    assert result.locations[0].count == 4


def test_extraction_is_deterministic():
    body = "Three naval vessels left the Taiwan Strait near Gaza."
    first = extract.extract(body)
    second = extract.extract(body)
    assert first == second


# --- persistence -----------------------------------------------------------

def test_seeding_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    storage.init_db()
    gazetteer.seed(storage)
    first = (len(storage.list_locations()), len(storage.list_entities()))
    gazetteer.seed(storage)
    gazetteer.seed(storage)
    assert (len(storage.list_locations()), len(storage.list_entities())) == first
    assert first == (6, len(gazetteer.ENTITY_SEED))


def test_extraction_links_reach_the_observation_architecture(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    storage.init_db()
    gazetteer.seed(storage)
    body = ("Three naval vessels departed Port X near the Taiwan Strait. "
            "NATO confirmed the report from Gaza. No additional vessels left the Red Sea.")
    observation_id = storage.insert_observation(
        {"platform": "x", "backend": "twitter-cli", "source_key": "k1", "body": body,
         "published_at": "2026-10-01T09:00:00Z"}, source_type="SOCIAL")

    locations = {row["key"] for row in storage.locations_for_observation(observation_id)}
    entities = {row["key"] for row in storage.entities_for_observation(observation_id)}
    assert {"loc:taiwan-strait", "loc:gaza", "loc:red-sea"} <= locations
    assert "ent:org:nato" in entities
    # Two recognised locations leave the observation's own location unset.
    assert storage.get_observation(observation_id)["location"] is None


def test_single_location_becomes_the_observation_location(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    storage.init_db()
    gazetteer.seed(storage)
    observation_id = storage.insert_observation(
        {"platform": "x", "source_key": "k2", "body": "Fighting reported in Gaza."})
    assert storage.get_observation(observation_id)["location"] is not None


def test_re_enrichment_does_not_duplicate_links(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    storage.init_db()
    gazetteer.seed(storage)
    item = {"platform": "x", "source_key": "k3",
            "body": "Three vessels near Gaza. NATO confirmed the report."}
    observation_id = storage.insert_observation(item)
    assert len(storage.locations_for_observation(observation_id)) == 1
    assert len(storage.entities_for_observation(observation_id)) == 1
    for _ in range(3):
        extract.enrich_observation(observation_id)
    with storage.connect() as db:
        count = db.execute(
            "SELECT COUNT(*) FROM relationships WHERE from_kind='OBSERVATION' AND from_id=?",
            (observation_id,)).fetchone()[0]
    assert count == 2  # one location, one organisation, however often we re-run


def test_extraction_never_modifies_stored_content(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    storage.init_db()
    gazetteer.seed(storage)
    body = '<p>Three naval vessels left <b>Gaza</b> &amp; the Taiwan Strait</p>'
    observation_id = storage.insert_observation({"platform": "x", "source_key": "k4", "body": body})
    assert storage.get_observation(observation_id)["content"] == body
    assert storage.list_source_items()[0]["body"] == body


def test_extraction_does_not_pull_in_the_model(monkeypatch, tmp_path):
    # Extraction must stay offline and deterministic: no AI, no network.
    monkeypatch.setenv("VIDUR_DATA_DIR", str(tmp_path))
    storage.init_db()
    gazetteer.seed(storage)
    observation_id = storage.insert_observation(
        {"platform": "x", "source_key": "k5", "body": "Three vessels near Gaza."})
    assert storage.locations_for_observation(observation_id)
    assert not hasattr(extract, "research")
    assert not hasattr(extract, "AISettings")
