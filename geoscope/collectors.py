"""Public-source collectors. They only access documented, unauthenticated feeds."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

import requests

from geoscope.storage import add_item

USER_AGENT = "Geoscope/0.2.0 (+https://github.com/MacTash/Geoscope)"


def _get(url: str, *, params: dict | None = None) -> requests.Response:
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    response.raise_for_status()
    return response


def earthquakes(min_magnitude: float = 4.5, days: int = 7) -> list[dict]:
    days = max(1, min(int(days), 365))
    data = _get("https://earthquake.usgs.gov/fdsnws/event/1/query", params={
        "format": "geojson", "minmagnitude": min_magnitude, "orderby": "time", "limit": 100,
        "starttime": (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat(),
    }).json()
    items = []
    for feature in data.get("features", []):
        props, coords = feature["properties"], feature["geometry"]["coordinates"]
        magnitude = props.get("mag") or 0
        item = {"category": "GEOINT", "title": f"M{magnitude:.1f} — {props.get('place', 'Unknown location')}",
                "summary": props.get("title", ""), "source": "USGS Earthquake Hazards Program",
                "source_url": props.get("url", ""), "longitude": coords[0], "latitude": coords[1],
                "severity": "high" if magnitude >= 6 else "medium" if magnitude >= 5 else "info",
                "confidence": 0.95, "tags": ["earthquake", "public-feed"], "raw": feature}
        add_item(item); items.append(item)
    return items


def weather(location: str) -> list[dict]:
    geo = _get("https://geocoding-api.open-meteo.com/v1/search", params={"name": location, "count": 1}).json()
    results = geo.get("results") or []
    if not results:
        raise ValueError(f"No location found for {location!r}")
    place = results[0]
    data = _get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": place["latitude"], "longitude": place["longitude"], "current": "temperature_2m,wind_speed_10m,weather_code",
    }).json()["current"]
    item = {"category": "GEOINT", "title": f"Weather — {place['name']}",
            "summary": f"{data['temperature_2m']}°C; wind {data['wind_speed_10m']} km/h; weather code {data['weather_code']}.",
            "source": "Open-Meteo", "source_url": "https://open-meteo.com/", "location": place["name"],
            "latitude": place["latitude"], "longitude": place["longitude"], "confidence": 0.9,
            "tags": ["weather", "public-feed"], "raw": {"place": place, "weather": data}}
    add_item(item)
    return [item]


def news(query: str, limit: int = 20) -> list[dict]:
    """Collect RSS results from Google News; no account or scraping required."""
    import xml.etree.ElementTree as ET
    feed = _get(f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en-US&gl=US&ceid=US:en").content
    root = ET.fromstring(feed)
    items = []
    for entry in root.findall("./channel/item")[:max(1, min(limit, 50))]:
        title = entry.findtext("title", "Untitled")
        link = entry.findtext("link", "")
        source = entry.findtext("source", "Google News")
        item = {"category": "OSINT", "title": title, "summary": entry.findtext("description", ""),
                "source": source, "source_url": link, "confidence": 0.65,
                "tags": ["news", query], "raw": {"published": entry.findtext("pubDate", "")}}
        add_item(item); items.append(item)
    return items


COLLECTORS = {"earthquakes": earthquakes, "weather": weather, "news": news}
