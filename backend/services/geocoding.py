"""Forward / reverse geocoding service.

Forward search supports three providers selected per-request:

* ``nominatim`` — default, the OSM-backed public Nominatim instance. Free,
  no key, but Asian street-level accuracy is weak.
* ``photon`` — komoot-hosted, also OSM-backed, free, no key. Better fuzzy
  matching and typo tolerance than Nominatim, looser rate limits in
  practice. Returns GeoJSON instead of Nominatim's flat shape.
* ``google`` — Google Geocoding API. Requires the user's own API key;
  10k free events / month with the Essentials tier.

Reverse geocoding uses Photon. It feeds the country flag + short name in
the status bar and the bookmark flags, so it has to work for every user;
the public Nominatim instance answers this app's User-Agent with HTTP 403.
For the same reason a ``nominatim`` forward search that gets refused is
served by Photon for the rest of the session.

Photon is a free community service too, so reverse results are cached on
disk per ~100 m cell: teleporting back to a saved spot, bouncing between
two points, or the status bar and the recent-places list asking about the
same coordinate never costs a second request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx
from fastapi import HTTPException

from config import (
    NOMINATIM_BASE_URL,
    NOMINATIM_USER_AGENT,
    PHOTON_BASE_URL,
    REVERSE_GEOCODE_CACHE_FILE,
)
from models.schemas import GeocodingResult

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_GOOGLE_GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# Reverse cache: 3 decimals is a ~110 m cell, fine for a flag + place label.
_REVERSE_CACHE_DECIMALS = 3
_REVERSE_CACHE_MAX = 1000
_REVERSE_CACHE_TTL_S = 30 * 24 * 3600


class GeocodingService:
    """Async wrapper around forward / reverse geocoding."""

    # Set once Nominatim refuses us (403 / 429) so later searches go
    # straight to Photon instead of re-hitting a server that said no.
    _nominatim_refused = False

    def _headers(self) -> dict[str, str]:
        return {
            "User-Agent": NOMINATIM_USER_AGENT,
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # Forward geocoding — dispatcher
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 5,
        provider: str = "nominatim",
        google_key: str | None = None,
    ) -> list[GeocodingResult]:
        """Forward geocode: address or place name -> coordinates."""
        if provider == "google":
            if not google_key:
                raise HTTPException(
                    status_code=400,
                    detail="provider=google requires google_key",
                )
            return await self._search_google(query, limit, google_key)
        if provider == "photon" or GeocodingService._nominatim_refused:
            return await self._search_photon(query, limit)
        try:
            return await self._search_nominatim(query, limit)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in (403, 429):
                raise
            logger.warning(
                "Nominatim refused search (HTTP %d), using Photon for this session",
                exc.response.status_code,
            )
            GeocodingService._nominatim_refused = True
            return await self._search_photon(query, limit)

    async def _search_nominatim(self, query: str, limit: int) -> list[GeocodingResult]:
        params = {
            "q": query,
            "format": "json",
            "limit": min(limit, 40),
        }
        logger.debug("Nominatim search: %s", query)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{NOMINATIM_BASE_URL}/search",
                params=params,
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[GeocodingResult] = []
        for item in data:
            try:
                results.append(
                    GeocodingResult(
                        display_name=item.get("display_name", ""),
                        lat=float(item["lat"]),
                        lng=float(item["lon"]),
                        type=item.get("type", ""),
                        importance=float(item.get("importance", 0)),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed search result: %s", exc)
        return results

    async def _search_photon(self, query: str, limit: int) -> list[GeocodingResult]:
        # Photon returns GeoJSON: a FeatureCollection where each feature has
        # `geometry.coordinates = [lon, lat]` and `properties` with name /
        # city / country / etc. There's no `display_name` field, so we
        # synthesise one from the properties for parity with Nominatim.
        params = {"q": query, "limit": min(limit, 40)}
        logger.debug("Photon search: %s", query)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{PHOTON_BASE_URL}/api",
                params=params,
                headers={"User-Agent": NOMINATIM_USER_AGENT},
            )
            resp.raise_for_status()
            data = resp.json()

        results: list[GeocodingResult] = []
        for feat in data.get("features", []):
            r = _photon_feature_to_result(feat, query)
            if r:
                results.append(r)
        return results

    async def _search_google(
        self, query: str, limit: int, api_key: str
    ) -> list[GeocodingResult]:
        # Google's Geocoding API doesn't take a `limit` — it returns a
        # capped list (usually 1, sometimes more for ambiguous queries).
        # We slice client-side after the fact so behaviour matches the
        # Nominatim path.
        params = {
            "address": query,
            "key": api_key,
            "language": "zh-TW",
        }
        logger.debug("Google geocode search: %s", query)
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(_GOOGLE_GEOCODE_URL, params=params)
        if resp.status_code != 200:
            text = resp.text[:200] if resp.text else ""
            raise HTTPException(
                status_code=502,
                detail=f"Google geocode HTTP {resp.status_code}: {text}",
            )
        data = resp.json()
        status = data.get("status")
        if status not in ("OK", "ZERO_RESULTS"):
            # Surface Google's own error text so the user can fix things
            # like REQUEST_DENIED (key invalid / API not enabled) and
            # OVER_QUERY_LIMIT (free tier exhausted).
            err_msg = data.get("error_message") or status or "unknown error"
            raise HTTPException(
                status_code=502,
                detail=f"Google geocode {status}: {err_msg}",
            )

        results: list[GeocodingResult] = []
        for item in (data.get("results") or [])[:limit]:
            try:
                loc = item["geometry"]["location"]
                # Google's `types` is a list like ["street_address"];
                # take the first as our `type` field for compat with the
                # Nominatim shape.
                types = item.get("types") or []
                results.append(
                    GeocodingResult(
                        display_name=item.get("formatted_address", ""),
                        lat=float(loc["lat"]),
                        lng=float(loc["lng"]),
                        type=types[0] if types else "",
                        importance=0.0,  # Google doesn't expose this
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed Google result: %s", exc)
        return results

    # ------------------------------------------------------------------
    # Reverse geocoding
    # ------------------------------------------------------------------

    async def reverse(
        self, lat: float, lng: float, precise: bool = False
    ) -> GeocodingResult | None:
        """Reverse geocode: coordinates -> address.

        Returns ``None`` when no result is found. ``precise`` skips the
        cache read for callers that name a saved spot or answer "what's
        here", where a neighbour's label from the same cell would be wrong.
        """
        key = f"{lat:.{_REVERSE_CACHE_DECIMALS}f},{lng:.{_REVERSE_CACHE_DECIMALS}f}"
        if not precise:
            hit = _reverse_cache_get(key)
            if hit is not _MISS:
                return hit
            # Same cell already being fetched (status bar + recent-places
            # fire together on every teleport): share that request.
            pending = _reverse_inflight.get(key)
            if pending is not None:
                return await asyncio.shield(pending)

        task = asyncio.ensure_future(self._reverse_photon(lat, lng))
        if not precise:
            _reverse_inflight[key] = task
        try:
            result = await asyncio.shield(task)
        finally:
            if _reverse_inflight.get(key) is task:
                del _reverse_inflight[key]
        _reverse_cache_put(key, result)
        return result

    async def _reverse_photon(self, lat: float, lng: float) -> GeocodingResult | None:
        logger.debug("Photon reverse: %.6f, %.6f", lat, lng)

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(
                f"{PHOTON_BASE_URL}/reverse",
                params={"lat": lat, "lon": lng},
                headers={"User-Agent": NOMINATIM_USER_AGENT},
            )
            resp.raise_for_status()
            data = resp.json()

        for feat in data.get("features", []):
            r = _photon_feature_to_result(feat, "")
            if r:
                return r
        return None


# key -> [saved_at, result dict | None]. Insertion-ordered, oldest first.
_MISS = object()
_reverse_cache: dict[str, list] | None = None
_reverse_inflight: dict[str, asyncio.Future] = {}


def _reverse_cache_load() -> dict[str, list]:
    global _reverse_cache
    if _reverse_cache is None:
        try:
            raw = json.loads(REVERSE_GEOCODE_CACHE_FILE.read_text(encoding="utf-8"))
            _reverse_cache = raw if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            _reverse_cache = {}
    return _reverse_cache


def _reverse_cache_get(key: str):
    entry = _reverse_cache_load().get(key)
    if not entry:
        return _MISS
    try:
        saved_at, payload = entry
        if time.time() - float(saved_at) > _REVERSE_CACHE_TTL_S:
            return _MISS
        return GeocodingResult(**payload) if payload else None
    except (TypeError, ValueError):
        return _MISS


def _reverse_cache_put(key: str, result: GeocodingResult | None) -> None:
    cache = _reverse_cache_load()
    cache.pop(key, None)
    cache[key] = [time.time(), result.model_dump() if result else None]
    while len(cache) > _REVERSE_CACHE_MAX:
        del cache[next(iter(cache))]
    try:
        REVERSE_GEOCODE_CACHE_FILE.write_text(
            json.dumps(cache, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        logger.info("Could not persist reverse geocode cache: %s", exc)


def _photon_feature_to_result(feat: dict, fallback_name: str) -> GeocodingResult | None:
    """Convert one Photon GeoJSON feature into a GeocodingResult."""
    try:
        coords = feat["geometry"]["coordinates"]
        lng, lat = float(coords[0]), float(coords[1])
        props = feat.get("properties") or {}
        name = (props.get("name") or "").strip()
        # Build a Nominatim-style "specific, generic, country" line.
        parts: list[str] = []
        if name:
            parts.append(name)
        house = props.get("housenumber")
        street = props.get("street")
        if house and street:
            parts.append(f"{street} {house}")
        elif street:
            parts.append(street)
        for key in ("district", "city", "county", "state", "country"):
            v = props.get(key)
            if v and v not in parts:
                parts.append(v)
        display = ", ".join(parts) if parts else (name or fallback_name)
        # Unnamed features (plain addresses) still need a label for the
        # status bar / bookmark name: street first, then the area.
        short = name
        if len(short) < 2:
            for key in ("street", "locality", "district", "city", "county", "state"):
                v = props.get(key)
                if v and len(str(v).strip()) > 1:
                    short = str(v).strip()
                    break
        return GeocodingResult(
            display_name=display,
            lat=lat,
            lng=lng,
            # Photon's "type" is the OSM tag value (e.g. "city"),
            # mirror it into our `type` field for compat.
            type=props.get("type") or props.get("osm_value") or "",
            importance=0.0,
            country_code=(props.get("countrycode") or "").lower(),
            short_name=short,
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("Skipping malformed Photon result: %s", exc)
        return None
