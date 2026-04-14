"""Fulfillment logic for the Atlas virtual assistant demo."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from datetime import datetime

import requests


TMDB_API_KEY = os.getenv("TMDB_API_KEY", "").strip()
TMDB_BEARER_TOKEN = os.getenv("TMDB_BEARER_TOKEN", "").strip()

OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

DEFAULT_CONTROL_STATE = {
    "room": "theater",
    "light": "off",
    "brightness": 50,
    "blinds": "open",
    "temperature_c": 20,
    "scene": "relax",
    "timers": [],
}

SCENE_PRESETS = {
    "study": {
        "light": "on",
        "brightness": 80,
        "blinds": "open",
        "temperature_c": 21,
        "scene": "study",
    },
    "relax": {
        "light": "on",
        "brightness": 35,
        "blinds": "closed",
        "temperature_c": 22,
        "scene": "relax",
    },
    "sleep": {
        "light": "off",
        "brightness": 0,
        "blinds": "closed",
        "temperature_c": 19,
        "scene": "sleep",
    },
    "movie": {
        "light": "off",
        "brightness": 15,
        "blinds": "closed",
        "temperature_c": 20,
        "scene": "movie",
    },
}

WEATHER_CODE_MAP = {
    0: "clear skies",
    1: "mainly clear skies",
    2: "partly cloudy skies",
    3: "overcast skies",
    45: "foggy conditions",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "light rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "a thunderstorm",
}

MOVIE_FALLBACK_DB = {
    "interstellar": {
        "title": "Interstellar",
        "overview": "A team of astronauts travels through a wormhole to search for a new home for humanity as Earth becomes uninhabitable.",
        "rating": 8.7,
        "director": "Christopher Nolan",
        "cast": ["Matthew McConaughey", "Anne Hathaway", "Jessica Chastain"],
        "similar": ["The Martian", "Gravity", "Ad Astra"],
        "genres": ["science fiction", "drama"],
        "year": 2014,
    },
    "dune part two": {
        "title": "Dune: Part Two",
        "overview": "Paul Atreides unites with the Fremen while confronting war, prophecy, and the fate of Arrakis.",
        "rating": 8.5,
        "director": "Denis Villeneuve",
        "cast": ["Timothee Chalamet", "Zendaya", "Rebecca Ferguson"],
        "similar": ["Dune", "Blade Runner 2049", "Arrival"],
        "genres": ["science fiction", "adventure"],
        "year": 2024,
    },
    "the matrix": {
        "title": "The Matrix",
        "overview": "A hacker discovers that reality is a simulation and joins a rebellion against the machines controlling humanity.",
        "rating": 8.7,
        "director": "The Wachowskis",
        "cast": ["Keanu Reeves", "Carrie-Anne Moss", "Laurence Fishburne"],
        "similar": ["Dark City", "Equilibrium", "Inception"],
        "genres": ["science fiction", "action"],
        "year": 1999,
    },
    "arrival": {
        "title": "Arrival",
        "overview": "A linguist works with the military to communicate with extraterrestrial visitors whose language may alter humanity's future.",
        "rating": 7.9,
        "director": "Denis Villeneuve",
        "cast": ["Amy Adams", "Jeremy Renner", "Forest Whitaker"],
        "similar": ["Contact", "Interstellar", "Annihilation"],
        "genres": ["science fiction", "drama"],
        "year": 2016,
    },
    "blade runner 2049": {
        "title": "Blade Runner 2049",
        "overview": "A new blade runner uncovers a secret that could destabilize what remains of society and sends him searching for Rick Deckard.",
        "rating": 8.0,
        "director": "Denis Villeneuve",
        "cast": ["Ryan Gosling", "Harrison Ford", "Ana de Armas"],
        "similar": ["Blade Runner", "Dune", "Ex Machina"],
        "genres": ["science fiction", "thriller"],
        "year": 2017,
    },
    "inception": {
        "title": "Inception",
        "overview": "A skilled thief enters dreams to steal secrets and is offered a final job that requires planting an idea instead.",
        "rating": 8.8,
        "director": "Christopher Nolan",
        "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
        "similar": ["Shutter Island", "Tenet", "The Matrix"],
        "genres": ["science fiction", "action"],
        "year": 2010,
    },
}


class FulfillmentError(Exception):
    pass


def ensure_control_state(state: dict | None):
    if state is None:
        return deepcopy(DEFAULT_CONTROL_STATE)

    merged = deepcopy(DEFAULT_CONTROL_STATE)
    merged.update(state)
    if "timers" not in merged or not isinstance(merged["timers"], list):
        merged["timers"] = []
    return merged


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def parse_int_value(text: str, fallback: int | None = None):
    if text is None:
        return fallback
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else fallback


def parse_timer_duration(duration_text: str):
    if not duration_text:
        raise FulfillmentError("Timer duration is missing.")

    match = re.search(r"(?P<value>\d+)\s*(?P<unit>second|seconds|minute|minutes|hour|hours)", duration_text.lower())
    if not match:
        raise FulfillmentError(f"Could not parse timer duration from '{duration_text}'.")

    value = int(match.group("value"))
    unit = match.group("unit")
    if unit.startswith("second"):
        total_seconds = value
    elif unit.startswith("minute"):
        total_seconds = value * 60
    else:
        total_seconds = value * 3600

    return {
        "value": value,
        "unit": unit,
        "total_seconds": total_seconds,
    }


def geocode_city(city: str):
    response = requests.get(
        OPEN_METEO_GEOCODING_URL,
        params={"name": city, "count": 5, "language": "en", "format": "json"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    if not results:
        raise FulfillmentError(f"I could not find weather data for {city}.")
    return results[0]


def fetch_weather(city: str, date_label: str | None):
    geo = geocode_city(city)
    forecast_response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params={
            "latitude": geo["latitude"],
            "longitude": geo["longitude"],
            "timezone": "auto",
            "current": "temperature_2m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "forecast_days": 7,
        },
        timeout=10,
    )
    forecast_response.raise_for_status()
    forecast = forecast_response.json()

    date_label_normalized = (date_label or "today").lower()
    daily = forecast.get("daily", {})
    daily_dates = daily.get("time", [])

    if date_label_normalized in {"today", "tonight"}:
        current = forecast.get("current", {})
        weather_code = current.get("weather_code")
        description = WEATHER_CODE_MAP.get(weather_code, "current conditions")
        temperature = current.get("temperature_2m")
        message = f"In {geo['name']}, it is currently {temperature} degrees Celsius with {description}."
        if date_label_normalized == "tonight":
            message = f"For tonight in {geo['name']}, I expect {description} with a current temperature of {temperature} degrees Celsius."
        return {
            "status": "success",
            "intent": "GetWeather",
            "source": "open-meteo",
            "city": geo["name"],
            "date_label": date_label or "today",
            "message": message,
            "raw": forecast,
        }

    target_index = 1 if date_label_normalized == "tomorrow" and len(daily_dates) > 1 else 0
    if date_label_normalized not in {"tomorrow"} and len(daily_dates) > 0:
        # For relative weekday phrasing, use the closest available day while preserving the label in the message.
        target_index = min(1, len(daily_dates) - 1)

    description = WEATHER_CODE_MAP.get(daily.get("weather_code", [None])[target_index], "forecasted conditions")
    temp_min = daily.get("temperature_2m_min", [None])[target_index]
    temp_max = daily.get("temperature_2m_max", [None])[target_index]
    rain_prob = daily.get("precipitation_probability_max", [None])[target_index]
    message = (
        f"For {date_label or 'today'} in {geo['name']}, I expect {description}, "
        f"with temperatures between {temp_min} and {temp_max} degrees Celsius "
        f"and about {rain_prob} percent chance of rain."
    )
    return {
        "status": "success",
        "intent": "GetWeather",
        "source": "open-meteo",
        "city": geo["name"],
        "date_label": date_label or "today",
        "message": message,
        "raw": forecast,
    }


def tmdb_headers():
    if TMDB_BEARER_TOKEN:
        return {"Authorization": f"Bearer {TMDB_BEARER_TOKEN}"}
    return {}


def tmdb_get(path: str, params: dict | None = None):
    params = dict(params or {})
    headers = tmdb_headers()
    if not headers:
        if not TMDB_API_KEY:
            raise FulfillmentError("TMDB credentials are not configured.")
        params["api_key"] = TMDB_API_KEY

    response = requests.get(f"{TMDB_BASE_URL}{path}", params=params, headers=headers, timeout=10)
    response.raise_for_status()
    return response.json()


def lookup_fallback_movie(title: str):
    key = normalize_title(title)
    movie = MOVIE_FALLBACK_DB.get(key)
    if not movie:
        raise FulfillmentError(f"I do not have demo movie data for {title}.")
    return movie


def search_tmdb_movie(title: str):
    data = tmdb_get("/search/movie", {"query": title})
    results = data.get("results") or []
    if not results:
        raise FulfillmentError(f"I could not find a movie matching {title}.")
    return results[0]


def fetch_movie_details(title: str):
    if TMDB_API_KEY or TMDB_BEARER_TOKEN:
        movie = search_tmdb_movie(title)
        movie_id = movie["id"]
        details = tmdb_get(f"/movie/{movie_id}")
        credits = tmdb_get(f"/movie/{movie_id}/credits")
        director = next((person["name"] for person in credits.get("crew", []) if person.get("job") == "Director"), "Unknown")
        cast = [person["name"] for person in credits.get("cast", [])[:5]]
        similar = [item["title"] for item in tmdb_get(f"/movie/{movie_id}/similar").get("results", [])[:5]]
        return {
            "title": details.get("title", title),
            "overview": details.get("overview", "No overview available."),
            "rating": details.get("vote_average"),
            "director": director,
            "cast": cast,
            "similar": similar,
            "genres": [genre["name"] for genre in details.get("genres", [])],
            "year": int(details.get("release_date", "0000")[:4]) if details.get("release_date") else None,
            "source": "tmdb",
        }

    movie = lookup_fallback_movie(title)
    return {**movie, "source": "fallback"}


def discover_movies_by_genre(genre: str, year: str | None):
    if TMDB_API_KEY or TMDB_BEARER_TOKEN:
        genres = tmdb_get("/genre/movie/list").get("genres", [])
        genre_id = next((item["id"] for item in genres if item["name"].lower() == genre.lower()), None)
        if genre_id is None:
            raise FulfillmentError(f"I could not map the genre {genre} to TMDB.")

        params = {"with_genres": genre_id, "sort_by": "popularity.desc"}
        if year:
            params["primary_release_year"] = year

        results = tmdb_get("/discover/movie", params).get("results", [])[:5]
        if not results:
            raise FulfillmentError(f"I could not find {genre} movies for that request.")

        titles = [item["title"] for item in results]
        message = f"Here are some {genre} movies"
        if year:
            message += f" from {year}"
        message += f": {', '.join(titles)}."
        return {
            "status": "success",
            "intent": "DiscoverByGenre",
            "source": "tmdb",
            "genre": genre,
            "year": year,
            "results": titles,
            "message": message,
        }

    matches = []
    for movie in MOVIE_FALLBACK_DB.values():
        genres = [g.lower() for g in movie["genres"]]
        if genre.lower() in genres and (year is None or str(movie["year"]) == str(year)):
            matches.append(movie["title"])

    if not matches:
        raise FulfillmentError(f"I do not have fallback demo results for {genre} movies.")

    message = f"Here are some {genre} movies"
    if year:
        message += f" from {year}"
    message += f": {', '.join(matches[:5])}."
    return {
        "status": "success",
        "intent": "DiscoverByGenre",
        "source": "fallback",
        "genre": genre,
        "year": year,
        "results": matches[:5],
        "message": message,
    }


def fulfill_movie_intent(intent: str, slots: dict):
    title = slots.get("TITLE")
    genre = slots.get("GENRE")
    year = slots.get("YEAR")

    if intent == "DiscoverByGenre":
        if not genre:
            raise FulfillmentError("Genre is required for movie discovery.")
        return discover_movies_by_genre(genre, year)

    if not title:
        raise FulfillmentError("Movie title is required for this intent.")

    details = fetch_movie_details(title)
    movie_title = details["title"]

    if intent == "MovieOverview":
        message = f"{movie_title}: {details['overview']}"
    elif intent == "MovieRating":
        message = f"{movie_title} has a rating of {details['rating']} out of 10."
    elif intent == "MovieDirector":
        message = f"{movie_title} was directed by {details['director']}."
    elif intent == "MovieCast":
        message = f"The main cast of {movie_title} includes {', '.join(details['cast'][:3])}."
    elif intent == "SimilarMovies":
        message = f"If you liked {movie_title}, you could also watch {', '.join(details['similar'][:3])}."
    else:
        raise FulfillmentError(f"Unsupported movie intent: {intent}")

    return {
        "status": "success",
        "intent": intent,
        "source": details["source"],
        "title": movie_title,
        "message": message,
        "details": details,
    }


def fulfill_control_intent(intent: str, slots: dict, control_state: dict):
    room = slots.get("ROOM", control_state.get("room", "theater"))
    control_state["room"] = room

    if intent == "LightOn":
        control_state["light"] = "on"
        message = f"The {room} light is now on."
    elif intent == "LightOff":
        control_state["light"] = "off"
        message = f"The {room} light is now off."
    elif intent == "SetBrightness":
        brightness = parse_int_value(slots.get("BRIGHTNESS"), fallback=None)
        if brightness is None:
            raise FulfillmentError("Brightness value is missing.")
        brightness = max(0, min(100, brightness))
        control_state["brightness"] = brightness
        control_state["light"] = "on" if brightness > 0 else "off"
        message = f"The {room} light brightness is set to {brightness} percent."
    elif intent == "OpenBlinds":
        control_state["blinds"] = "open"
        message = f"The {room} blinds are now open."
    elif intent == "CloseBlinds":
        control_state["blinds"] = "closed"
        message = f"The {room} blinds are now closed."
    elif intent == "SetTemperature":
        temperature = parse_int_value(slots.get("TEMPERATURE"), fallback=None)
        if temperature is None:
            raise FulfillmentError("Temperature value is missing.")
        control_state["temperature_c"] = temperature
        message = f"The home theater temperature is now set to {temperature} degrees Celsius."
    elif intent == "SetScene":
        scene = (slots.get("SCENE") or "").lower()
        if scene not in SCENE_PRESETS:
            raise FulfillmentError(f"Unsupported scene '{scene}'.")
        control_state.update(SCENE_PRESETS[scene])
        message = f"Atlas switched the home theater to {scene} mode."
    else:
        raise FulfillmentError(f"Unsupported control intent: {intent}")

    return {
        "status": "success",
        "intent": intent,
        "source": "simulated_home_theater",
        "message": message,
        "control_state": deepcopy(control_state),
    }


def fulfill_intent(intent: str, slots: dict, control_state: dict | None = None):
    slots = slots or {}
    control_state = ensure_control_state(control_state)

    if intent == "Greetings":
        return {"status": "success", "intent": intent, "source": "canned", "message": "Hello. Atlas is ready."}, control_state

    if intent == "Goodbye":
        return {"status": "success", "intent": intent, "source": "canned", "message": "Goodbye. Atlas is going back to sleep soon."}, control_state

    if intent == "OOS":
        return {
            "status": "success",
            "intent": intent,
            "source": "canned",
            "message": "That request is outside Atlas's current demo scope. I can help with weather, movies, timers, and home theater controls.",
        }, control_state

    if intent == "SetTimer":
        duration_info = parse_timer_duration(slots.get("DURATION"))
        timer_name = slots.get("TIMER_NAME", "general")
        timer_record = {
            "name": timer_name,
            "duration_text": slots.get("DURATION"),
            "total_seconds": duration_info["total_seconds"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        control_state["timers"].append(timer_record)
        return {
            "status": "success",
            "intent": intent,
            "source": "local_timer",
            "timer": timer_record,
            "message": f"I set the {timer_name} timer for {slots.get('DURATION')}.",
        }, control_state

    if intent == "GetWeather":
        city = slots.get("CITY")
        if not city:
            raise FulfillmentError("City is required for weather requests.")
        return fetch_weather(city, slots.get("DATE")), control_state

    if intent in {"MovieOverview", "MovieRating", "MovieDirector", "MovieCast", "SimilarMovies", "DiscoverByGenre"}:
        return fulfill_movie_intent(intent, slots), control_state

    if intent in {"LightOn", "LightOff", "SetBrightness", "OpenBlinds", "CloseBlinds", "SetTemperature", "SetScene"}:
        api_result = fulfill_control_intent(intent, slots, control_state)
        return api_result, control_state

    raise FulfillmentError(f"Unsupported intent: {intent}")
