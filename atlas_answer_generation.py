"""Template-based answer generation for the Atlas demo pipeline."""

from __future__ import annotations

import random


MOVIE_INTENTS = {
    "MovieOverview",
    "MovieRating",
    "MovieDirector",
    "MovieCast",
    "SimilarMovies",
    "DiscoverByGenre",
}

CONTROL_INTENTS = {
    "LightOn",
    "LightOff",
    "SetBrightness",
    "OpenBlinds",
    "CloseBlinds",
    "SetTemperature",
    "SetScene",
}


def _pick(options: list[str]) -> str:
    return random.choice(options)


def _format_top_items(items: list[str], max_items: int = 3) -> str:
    if not items:
        return ""
    return ", ".join(items[:max_items])


def generate_answer(intent: str, slots: dict, api_result: dict, control_state: dict | None = None) -> str:
    """Generate a user-facing answer from fulfillment output."""

    intent = intent or api_result.get("intent", "")
    slots = slots or {}
    api_result = api_result or {}
    control_state = control_state or {}

    if not api_result:
        return "No fulfilled response is available yet."

    if api_result.get("status") == "error":
        error_message = api_result.get("message", "I could not complete that request.")
        return _pick(
            [
                f"Sorry, I could not complete that request. {error_message}",
                f"I ran into a problem while handling that request. {error_message}",
                f"That did not work as expected. {error_message}",
            ]
        )

    if intent == "Greetings":
        user_name = slots.get("USER") or control_state.get("last_user") or "there"
        return _pick(
            [
                f"Hello {user_name}. Atlas is ready.",
                f"Hi {user_name}. I am ready when you are.",
                f"Welcome back {user_name}. Atlas is online.",
            ]
        )

    if intent == "Goodbye":
        return _pick(
            [
                "Goodbye. Atlas is going back to sleep.",
                "Talk to you later. Atlas is standing by.",
                "See you later. Atlas is wrapping up this session.",
            ]
        )

    if intent == "OOS":
        return _pick(
            [
                "That request is outside Atlas's current scope. I can help with weather, movies, timers, and smart dorm controls.",
                "I cannot do that yet. Atlas currently supports weather, movie information, timers, and smart dorm controls.",
                "That request is not available in this demo. Try weather, movie, timer, or dorm commands instead.",
            ]
        )

    if intent == "SetTimer":
        timer = api_result.get("timer", {})
        timer_name = timer.get("name", slots.get("TIMER_NAME", "general"))
        duration_text = timer.get("duration_text", slots.get("DURATION", "that duration"))
        return _pick(
            [
                f"I set the {timer_name} timer for {duration_text}.",
                f"The {timer_name} timer is now running for {duration_text}.",
                f"Done. Your {timer_name} timer is set for {duration_text}.",
            ]
        )

    if intent == "GetWeather":
        city = api_result.get("city", slots.get("CITY", "that city"))
        date_label = api_result.get("date_label", slots.get("DATE", "today"))
        raw = api_result.get("raw", {})
        current = raw.get("current", {})
        daily = raw.get("daily", {})
        if date_label.lower() in {"today", "tonight"} and current:
            temperature = current.get("temperature_2m")
            return _pick(
                [
                    f"In {city}, it is currently {temperature} degrees Celsius. {api_result.get('message', '').strip()}",
                    f"Here is the latest weather for {city}: {api_result.get('message', '').strip()}",
                    f"The current weather in {city} is ready. {api_result.get('message', '').strip()}",
                ]
            )

        temps_min = daily.get("temperature_2m_min", [])
        temps_max = daily.get("temperature_2m_max", [])
        rain_probs = daily.get("precipitation_probability_max", [])
        extra = ""
        if temps_min and temps_max and rain_probs:
            extra = (
                f" Temperatures look to range between {temps_min[0]} and {temps_max[0]} degrees Celsius,"
                f" with about {rain_probs[0]} percent chance of rain."
            )
        return _pick(
            [
                f"For {date_label} in {city}, {api_result.get('message', '').strip()}",
                f"Here is the forecast for {date_label} in {city}. {api_result.get('message', '').strip()}",
                f"I checked the weather for {date_label} in {city}.{extra}".strip(),
            ]
        )

    if intent in MOVIE_INTENTS:
        details = api_result.get("details", {})
        title = api_result.get("title", details.get("title", slots.get("TITLE", "that movie")))

        if intent == "MovieOverview":
            overview = details.get("overview", api_result.get("message", ""))
            return _pick(
                [
                    f"Here is a quick overview of {title}: {overview}",
                    f"{title} is summarized like this: {overview}",
                    f"For {title}, the overview is: {overview}",
                ]
            )

        if intent == "MovieRating":
            rating = details.get("rating")
            return _pick(
                [
                    f"{title} currently has a rating of {rating} out of 10.",
                    f"The rating I found for {title} is {rating} out of 10.",
                    f"{title} is rated {rating} out of 10.",
                ]
            )

        if intent == "MovieDirector":
            director = details.get("director", "Unknown")
            return _pick(
                [
                    f"{title} was directed by {director}.",
                    f"The director of {title} is {director}.",
                    f"I found that {director} directed {title}.",
                ]
            )

        if intent == "MovieCast":
            cast = _format_top_items(details.get("cast", []))
            return _pick(
                [
                    f"The main cast of {title} includes {cast}.",
                    f"For {title}, the top cast members are {cast}.",
                    f"{title} features actors such as {cast}.",
                ]
            )

        if intent == "SimilarMovies":
            similar = _format_top_items(details.get("similar", api_result.get("results", [])))
            return _pick(
                [
                    f"If you liked {title}, you could try {similar}.",
                    f"Movies similar to {title} include {similar}.",
                    f"Based on {title}, I would suggest {similar}.",
                ]
            )

        if intent == "DiscoverByGenre":
            genre = api_result.get("genre", slots.get("GENRE", "that genre"))
            year = api_result.get("year", slots.get("YEAR"))
            results = _format_top_items(api_result.get("results", []), max_items=5)
            if year:
                return _pick(
                    [
                        f"Here are some {genre} movies from {year}: {results}.",
                        f"For {genre} movies released in {year}, I found {results}.",
                        f"Some {genre} titles from {year} are {results}.",
                    ]
                )
            return _pick(
                [
                    f"Here are some {genre} movies: {results}.",
                    f"I found these {genre} titles: {results}.",
                    f"Some {genre} movies you can try are {results}.",
                ]
            )

    if intent in CONTROL_INTENTS:
        room = control_state.get("room", slots.get("ROOM", "room"))
        light = control_state.get("light", "off")
        brightness = control_state.get("brightness", 0)
        blinds = control_state.get("blinds", "open")
        temperature = control_state.get("temperature_c", 20)
        scene = control_state.get("scene", "relax")

        templates = {
            "LightOn": [
                f"The {room} light is on now.",
                f"I switched on the {room} light.",
                f"The light in the {room} is now on.",
            ],
            "LightOff": [
                f"The {room} light is off now.",
                f"I switched off the {room} light.",
                f"The light in the {room} is now off.",
            ],
            "SetBrightness": [
                f"The {room} light brightness is now {brightness} percent.",
                f"I set the {room} brightness to {brightness} percent.",
                f"The light level in the {room} is now {brightness} percent.",
            ],
            "OpenBlinds": [
                f"The {room} blinds are open now.",
                f"I opened the {room} blinds.",
                f"The blinds in the {room} are now open.",
            ],
            "CloseBlinds": [
                f"The {room} blinds are closed now.",
                f"I closed the {room} blinds.",
                f"The blinds in the {room} are now closed.",
            ],
            "SetTemperature": [
                f"The room temperature is now {temperature} degrees Celsius.",
                f"I set the temperature to {temperature} degrees Celsius.",
                f"The dorm temperature is now {temperature} degrees Celsius.",
            ],
            "SetScene": [
                f"The room is now in {scene} mode.",
                f"I switched the dorm to {scene} mode.",
                f"The current room scene is now {scene}.",
            ],
        }
        return _pick(templates.get(intent, [api_result.get("message", "Done.")]))

    message = api_result.get("message", "").strip()
    if message:
        return message

    return "Done."
