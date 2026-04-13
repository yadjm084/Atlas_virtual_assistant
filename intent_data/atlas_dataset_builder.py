"""Build a training-ready Atlas intent dataset for the April 14 demo.

The output format is JSONL with one record per example:
{"intent": "...", "plain_text": "...", "annotated": "..."}
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

from intent_data.atlas_demo_intents import INTENT_EXAMPLES, INTENT_SCHEMA, get_all_intents


TARGET_EXAMPLES_PER_INTENT = 24
PLACEHOLDER_RE = re.compile(r"{([A-Z_]+)}")


SLOT_VALUES = {
    "TITLE": [
        "Interstellar",
        "Dune Part Two",
        "The Matrix",
        "Arrival",
        "Blade Runner 2049",
        "Inception",
    ],
    "GENRE": [
        "science fiction",
        "action",
        "comedy",
        "horror",
        "drama",
        "animation",
    ],
    "YEAR": [
        "2022",
        "2023",
        "2024",
    ],
    "CITY": [
        "Ottawa",
        "Montreal",
        "Toronto",
        "Paris",
        "New York",
        "Vancouver",
    ],
    "DATE": [
        "today",
        "tomorrow",
        "tonight",
        "next Monday",
        "this evening",
        "on Friday",
    ],
    "DURATION": [
        "5 minutes",
        "10 minutes",
        "15 minutes",
        "25 minutes",
        "45 seconds",
        "2 hours",
    ],
    "TIMER_NAME": [
        "study",
        "pasta",
        "laundry",
        "break",
        "focus",
        "soup",
    ],
    "ROOM": [
        "bedroom",
        "desk area",
        "study area",
        "main room",
        "reading corner",
        "window side",
    ],
    "BRIGHTNESS": [
        "20 percent",
        "30 percent",
        "40 percent",
        "60 percent",
        "75 percent",
        "90 percent",
    ],
    "TEMPERATURE": [
        "19 degrees",
        "20 degrees",
        "21 degrees",
        "22 degrees",
        "23 celsius",
        "24 celsius",
    ],
    "SCENE": [
        "study",
        "relax",
        "sleep",
        "movie",
    ],
}


STATIC_INTENT_EXAMPLES = {
    "Greetings": [
        "hello",
        "hello atlas",
        "hi",
        "hi atlas",
        "hey atlas",
        "good morning atlas",
        "good evening atlas",
        "good afternoon atlas",
        "hey there atlas",
        "hello there",
        "hi there",
        "atlas are you there",
        "wake up atlas hello",
        "greetings atlas",
        "good day atlas",
        "hey assistant",
        "hello assistant",
        "nice to see you atlas",
        "howdy atlas",
        "good morning there",
        "yo atlas",
        "hi again atlas",
        "hello again atlas",
        "hey good morning",
    ],
    "Goodbye": [
        "goodbye",
        "goodbye atlas",
        "bye atlas",
        "bye",
        "see you later atlas",
        "talk to you later atlas",
        "good night atlas",
        "thanks goodbye",
        "bye for now",
        "see you atlas",
        "catch you later atlas",
        "farewell atlas",
        "see you soon",
        "talk later",
        "thanks atlas goodbye",
        "that is all goodbye",
        "i am done goodbye",
        "bye bye atlas",
        "good night see you later",
        "catch you later",
        "thanks for the help goodbye",
        "later atlas",
        "i will come back later",
        "signing off atlas",
    ],
    "OOS": [
        "book a flight to Toronto",
        "write me a poem about the ocean",
        "translate hello into Japanese",
        "do my grocery shopping",
        "solve this calculus problem",
        "open my email inbox",
        "send a text to my mom",
        "make me a sandwich",
        "what is the capital of Peru",
        "call an uber for me",
        "summarize this PDF",
        "generate a rap song",
        "create a budget spreadsheet",
        "play my spotify playlist",
        "order dinner for tonight",
        "open youtube",
        "read my unread messages",
        "help me with my taxes",
        "book a dentist appointment",
        "find me cheap flights to Paris",
        "show my calendar for tomorrow",
        "compose an email to my professor",
        "write code for a binary tree",
        "tell me a bedtime story",
    ],
}


TEMPLATES = {
    "SetTimer": [
        "set a timer for {DURATION}",
        "start a timer for {DURATION}",
        "set a {TIMER_NAME} timer for {DURATION}",
        "start a {TIMER_NAME} timer for {DURATION}",
        "please set a timer for {DURATION}",
        "begin a {TIMER_NAME} timer for {DURATION}",
        "i need a {TIMER_NAME} timer for {DURATION}",
        "create a timer for {DURATION}",
    ],
    "GetWeather": [
        "what is the weather in {CITY}",
        "what is the weather in {CITY} {DATE}",
        "does it rain in {CITY}",
        "does it rain in {CITY} {DATE}",
        "is it cold in {CITY}",
        "is it cold in {CITY} {DATE}",
        "give me the forecast for {CITY}",
        "give me the forecast for {CITY} {DATE}",
    ],
    "MovieOverview": [
        "give me an overview of {TITLE}",
        "what is {TITLE} about",
        "tell me the plot of {TITLE}",
        "summarize {TITLE}",
        "what happens in {TITLE}",
        "can you describe {TITLE}",
    ],
    "MovieRating": [
        "what is the rating for {TITLE}",
        "how highly rated is {TITLE}",
        "show me the rating of {TITLE}",
        "what score does {TITLE} have",
        "is {TITLE} well rated",
        "tell me the score for {TITLE}",
    ],
    "MovieDirector": [
        "who directed {TITLE}",
        "who is the director of {TITLE}",
        "tell me who directed {TITLE}",
        "who made {TITLE}",
        "who was behind {TITLE}",
        "which director worked on {TITLE}",
    ],
    "MovieCast": [
        "who is in the cast of {TITLE}",
        "show me the cast for {TITLE}",
        "who stars in {TITLE}",
        "who acted in {TITLE}",
        "which actors are in {TITLE}",
        "tell me the main cast of {TITLE}",
    ],
    "SimilarMovies": [
        "find movies like {TITLE}",
        "recommend movies similar to {TITLE}",
        "what should i watch if i liked {TITLE}",
        "show movies like {TITLE}",
        "give me recommendations based on {TITLE}",
        "what movies are similar to {TITLE}",
    ],
    "DiscoverByGenre": [
        "discover {GENRE} movies",
        "discover {GENRE} movies from {YEAR}",
        "find {GENRE} movies",
        "find {GENRE} movies from {YEAR}",
        "show me {GENRE} films",
        "show me {GENRE} films from {YEAR}",
        "recommend {GENRE} movies",
        "recommend {GENRE} movies released in {YEAR}",
    ],
    "LightOn": [
        "turn on the {ROOM} light",
        "switch on the {ROOM} lamp",
        "turn the {ROOM} lights on",
        "lights on in the {ROOM}",
        "turn on the light",
        "switch the lamp on",
    ],
    "LightOff": [
        "turn off the {ROOM} light",
        "switch off the {ROOM} lamp",
        "turn the {ROOM} lights off",
        "lights off in the {ROOM}",
        "turn off the light",
        "switch the lamp off",
    ],
    "SetBrightness": [
        "set the {ROOM} light to {BRIGHTNESS}",
        "make the {ROOM} lamp {BRIGHTNESS} bright",
        "set brightness in the {ROOM} to {BRIGHTNESS}",
        "dim the {ROOM} lights to {BRIGHTNESS}",
        "set the light to {BRIGHTNESS}",
        "make the lamp {BRIGHTNESS} bright",
    ],
    "OpenBlinds": [
        "open the {ROOM} blinds",
        "raise the {ROOM} blinds",
        "open blinds in the {ROOM}",
        "let the light in by opening the {ROOM} blinds",
        "open the blinds",
        "raise the blinds",
    ],
    "CloseBlinds": [
        "close the {ROOM} blinds",
        "lower the {ROOM} blinds",
        "shut the blinds in the {ROOM}",
        "close the {ROOM} window blinds",
        "close the blinds",
        "lower the blinds",
    ],
    "SetTemperature": [
        "set the temperature to {TEMPERATURE}",
        "make the room {TEMPERATURE}",
        "set dorm temperature to {TEMPERATURE}",
        "change the temperature to {TEMPERATURE}",
        "adjust the room temperature to {TEMPERATURE}",
        "set it to {TEMPERATURE}",
    ],
    "SetScene": [
        "switch to {SCENE} mode",
        "set the room to {SCENE} mode",
        "activate {SCENE} scene",
        "turn on {SCENE} mode",
        "apply the {SCENE} scene",
        "change the room to {SCENE} mode",
    ],
}


def strip_inline_tags(example: str) -> str:
    words = []
    for token in example.split():
        if "/" in token:
            token = token.split("/")[0]
        words.append(token)
    return " ".join(words)


def unique_preserve(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def template_placeholders(template: str):
    return PLACEHOLDER_RE.findall(template)


def annotate_value(slot_name: str, value: str):
    tokens = value.split()
    annotated = []
    for idx, token in enumerate(tokens):
        prefix = "B-" if idx == 0 else "I-"
        annotated.append(f"{token}/{prefix}{slot_name}")
    return annotated


def render_annotated_template(template: str, values: dict[str, str]) -> str:
    rendered = []
    for token in template.split():
        match = PLACEHOLDER_RE.fullmatch(token)
        if match:
            slot_name = match.group(1)
            rendered.extend(annotate_value(slot_name, values[slot_name]))
        else:
            rendered.append(token)
    return " ".join(rendered)


def build_examples_for_intent(intent_name: str, target_count: int = TARGET_EXAMPLES_PER_INTENT):
    examples = list(INTENT_EXAMPLES.get(intent_name, []))
    examples.extend(STATIC_INTENT_EXAMPLES.get(intent_name, []))

    if intent_name in TEMPLATES:
        for template in TEMPLATES[intent_name]:
            placeholders = template_placeholders(template)
            pools = [SLOT_VALUES[placeholder] for placeholder in placeholders]

            if pools:
                combinations = itertools.product(*pools)
            else:
                combinations = [()]

            for combo in combinations:
                values = dict(zip(placeholders, combo))
                examples.append(render_annotated_template(template, values))
                examples = unique_preserve(examples)
                if len(examples) >= target_count:
                    return examples[:target_count]

    examples = unique_preserve(examples)
    if len(examples) < target_count:
        raise ValueError(
            f"Intent {intent_name} only has {len(examples)} examples; "
            f"expected at least {target_count}."
        )

    return examples[:target_count]


def build_dataset(target_count: int = TARGET_EXAMPLES_PER_INTENT):
    dataset = {}
    for intent_name in get_all_intents():
        dataset[intent_name] = build_examples_for_intent(intent_name, target_count=target_count)
    return dataset


def build_records(target_count: int = TARGET_EXAMPLES_PER_INTENT):
    dataset = build_dataset(target_count=target_count)
    records = []

    for intent_name, examples in dataset.items():
        for example in examples:
            records.append(
                {
                    "intent": intent_name,
                    "domain": INTENT_SCHEMA[intent_name]["domain"],
                    "plain_text": strip_inline_tags(example),
                    "annotated": example,
                }
            )

    return records


def write_jsonl(output_path: Path, target_count: int = TARGET_EXAMPLES_PER_INTENT):
    records = build_records(target_count=target_count)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    return records


def main():
    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / "atlas_training_dataset.jsonl"
    records = write_jsonl(output_path)

    print(f"Wrote {len(records)} records to {output_path}")

    counts = {}
    for record in records:
        counts[record["intent"]] = counts.get(record["intent"], 0) + 1

    for intent_name in get_all_intents():
        print(f"{intent_name}: {counts[intent_name]}")


if __name__ == "__main__":
    main()
