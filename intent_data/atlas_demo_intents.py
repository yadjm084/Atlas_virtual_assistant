"""Seed intent schema and examples for the Atlas April 14 demo.

This file is the source of truth for:
- the supported demo intents
- the slot inventory
- starter examples using the Activity 2 inline BIO format

The examples below are seed data only.
They still need to be expanded to a larger training set before model training.
"""

MANDATORY_INTENTS = [
    "Greetings",
    "Goodbye",
    "OOS",
    "SetTimer",
    "GetWeather",
]

MOVIE_INTENTS = [
    "MovieOverview",
    "MovieRating",
    "MovieDirector",
    "MovieCast",
    "SimilarMovies",
    "DiscoverByGenre",
]

CONTROL_INTENTS = [
    "LightOn",
    "LightOff",
    "SetBrightness",
    "OpenBlinds",
    "CloseBlinds",
    "SetTemperature",
    "SetScene",
]

INTENT_SCHEMA = {
    "Greetings": {
        "domain": "mandatory",
        "required_slots": [],
        "optional_slots": [],
    },
    "Goodbye": {
        "domain": "mandatory",
        "required_slots": [],
        "optional_slots": [],
    },
    "OOS": {
        "domain": "mandatory",
        "required_slots": [],
        "optional_slots": [],
    },
    "SetTimer": {
        "domain": "mandatory",
        "required_slots": ["DURATION"],
        "optional_slots": ["TIMER_NAME"],
    },
    "GetWeather": {
        "domain": "mandatory",
        "required_slots": ["CITY"],
        "optional_slots": ["DATE"],
    },
    "MovieOverview": {
        "domain": "movie",
        "required_slots": ["TITLE"],
        "optional_slots": [],
    },
    "MovieRating": {
        "domain": "movie",
        "required_slots": ["TITLE"],
        "optional_slots": [],
    },
    "MovieDirector": {
        "domain": "movie",
        "required_slots": ["TITLE"],
        "optional_slots": [],
    },
    "MovieCast": {
        "domain": "movie",
        "required_slots": ["TITLE"],
        "optional_slots": [],
    },
    "SimilarMovies": {
        "domain": "movie",
        "required_slots": ["TITLE"],
        "optional_slots": [],
    },
    "DiscoverByGenre": {
        "domain": "movie",
        "required_slots": ["GENRE"],
        "optional_slots": ["YEAR"],
    },
    "LightOn": {
        "domain": "home_theater",
        "required_slots": [],
        "optional_slots": ["ROOM"],
    },
    "LightOff": {
        "domain": "home_theater",
        "required_slots": [],
        "optional_slots": ["ROOM"],
    },
    "SetBrightness": {
        "domain": "home_theater",
        "required_slots": ["BRIGHTNESS"],
        "optional_slots": ["ROOM"],
    },
    "OpenBlinds": {
        "domain": "home_theater",
        "required_slots": [],
        "optional_slots": ["ROOM"],
    },
    "CloseBlinds": {
        "domain": "home_theater",
        "required_slots": [],
        "optional_slots": ["ROOM"],
    },
    "SetTemperature": {
        "domain": "home_theater",
        "required_slots": ["TEMPERATURE"],
        "optional_slots": [],
    },
    "SetScene": {
        "domain": "home_theater",
        "required_slots": ["SCENE"],
        "optional_slots": [],
    },
}

INTENT_EXAMPLES = {
    "Greetings": [
        "hello atlas",
        "hi atlas",
        "good morning atlas",
        "hey there atlas",
    ],
    "Goodbye": [
        "goodbye atlas",
        "bye atlas",
        "thanks goodbye",
        "talk to you later atlas",
    ],
    "OOS": [
        "book a flight to Toronto",
        "write me a poem about the ocean",
        "translate hello into Japanese",
        "do my grocery shopping",
    ],
    "SetTimer": [
        "set a timer for 10/B-DURATION minutes/I-DURATION",
        "start a study/B-TIMER_NAME timer/I-TIMER_NAME for 25/B-DURATION minutes/I-DURATION",
        "set a pasta/B-TIMER_NAME timer/I-TIMER_NAME for 8/B-DURATION minutes/I-DURATION",
        "start a timer for 45/B-DURATION seconds/I-DURATION",
    ],
    "GetWeather": [
        "what is the weather in Ottawa/B-CITY today/B-DATE",
        "does it rain in Montreal/B-CITY tomorrow/B-DATE",
        "give me the weather for Toronto/B-CITY on Monday/B-DATE",
        "is it cold in New/B-CITY York/I-CITY tonight/B-DATE",
    ],
    "MovieOverview": [
        "give me an overview of Interstellar/B-TITLE",
        "what is Dune/B-TITLE Part/I-TITLE Two/I-TITLE about",
        "tell me the plot of The/B-TITLE Matrix/I-TITLE",
        "what happens in Blade/B-TITLE Runner/I-TITLE 2049/I-TITLE",
    ],
    "MovieRating": [
        "what is the rating for Interstellar/B-TITLE",
        "how highly rated is The/B-TITLE Dark/I-TITLE Knight/I-TITLE",
        "show me the rating of Dune/B-TITLE Part/I-TITLE Two/I-TITLE",
        "what score does Arrival/B-TITLE have",
    ],
    "MovieDirector": [
        "who directed Interstellar/B-TITLE",
        "who is the director of Dune/B-TITLE Part/I-TITLE Two/I-TITLE",
        "tell me who directed The/B-TITLE Matrix/I-TITLE",
        "who made Arrival/B-TITLE",
    ],
    "MovieCast": [
        "who is in the cast of Interstellar/B-TITLE",
        "show me the cast for Dune/B-TITLE Part/I-TITLE Two/I-TITLE",
        "who stars in The/B-TITLE Matrix/I-TITLE",
        "who acted in Arrival/B-TITLE",
    ],
    "SimilarMovies": [
        "find movies like Interstellar/B-TITLE",
        "recommend movies similar to The/B-TITLE Matrix/I-TITLE",
        "what should I watch if I liked Arrival/B-TITLE",
        "show movies like Blade/B-TITLE Runner/I-TITLE 2049/I-TITLE",
    ],
    "DiscoverByGenre": [
        "discover science/B-GENRE fiction/I-GENRE movies from 2024/B-YEAR",
        "find comedy/B-GENRE movies",
        "show me horror/B-GENRE films from 2023/B-YEAR",
        "recommend action/B-GENRE movies released in 2022/B-YEAR",
    ],
    "LightOn": [
        "turn on the bedroom/B-ROOM light/I-ROOM",
        "switch on the desk/B-ROOM lamp/I-ROOM",
        "turn the bedroom/B-ROOM lights/I-ROOM on",
        "lights on in the study/B-ROOM area/I-ROOM",
    ],
    "LightOff": [
        "turn off the bedroom/B-ROOM light/I-ROOM",
        "switch off the desk/B-ROOM lamp/I-ROOM",
        "turn the bedroom/B-ROOM lights/I-ROOM off",
        "lights off in the study/B-ROOM area/I-ROOM",
    ],
    "SetBrightness": [
        "set the bedroom/B-ROOM light/I-ROOM to 40/B-BRIGHTNESS percent/I-BRIGHTNESS",
        "make the desk/B-ROOM lamp/I-ROOM 75/B-BRIGHTNESS percent/I-BRIGHTNESS bright/I-BRIGHTNESS",
        "set brightness in the bedroom/B-ROOM to 20/B-BRIGHTNESS percent/I-BRIGHTNESS",
        "dim the study/B-ROOM area/I-ROOM lights/I-ROOM to 30/B-BRIGHTNESS percent/I-BRIGHTNESS",
    ],
    "OpenBlinds": [
        "open the bedroom/B-ROOM blinds/I-ROOM",
        "raise the bedroom/B-ROOM blinds/I-ROOM",
        "open blinds in the study/B-ROOM area/I-ROOM",
        "let the light in by opening the bedroom/B-ROOM blinds/I-ROOM",
    ],
    "CloseBlinds": [
        "close the bedroom/B-ROOM blinds/I-ROOM",
        "lower the bedroom/B-ROOM blinds/I-ROOM",
        "shut the blinds in the study/B-ROOM area/I-ROOM",
        "close the bedroom/B-ROOM window/I-ROOM blinds/I-ROOM",
    ],
    "SetTemperature": [
        "set the temperature to 21/B-TEMPERATURE degrees/I-TEMPERATURE",
        "make the room 19/B-TEMPERATURE degrees/I-TEMPERATURE",
        "set home theater temperature to 23/B-TEMPERATURE celsius/I-TEMPERATURE",
        "change the temperature to 20/B-TEMPERATURE degrees/I-TEMPERATURE",
    ],
    "SetScene": [
        "switch to study/B-SCENE mode/I-SCENE",
        "set the room to relax/B-SCENE mode/I-SCENE",
        "activate sleep/B-SCENE scene/I-SCENE",
        "turn on movie/B-SCENE mode/I-SCENE",
    ],
}


def get_all_intents():
    return MANDATORY_INTENTS + MOVIE_INTENTS + CONTROL_INTENTS
