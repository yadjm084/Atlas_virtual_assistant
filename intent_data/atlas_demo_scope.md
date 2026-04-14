## Atlas Demo Scope

This document freezes the supported intent scope for the April 14 demo.

The goal is a stable end-to-end path:
- user verification
- wake word detection
- speech recognition
- intent detection and slot filling
- fulfillment
- answer generation
- text to speech

### Supported intent groups

#### Mandatory intents
- `Greetings`
- `Goodbye`
- `OOS`
- `SetTimer`
- `GetWeather`

#### Movie intents
- `MovieOverview`
- `MovieRating`
- `MovieDirector`
- `MovieCast`
- `SimilarMovies`
- `DiscoverByGenre`

#### Home theater control intents
- `LightOn`
- `LightOff`
- `SetBrightness`
- `OpenBlinds`
- `CloseBlinds`
- `SetTemperature`
- `SetScene`

### Slot inventory

#### Shared rules
- Slots use BIO labels in the training dataset.
- Optional slots should be filled with defaults during fulfillment, not by the classifier.
- Slot names are frozen for the demo so the dataset, model, and fulfillment code all agree.

#### Slot definitions
- `TITLE`: movie title
- `GENRE`: movie genre
- `YEAR`: four-digit release year
- `CITY`: weather city
- `DATE`: relative or absolute day reference
- `DURATION`: timer duration with unit
- `TIMER_NAME`: optional named timer
- `ROOM`: home theater zone or area
- `BRIGHTNESS`: brightness percentage or level
- `TEMPERATURE`: target temperature value
- `SCENE`: scene preset name

### Intent contract

| Intent | Domain | Required slots | Optional slots | Fulfillment target |
| --- | --- | --- | --- | --- |
| `Greetings` | mandatory | none | none | canned response |
| `Goodbye` | mandatory | none | none | canned response |
| `OOS` | mandatory | none | none | refusal / fallback response |
| `SetTimer` | mandatory | `DURATION` | `TIMER_NAME` | local timer state |
| `GetWeather` | mandatory | `CITY` | `DATE` | weather API |
| `MovieOverview` | movie | `TITLE` | none | TMDB search + details |
| `MovieRating` | movie | `TITLE` | none | TMDB search + details |
| `MovieDirector` | movie | `TITLE` | none | TMDB search + credits |
| `MovieCast` | movie | `TITLE` | none | TMDB search + credits |
| `SimilarMovies` | movie | `TITLE` | none | TMDB search + similar |
| `DiscoverByGenre` | movie | `GENRE` | `YEAR` | TMDB discover |
| `LightOn` | home_theater | none | `ROOM` | home theater state change |
| `LightOff` | home_theater | none | `ROOM` | home theater state change |
| `SetBrightness` | home_theater | `BRIGHTNESS` | `ROOM` | home theater state change |
| `OpenBlinds` | home_theater | none | `ROOM` | home theater state change |
| `CloseBlinds` | home_theater | none | `ROOM` | home theater state change |
| `SetTemperature` | home_theater | `TEMPERATURE` | none | home theater state change |
| `SetScene` | home_theater | `SCENE` | none | home theater state change |

### Default behavior

- `GetWeather` without `DATE`: default to `today`
- `SetTimer` without `TIMER_NAME`: use generic timer label
- `LightOn`, `LightOff`, `SetBrightness`, `OpenBlinds`, `CloseBlinds` without `ROOM`: default to `theater`
- `DiscoverByGenre` without `YEAR`: omit year filter

### Home theater state for the demo

The UI and fulfillment layer should support this minimum state:
- `room`: `bedroom`
- `light`: `on` or `off`
- `brightness`: `0-100`
- `blinds`: `open` or `closed`
- `temperature_c`: numeric
- `scene`: one of `study`, `relax`, `sleep`, `movie`

### Demo commands to support

- "hello atlas"
- "set a study timer for 10 minutes"
- "does it rain in Ottawa today"
- "who directed Dune Part Two"
- "what is the rating for Interstellar"
- "find movies like The Matrix"
- "discover science fiction movies from 2024"
- "turn on the theater light"
- "set the theater light to 40 percent"
- "close the theater blinds"
- "set the temperature to 21 degrees"
- "switch to study mode"

### Explicitly out of scope for April 14

- multi-turn slot recovery dialogs
- multilingual support
- live home-theater hardware integration
- emotion-aware behavior
- broad free-form LLM agent behavior

### Next implementation steps

1. Expand the seed examples into a full training dataset.
2. Train the joint intent and slot model.
3. Replace the hardcoded intent stub in `app.py`.
4. Build fulfillment handlers matching this intent contract.
