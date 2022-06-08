import re
from pathlib import Path
from typing import Optional, Tuple, Union

import miditoolkit

from .constants import (
    CHORD_TRACK_NAME,
    UNKNOWN,
)

def get_velocity_range(
    midi_path: Union[str, Path], keyswitch_velocity: Optional[int] = None
) -> Tuple[Union[int, str], Union[int, str]]:
    midi = miditoolkit.MidiFile(str(midi_path))
    velocities = []
    for track in midi.instruments:
        if track.name == CHORD_TRACK_NAME:
            continue
        for note in track.notes:
            if keyswitch_velocity is not None:
                if note.velocity != keyswitch_velocity:
                    velocities.append(note.velocity)
            else:
                velocities.append(note.velocity)

    if not velocities or max(velocities) == 0:
        return UNKNOWN, UNKNOWN
    return min(velocities), max(velocities)

def get_time_signature(midi_path):
    time_signature = miditoolkit.MidiFile(midi_path).time_signature_changes[0]
    numerator = time_signature.numerator
    denominator = time_signature.denominator
    return numerator, denominator

def sync_key_augment(chords, aug_key, origin_key):
    chord_lst = [
        "a",
        "a#",
        "b",
        "c",
        "c#",
        "d",
        "d#",
        "e",
        "f",
        "f#",
        "g",
        "g#",
        "ab",
        "bb",
        "db",
        "eb",
        "gb",
    ]
    chord2symbol = {k: v for k, v in zip(chord_lst, range(12))}
    chord2symbol["ab"] = 11
    chord2symbol["bb"] = 1
    chord2symbol["db"] = 4
    chord2symbol["eb"] = 6
    chord2symbol["gb"] = 9
    symbol2chord = {v: k for k, v in chord2symbol.items()}

    basic_chord = []
    for c in chords:
        match = re.match(r"[A-G](#|b|)", c)
        basic_chord.append(match[0])

    chord_type = [c.replace(b, "") for c, b in zip(chords, basic_chord)]
    symbol_lst = [chord2symbol[c.lower()] for c in basic_chord]

    origin_key_symbol = chord2symbol[origin_key]

    augment_key_symbol = chord2symbol[aug_key]

    key_diff = origin_key_symbol - augment_key_symbol
    key_change = abs(key_diff)
    if key_diff < 0:
        new_symbol_lst = []
        for s in symbol_lst:
            new_s = s + key_change
            if new_s >= 12:
                new_s = new_s - 12
            new_symbol_lst.append(new_s)
    else:
        new_symbol_lst = []
        for s in symbol_lst:
            new_s = s - key_change
            if new_s < 0:
                new_s = new_s + 12
            new_symbol_lst.append(new_s)

    new_chord_lst = [symbol2chord[s] for s in new_symbol_lst]
    new_chord_lst = [c + t for c, t in zip(new_chord_lst, chord_type)]
    return [new_chord_lst]