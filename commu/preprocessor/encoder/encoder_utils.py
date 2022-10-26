import copy
from typing import Dict

import miditoolkit
import numpy as np

from .event_tokens import base_event, TOKEN_OFFSET
from ..utils.constants import (
    BPM_INTERVAL,
    DEFAULT_POSITION_RESOLUTION,
    DEFAULT_TICKS_PER_BEAT,
    VELOCITY_INTERVAL,
    SIG_TIME_MAP,
    KEY_NUM_MAP
)

NUM_VELOCITY_BINS = int(128 / VELOCITY_INTERVAL)
DEFAULT_VELOCITY_BINS = np.linspace(2, 127, NUM_VELOCITY_BINS, dtype=np.int)

class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return "Item(name={}, start={}, end={}, velocity={}, pitch={})".format(
            self.name, self.start, self.end, self.velocity, self.pitch
        )


class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return "Event(name={}, time={}, value={}, text={})".format(
            self.name, self.time, self.value, self.text
        )


def mk_remi_map():
    event = copy.deepcopy(base_event)
    for i in range(DEFAULT_POSITION_RESOLUTION):
        event.append(f"Note Duration_{i}")
    for i in range(1, DEFAULT_POSITION_RESOLUTION + 1):
        event.append(f"Position_{i}/{DEFAULT_POSITION_RESOLUTION}")

    event2word = {k: v for k, v in zip(event, range(2, len(event) + 2))}
    word2event = {v: k for k, v in zip(event, range(2, len(event) + 2))}

    return event2word, word2event

def add_flat_chord2map(event2word: Dict):
    flat_chord = ["Chord_ab:", "Chord_bb:", "Chord_db:", "Chord_eb:", "Chord_gb:"]
    scale = [
        "",
        "maj",
        "maj7",
        "7",
        "dim",
        "dim7",
        "+",
        "m",
        "m7",
        "sus4",
        "7sus4",
        "m6",
        "m7b5",
        "sus2",
        "add2",
        "6",
        "madd2",
        "mM7",
    ]

    flat_chords = []
    for c in flat_chord:
        for s in scale:
            flat_chords.append(c + s)

    for c in flat_chords:
        scale = c.split(":")[1]
        key = c.split(":")[0].split("_")[1][0]
        c = c.replace(":", "")
        if c.startswith("Chord_ab"):
            if scale == "" or scale == "maj" or scale == "6":
                event2word[c] = event2word["Chord_g#"]
            elif scale == "maj7" or scale == "add2" or scale == "sus2":
                event2word[c] = event2word["Chord_g#maj7"]
            elif scale == "7":
                event2word[c] = event2word["Chord_g#7"]
            elif scale == "dim" or scale == "dim7":
                event2word[c] = event2word["Chord_g#dim"]
            elif scale == "+":
                event2word[c] = event2word["Chord_g#+"]
            elif scale == "m" or scale == "m6" or scale == "mM7":
                event2word[c] = event2word["Chord_g#m"]
            elif scale == "m7" or scale == "madd2":
                event2word[c] = event2word["Chord_g#m7"]
            elif scale == "sus4" or scale == "7sus4":
                event2word[c] = event2word["Chord_g#sus4"]
            elif scale == "m7b5":
                event2word[c] = event2word["Chord_g#m7b5"]
        else:
            if scale == "" or scale == "maj" or scale == "6":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#"
                event2word[c] = event2word[word]
            elif scale == "maj7" or scale == "add2" or scale == "sus2":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#maj7"
                event2word[c] = event2word[word]
            elif scale == "7":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#7"
                event2word[c] = event2word[word]
            elif scale == "dim" or scale == "dim7":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#dim"
                event2word[c] = event2word[word]
            elif scale == "+":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#+"
                event2word[c] = event2word[word]
            elif scale == "m" or scale == "m6" or scale == "mM7":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#m"
                event2word[c] = event2word[word]
            elif scale == "m7" or scale == "madd2":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#m7"
                event2word[c] = event2word[word]
            elif scale == "sus4" or scale == "7sus4":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#sus4"
                event2word[c] = event2word[word]
            elif scale == "m7b5":
                new_key = chr(ord(key) - 1)
                word = "Chord_" + new_key + "#m7b5"
                event2word[c] = event2word[word]

    return event2word

def abstract_chord_types(event2word):
    chord = ["Chord_a:", "Chord_b:", "Chord_c:", "Chord_d:", "Chord_e:", "Chord_f:", "Chord_g:"]
    scale = ["7sus4", "m6", "sus2", "add2", "dim7", "6", "madd2", "mM7", ]

    chords = []
    for c in chord:
        for s in scale:
            chords.append(c + s)

    for c in chords:
        scale = c.split(":")[1]
        key = c.split(":")[0].split("_")[1][0]
        c = c.replace(":", "")
        if scale == "7sus4":
            word = "Chord_" + key + "sus4"
            event2word[c] = event2word[word]
        if scale == "m6":
            word = "Chord_" + key + "m"
            event2word[c] = event2word[word]
        if scale == "sus2" or scale == "add2":
            word = "Chord_" + key + "maj7"
            event2word[c] = event2word[word]
        if scale == "6":
            word = "Chord_" + key
            event2word[c] = event2word[word]
        if scale == "dim7":
            word = "Chord_" + key + "dim"
            event2word[c] = event2word[word]
        if scale == "madd2" or scale == "mM7":
            word = "Chord_" + key + "m7"
            event2word[c] = event2word[word]

    return event2word

def extract_events(
    input_path,
    duration_bins,
    ticks_per_bar=None,
    ticks_per_beat=None,
    chord_progression=None,
    num_measures=None,
    is_incomplete_measure=None,
):
    note_items = read_items(input_path)
    max_time = note_items[-1].end
    if not chord_progression[0]:
        return None
    else:
        items = note_items
    groups = group_items(items, max_time, ticks_per_bar)
    events = item2event(groups, duration_bins)
    beats_per_bar = int(ticks_per_bar/ticks_per_beat)

    if chord_progression:
        new_chords = chord_progression[0]
        events = insert_chord_on_event(
            events,
            new_chords,
            ticks_per_bar,
            num_measures,
            is_incomplete_measure,
            beats_per_bar,
        )

    return events

def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(
            Item(
                name="Note",
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                pitch=note.pitch,
            )
        )
    note_items.sort(key=lambda x: x.start)
    return note_items

def group_items(items, max_time, ticks_per_bar):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        if not insiders:
            insiders.append(Item(name="None", start=None, end=None, velocity=None, pitch="NN"))
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

def item2event(groups, duration_bins):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if "NN" in [item.pitch for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        if groups[i][1].name == "Chord":
            events.append(Event(name="Bar", time=bar_st, value=None, text="{}".format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_POSITION_RESOLUTION, endpoint=False)
            index = np.argmin(abs(flags - item.start))
            events.append(
                Event(
                    name="Position",
                    time=item.start,
                    value="{}/{}".format(index + 1, DEFAULT_POSITION_RESOLUTION),
                    text="{}".format(item.start),
                )
            )
            if item.name == "Note":
                # velocity
                velocity_index = (
                    np.searchsorted(DEFAULT_VELOCITY_BINS, item.velocity, side="right") - 1
                )
                events.append(
                    Event(
                        name="Note Velocity",
                        time=item.start,
                        value=velocity_index,
                        text="{}/{}".format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index]),
                    )
                )
                # pitch
                events.append(
                    Event(
                        name="Note On",
                        time=item.start,
                        value=item.pitch,
                        text="{}".format(item.pitch),
                    )
                )
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(duration_bins - duration))
                events.append(
                    Event(
                        name="Note Duration",
                        time=item.start,
                        value=index,
                        text="{}/{}".format(duration, duration_bins[index]),
                    )
                )
            elif item.name == "Chord":
                events.append(
                    Event(
                        name="Chord",
                        time=item.start,
                        value=item.pitch,
                        text="{}".format(item.pitch),
                    )
                )
    return events

def insert_chord_on_event(
    events,
    chord_progression,
    tick_per_bar,
    num_measures,
    is_incomplete_measure,
    beats_per_bar,
):
    chord_idx_lst, chords = detect_chord(chord_progression, beats_per_bar)
    start_time = tick_per_bar * is_incomplete_measure
    chord_events = []
    for i in range(num_measures):
        chord_events.append(
            Event(name="Bar", time=i * tick_per_bar, value=None, text="{}".format(i + 1))
        )
        while chord_idx_lst and chord_idx_lst[0] < i + 1 - is_incomplete_measure:
            chord_position = chord_idx_lst.pop(0)
            chord_time = int(chord_position * tick_per_bar + start_time)
            chord = chords.pop(0)
            chord_events.append(
                Event(
                    name="Position",
                    time=chord_time,
                    value="{}/{}".format(
                        int((chord_position - i + is_incomplete_measure) * DEFAULT_POSITION_RESOLUTION) + 1,
                        DEFAULT_POSITION_RESOLUTION
                    ),
                    text=chord_time,
                )
            )
            chord_events.append(
                Event(name="Chord",
                      time=chord_time,
                      value=chord.split("/")[0].split("(")[0],
                      text=chord.split("/")[0].split("(")[0])
            )

    inserted_events = chord_events + events
    inserted_events.sort(key=lambda x: x.time)
    return inserted_events

def detect_chord(chord_progression, beats_per_bar):
    chords_per_bar = beats_per_bar * 2
    num_measures = int(len(chord_progression)/chords_per_bar)
    split_by_bar = np.array_split(np.array(chord_progression), num_measures)
    chord_idx = []
    chord_name = []
    for bar_idx, bar in enumerate(split_by_bar):
        for c_idx, chord in enumerate(bar):
            chord = chord.lower()
            if c_idx == 0 or chord != chord_name[-1]:
                chord_idx.append(bar_idx + c_idx / chords_per_bar)
                chord_name.append(chord)
    return chord_idx, chord_name

def word_to_event(words, word2event):
    events = []
    for word in words:
        try:
            event_name, event_value = word2event[word].split("_")
        except KeyError:
            if word == 1:
                # 따로 디코딩 되지 않는 EOS
                continue
            else:
                print(f"OOV: {word}")
            continue
        events.append(Event(event_name, None, event_value, None))
    return events

def write_midi(
    midi_info,
    word2event,
    duration_bins,
    beats_per_bar,
):
    events = word_to_event(midi_info.event_seq, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    for i in range(len(events) - 3):
        if events[i].name == "Bar" and i > 0:
            temp_notes.append("Bar")
            temp_chords.append("Bar")
        elif (
            events[i].name == "Position"
            and events[i + 1].name == "Note Velocity"
            and events[i + 2].name == "Note On"
            and events[i + 3].name == "Note Duration"
        ):
            # start time and end time from position
            position = int(events[i].value.split("/")[0]) - 1
            # velocity
            index = int(events[i + 1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i + 2].value)
            # duration
            index = int(events[i + 3].value)
            duration = duration_bins[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])
        elif events[i].name == "Position" and events[i + 1].name == "Chord":
            position = int(events[i].value.split("/")[0]) - 1
            temp_chords.append([position, events[i + 1].value])
    # get specific time for notes
    ticks_per_beat = DEFAULT_TICKS_PER_BEAT
    ticks_per_bar = ticks_per_beat * beats_per_bar
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == "Bar":
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(
                int(current_bar_st),
                int(current_bar_et),
                int(DEFAULT_POSITION_RESOLUTION),
                endpoint=False,
                dtype=int,
            )
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))
    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        current_bar = 0
        for chord in temp_chords:
            if chord == "Bar":
                current_bar += 1
            else:
                position, value = chord
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(
                    current_bar_st, current_bar_et, DEFAULT_POSITION_RESOLUTION, endpoint=False, dtype=int
                )
                st = flags[position]
                chords.append([st, value])

    midi = miditoolkit.midi.parser.MidiFile()
    numerator, denominator = SIG_TIME_MAP[
        midi_info.time_signature
        - (TOKEN_OFFSET.TS.value + 1)
    ].split("/")
    ts = miditoolkit.midi.containers.TimeSignature(
        numerator=int(numerator), denominator=int(denominator), time=0
    )
    key_num = midi_info.audio_key - (TOKEN_OFFSET.KEY.value + 1)
    ks = miditoolkit.KeySignature(
        key_name=KEY_NUM_MAP[key_num],
        time=0)
    midi.time_signature_changes.append(ts)
    midi.key_signature_changes.append(ks)
    midi.ticks_per_beat = DEFAULT_TICKS_PER_BEAT
    # write instrument
    inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
    inst.notes = notes
    midi.instruments.append(inst)
    # write bpm info
    tempo_changes = []
    tempo_changes.append(
        miditoolkit.midi.containers.TempoChange(
            (midi_info.bpm - TOKEN_OFFSET.BPM.value)
            * BPM_INTERVAL,
            0,
        )
    )
    midi.tempo_changes = tempo_changes

    # write chord into marker
    if len(temp_chords) > 0:
        for c in chords:
            midi.markers.append(miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))

    return midi
