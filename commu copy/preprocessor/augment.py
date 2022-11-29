import os
from pathlib import Path
from typing import List, Union

import miditoolkit
import numpy as np
import parmap
import pretty_midi

from .utils.constants import (
    BPM_INTERVAL,
    KEY_NUM_MAP,
    NUM_BPM_AUGMENT,
    NUM_KEY_AUGMENT,
    MAJOR_KEY,
    MINOR_KEY,
)

def get_avg_bpm(event_times: np.ndarray, tempo_infos: np.ndarray, end_time: float) -> int:
    def _normalize(_avg_bpm):
        return _avg_bpm - _avg_bpm % BPM_INTERVAL

    if len(tempo_infos) == 1:
        return _normalize(tempo_infos[-1])

    event_times_with_end_time = np.concatenate([event_times, [end_time]])
    bpm_durations = np.diff(event_times_with_end_time)
    total_bpm = 0
    for duration, bpm in zip(bpm_durations, tempo_infos):
        total_bpm += duration * bpm

    avg_bpm = int(total_bpm / end_time)
    return _normalize(avg_bpm)

def augment_by_key(midi_path: str, augmented_tmp_dir: str, key_change: int) -> Union[Path, str]:
    midi = miditoolkit.MidiFile(midi_path)
    midi_id = Path(midi_path).stem
    pitch_track_notes = [midi.instruments[0].notes]

    for idx, key in enumerate(midi.key_signature_changes):
        origin_key = int(key.key_number)
        if origin_key < MINOR_KEY[0]:
            try:
                midi.key_signature_changes[idx].key_number = MAJOR_KEY[origin_key + key_change]
            except IndexError:
                midi.key_signature_changes[idx].key_number = MAJOR_KEY[
                    origin_key + key_change - len(MAJOR_KEY)
                ]
        else:
            origin_key = origin_key - MINOR_KEY[0]
            try:
                midi.key_signature_changes[idx].key_number = MINOR_KEY[origin_key + key_change]
            except IndexError:
                midi.key_signature_changes[idx].key_number = MINOR_KEY[
                    origin_key + key_change - len(MINOR_KEY)
                ]

    new_key_number = midi.key_signature_changes[0].key_number
    new_key = KEY_NUM_MAP[new_key_number]

    for track in pitch_track_notes:
        for note in track:
            note.pitch = note.pitch + key_change
    try:
        midi.dump(os.path.join(augmented_tmp_dir, midi_id + f"_{new_key}.mid"))
    except ValueError as e:
        print(e, midi_id)
        # exceeds note pitch range
        return None
    return os.path.join(augmented_tmp_dir, midi_id + f"_{new_key}.mid")


def augment_by_bpm(augment_tmp_midi_pth, augmented_dir, bpm_change) -> None:
    midi = pretty_midi.PrettyMIDI(augment_tmp_midi_pth)
    event_times, origin_bpm = midi.get_tempo_changes()

    if len(origin_bpm) > 1:
        origin_bpm = get_avg_bpm(event_times, origin_bpm, midi.get_end_time())

    midi_object = miditoolkit.MidiFile(augment_tmp_midi_pth)
    augment_midi_name = Path(augment_tmp_midi_pth).parts[-1].split(".")[0]

    new_bpm = int(origin_bpm) + bpm_change * BPM_INTERVAL
    midi_object.tempo_changes = [miditoolkit.TempoChange(tempo=new_bpm, time=0)]
    midi_object.dump(os.path.join(augmented_dir, augment_midi_name + f"_{round(new_bpm)}.mid"))


def augment_data_map(
    midi_list: List,
    augmented_dir: str,
    augmented_tmp_dir: str,
) -> None:
    for midi_path in midi_list:
        for key_change in range(-NUM_KEY_AUGMENT, NUM_KEY_AUGMENT):
            augment_tmp_midi_pth = augment_by_key(midi_path, augmented_tmp_dir, key_change)
            if augment_tmp_midi_pth is not None:
                for bpm_change in range(-NUM_BPM_AUGMENT, NUM_BPM_AUGMENT + 1):
                    augment_by_bpm(augment_tmp_midi_pth, augmented_dir, bpm_change)


def augment_data(
    midi_path: Union[str, Path],
    augmented_dir: Union[str, Path],
    augmented_tmp_dir: Union[str, Path],
    num_cores: int,
) -> None:

    midifiles = []

    for _, (dirpath, _, filenames) in enumerate(os.walk(midi_path)):
        midi_extensions = [".mid", ".MID", ".MIDI", ".midi"]
        for ext in midi_extensions:
            tem = [os.path.join(dirpath, _) for _ in filenames if _.endswith(ext)]
            if tem:
                midifiles += tem

    split_midi = np.array_split(np.array(midifiles), num_cores)
    split_midi = [x.tolist() for x in split_midi]
    parmap.map(
        augment_data_map,
        split_midi,
        augmented_dir,
        augmented_tmp_dir,
        pm_pbar=True,
        pm_processes=num_cores,
    )