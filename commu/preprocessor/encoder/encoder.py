import math

import miditoolkit
import numpy as np

from . import encoder_utils
from .event_tokens import TOKEN_OFFSET
from ..utils.constants import (
    DEFAULT_POSITION_RESOLUTION,
    DEFAULT_TICKS_PER_BEAT,
    SIG_TIME_MAP
)

class EventSequenceEncoder:
    def __init__(self):
        self.event2word, self.word2event = encoder_utils.mk_remi_map()
        self.event2word = encoder_utils.add_flat_chord2map(self.event2word)
        self.event2word = encoder_utils.abstract_chord_types(self.event2word)
        self.position_resolution = DEFAULT_POSITION_RESOLUTION

    def encode(self, midi_paths, sample_info=None, for_cp=False):
        midi_file = miditoolkit.MidiFile(midi_paths)
        ticks_per_beat = midi_file.ticks_per_beat
        chord_progression = sample_info["chord_progressions"]
        num_measures = math.ceil(sample_info["num_measures"])
        numerator = int(sample_info["time_signature"].split("/")[0])
        denominator = int(sample_info["time_signature"].split("/")[1])
        is_incomplete_measure = sample_info["is_incomplete_measure"]

        beats_per_bar = numerator / denominator * 4
        ticks_per_bar = int(ticks_per_beat * beats_per_bar)
        duration_bins = np.arange(
            int(ticks_per_bar / self.position_resolution),
            ticks_per_bar + 1,
            int(ticks_per_bar / self.position_resolution),
            dtype=int,
        )

        events = encoder_utils.extract_events(
            midi_paths,
            duration_bins,
            ticks_per_bar=ticks_per_bar,
            ticks_per_beat=ticks_per_beat,
            chord_progression=chord_progression,
            num_measures=num_measures,
            is_incomplete_measure=is_incomplete_measure,
        )
        if for_cp:
            return events

        words = []
        for event in events:
            e = "{}_{}".format(event.name, event.value)
            if e in self.event2word:
                words.append(self.event2word[e])
            else:
                # OOV
                if event.name == "Note Velocity":
                    # replace with max velocity based on our training data
                    words.append(self.event2word["Note Velocity_63"])
                if event.name == "Note Duration":
                    # replace with max duration
                    words.append(self.event2word[f"Note Duration_{self.position_resolution-1}"])
                else:
                    # something is wrong
                    # you should handle it for your own purpose
                    print("OOV {}".format(e))
        words.append(TOKEN_OFFSET.EOS.value)  # eos token
        return np.array(words)

    def decode(
        self,
        midi_info,
    ):
        time_sig_word = midi_info.time_signature
        time_sig = SIG_TIME_MAP[time_sig_word - TOKEN_OFFSET.TS.value - 1]
        numerator = int(time_sig.split("/")[0])
        denominator = int(time_sig.split("/")[1])
        beats_per_bar = int(numerator/denominator * 4)

        ticks_per_bar = DEFAULT_TICKS_PER_BEAT * beats_per_bar

        duration_bins = np.arange(
            int(ticks_per_bar / self.position_resolution),
            ticks_per_bar + 1,
            int(ticks_per_bar / self.position_resolution),
            dtype=int,
        )

        decoded_midi = encoder_utils.write_midi(
            midi_info,
            self.word2event,
            duration_bins=duration_bins,
            beats_per_bar=beats_per_bar,
        )

        return decoded_midi