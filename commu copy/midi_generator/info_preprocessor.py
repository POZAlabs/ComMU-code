from typing import List, Any

from commu.midi_generator.container import TransXlInputData
from commu.preprocessor.encoder import MetaEncoder
from commu.preprocessor.utils.container import MidiMeta


def parse_meta(**kwargs: Any) -> MidiMeta:
    return MidiMeta(**kwargs)


def encode_meta(meta_encoder: MetaEncoder, midi_meta: MidiMeta) -> List[int]:
    return meta_encoder.encode(midi_meta)


def normalize_chord_progression(chord_progression: str) -> List[str]:
    return chord_progression.split("-")


class PreprocessTask:
    def __init__(self):
        self.input_data = None
        self.midi_meta = None

    def get_meta_info_length(self):
        return len(self.midi_meta.__fields__)

    def normalize_input_data(self, input_data: dict):
        input_data["chord_progression"] = normalize_chord_progression(input_data["chord_progression"])
        self.input_data = TransXlInputData(**input_data)

    def preprocess(self) -> List[int]:
        self.midi_meta = parse_meta(**self.input_data.dict())
        meta_encoder = MetaEncoder()
        encoded_meta = encode_meta(
            meta_encoder=meta_encoder, midi_meta=self.midi_meta
        )
        return encoded_meta

    def excecute(self, input_data: dict) -> List[int]:
        if self.input_data is None:
            self.normalize_input_data(input_data)

        encoded_meta = self.preprocess()
        return encoded_meta