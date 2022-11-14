import json
from fractions import Fraction
from pathlib import Path
from typing import Dict, Any, List

from pydantic import BaseModel, validator

from commu.preprocessor.encoder import encoder_utils, TOKEN_OFFSET
from commu.preprocessor.utils import constants
from commu.preprocessor.utils.container import MidiMeta


class ModelArguments(BaseModel):
    checkpoint_dir: str


class TransXlInputData(MidiMeta):
    output_dir: Path

    num_generate: int
    top_k: int
    temperature: float
    chord_progression: List[str]

    @validator("chord_progression")
    def validate_chord_progression_length(cls, value: List[str], values: Dict[str, Any]) -> List[str]:
        num_measures = values.get("num_measures")
        time_signature = values.get("time_signature")
        expected_result = (num_measures - (num_measures % 4)) * Fraction(time_signature) * 8
        result = len(value)
        if expected_result != result:
            raise ValueError("num_measures not matched with chord progression length")
        return value


    @property
    def chord_token_components(self) -> Dict[str, list]:
        event2word, _ = encoder_utils.mk_remi_map()
        event2word = encoder_utils.add_flat_chord2map(event2word)
        event2word = encoder_utils.abstract_chord_types(event2word)

        beats_per_bar = int(Fraction(self.time_signature) * 4)
        chord_idx_lst, unique_cp = encoder_utils.detect_chord(self.chord_progression, beats_per_bar)
        resolution = constants.DEFAULT_POSITION_RESOLUTION
        chord_position = []
        for i in chord_idx_lst:
            if isinstance(i, int):
                chord_position.append(TOKEN_OFFSET.POSITION.value)
            else:
                bit_offset = (float(str(i).split(".")[-1]) * resolution) / (
                        10 ** len(str(i).split(".")[-1])
                )  # 10진수 소수점으로 표현된 position index를 32bit 표현으로 변환
                chord_position.append(int(TOKEN_OFFSET.POSITION.value + bit_offset))

        chord_token = []
        for chord in unique_cp:
            chord = "Chord_" + chord.split("/")[0].split("(")[0]
            chord_token.append(event2word[chord])

        chord_token_components = {
            "chord_token": chord_token,
            "chord_position": chord_position
        }
        return chord_token_components

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(self.json())