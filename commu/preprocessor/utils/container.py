from dataclasses import dataclass
from typing import List, Union

from pydantic import BaseModel

@dataclass
class MidiInfo:
    # meta
    bpm: int
    audio_key: int
    time_signature: int
    pitch_range: int
    num_measures: int
    inst: int
    genre: str
    min_velocity: int
    max_velocity: int
    track_category: int
    rhythm: int
    # event
    event_seq: List[int]

class MidiMeta(BaseModel):
    bpm: Union[int, str]
    audio_key: str
    time_signature: str
    pitch_range: str
    num_measures: Union[float, str]
    inst: str
    genre: str
    min_velocity: Union[int, str]
    max_velocity: Union[int, str]
    track_category: str
    rhythm: str

