import copy
import re
from pathlib import Path
from typing import Any, Dict, Union

from .. import utils
from ..utils import constants
from ..utils.container import MidiMeta

class MetaParser:
    def __init__(self):
        pass

    def parse(self, meta_dict: Dict[str, Any], midi_path: Union[str, Path]) -> MidiMeta:
        copied_meta_dict = copy.deepcopy(meta_dict)
        audio_key = copied_meta_dict["audio_key"]
        if not (constants.ChordType.MAJOR in audio_key or constants.ChordType.MINOR in audio_key):
            copied_meta_dict["audio_key"] = (
                copied_meta_dict["audio_key"] + copied_meta_dict["chord_type"]
            )
        copied_meta_dict["inst"] = remove_number_from_inst(copied_meta_dict["inst"])
        copied_meta_dict["chord_progression"] = copied_meta_dict.pop("chord_progressions")[0]
        min_velocity, max_velocity = utils.get_velocity_range(
            midi_path,
            keyswitch_velocity=constants.KeySwitchVelocity.get_value(copied_meta_dict["inst"]),
        )
        midi_meta = MidiMeta(
            **copied_meta_dict,
            min_velocity=min_velocity,
            max_velocity=max_velocity,
        )
        return midi_meta

def remove_number_from_inst(inst: str) -> str:
    """`{inst}-[0-9]` => `{inst}`"""
    inst_number_pattern = re.compile("-[0-9]+")
    return inst_number_pattern.sub("", inst)
