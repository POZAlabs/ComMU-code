import copy
import re
from typing import Any, Dict
from ..utils.container import MidiMeta

class MetaParser:
    def __init__(self):
        pass

    def parse(self, meta_dict: Dict[str, Any]) -> MidiMeta:
        copied_meta_dict = copy.deepcopy(meta_dict)
        copied_meta_dict["inst"] = remove_number_from_inst(copied_meta_dict["inst"])

        copied_meta_dict["chord_progression"] = copied_meta_dict.pop("chord_progressions")[0]

        midi_meta = MidiMeta(
            **copied_meta_dict,
        )
        return midi_meta

def remove_number_from_inst(inst: str) -> str:
    """`{inst}-[0-9]` => `{inst}`"""
    inst_number_pattern = re.compile("-[0-9]+")
    return inst_number_pattern.sub("", inst)
