import enum
from typing import List, Optional

class KeySwitchVelocity(int, enum.Enum):
    DEFAULT = 1

    @classmethod
    def get_value(cls, key: Optional[str]) -> int:
        key = key or "DEFAULT"
        if hasattr(cls, key):
            return getattr(cls, key).value
        return cls.DEFAULT.value

class ChordType(str, enum.Enum):
    MAJOR = "major"
    MINOR = "minor"

    @classmethod
    def values(cls) -> List[str]:
        return list(cls.__members__.values())

BPM_INTERVAL = 5
CHORD_TRACK_NAME = "chord"
DEFAULT_NUM_BEATS = 4
DEFAULT_POSITION_RESOLUTION = 128
DEFAULT_TICKS_PER_BEAT = 480
MAX_BPM = 200
NUM_BPM_AUGMENT = 2
NUM_KEY_AUGMENT = 6
UNKNOWN = "unknown"
VELOCITY_INTERVAL = 2

MAJOR_KEY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
MINOR_KEY = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

KEY_MAP = {
    "cmajor": 0,
    "c#major": 1,
    "dbmajor": 1,
    "dmajor": 2,
    "d#major": 3,
    "ebmajor": 3,
    "emajor": 4,
    "fmajor": 5,
    "f#major": 6,
    "gbmajor": 6,
    "gmajor": 7,
    "g#major": 8,
    "abmajor": 8,
    "amajor": 9,
    "a#major": 10,
    "bbmajor": 10,
    "bmajor": 11,
    "cminor": 12,
    "c#minor": 13,
    "dbminor": 13,
    "dminor": 14,
    "d#minor": 15,
    "ebminor": 15,
    "eminor": 16,
    "fminor": 17,
    "f#minor": 18,
    "gbminor": 18,
    "gminor": 19,
    "g#minor": 20,
    "abminor": 20,
    "aminor": 21,
    "a#minor": 22,
    "bbminor": 22,
    "bminor": 23,
}

KEY_NUM_MAP = {v: k for k, v in KEY_MAP.items()}

TIME_SIG_MAP = {
    "4/4": 0,
    "3/4": 1,
    "6/8": 2,
    "12/8": 3,
}

SIG_TIME_MAP = {v: k for k, v in TIME_SIG_MAP.items()}

PITCH_RANGE_MAP = {
    "very_low": 0,
    "low": 1,
    "mid_low": 2,
    "mid": 3,
    "mid_high": 4,
    "high": 5,
    "very_high": 6,
}

INST_MAP = {
    "accordion": 1,
    "acoustic_bass": 3,
    "acoustic_guitar": 3,
    "acoustic_piano": 0,
    "banjo": 3,
    "bassoon": 5,
    "bell": 2,
    "brass_ensemble": 5,
    "celesta": 2,
    "choir": 7,
    "clarinet": 5,
    "drums_full": 6,
    "drums_tops": 6,
    "electric_bass": 3,
    "electric_guitar_clean": 3,
    "electric_guitar_distortion": 3,
    "electric_piano": 0,
    "fiddle": 4,
    "flute": 5,
    "glockenspiel": 2,
    "harp": 3,
    "harpsichord": 0,
    "horn": 5,
    "keyboard": 0,
    "mandolin": 3,
    "marimba": 2,
    "nylon_guitar": 3,
    "oboe": 5,
    "organ": 0,
    "oud": 3,
    "pad_synth": 4,
    "percussion": 6,
    "recorder": 5,
    "sitar": 3,
    "string_cello": 4,
    "string_double_bass": 4,
    "string_ensemble": 4,
    "string_viola": 4,
    "string_violin": 4,
    "synth_bass": 3,
    "synth_bass_808": 3,
    "synth_bass_wobble": 3,
    "synth_bell": 2,
    "synth_lead": 1,
    "synth_pad": 4,
    "synth_pluck": 7,
    "synth_voice": 7,
    "timpani": 6,
    "trombone": 5,
    "trumpet": 5,
    "tuba": 5,
    "ukulele": 3,
    "vibraphone": 2,
    "whistle": 7,
    "xylophone": 2,
    "zither": 3,
    "orgel": 2,
    "synth_brass": 5,
    "sax": 5,
    "bamboo_flute": 5,
    "yanggeum": 3,
    "vocal": 8,
}

GENRE_MAP = {
    "newage": 0,
    "cinematic": 1,
}

TRACK_ROLE_MAP = {
    "main_melody": 0,
    "sub_melody": 1,
    "accompaniment": 2,
    "bass": 3,
    "pad": 4,
    "riff": 5,
}

RHYTHM_MAP = {
    "standard": 0,
    "triplet": 1,
}

