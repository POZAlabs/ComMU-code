import enum
import functools
import inspect
import math
from typing import Any, Callable, Dict, List, Union

from commu.preprocessor.utils.exceptions import ErrorMessage, UnprocessableMidiError
from ..utils import constants
from ..utils.container import MidiMeta
from .event_tokens import TOKEN_OFFSET

EncodeFunc = Union[Callable[[Any], int], Callable[[Any, Dict[Any, int]], int]]
META_ENCODING_ORDER = tuple(MidiMeta.__fields__.keys())
DEFAULT_ENCODING_MAPS = {
    "audio_key": constants.KEY_MAP,
    "time_signature": constants.TIME_SIG_MAP,
    "pitch_range": constants.PITCH_RANGE_MAP,
    "inst": constants.INST_MAP,
    "genre": constants.GENRE_MAP,
    "track_role": constants.TRACK_ROLE_MAP,
    "rhythm": constants.RHYTHM_MAP,
}
ATTR_ALIAS = {
    "min_velocity": "velocity",
    "max_velocity": "velocity",
}


class AliasMixin:
    @classmethod
    def get(cls, key: str):
        key = key.lower()
        if key in ATTR_ALIAS:
            return getattr(cls, ATTR_ALIAS[key].upper())
        return getattr(cls, key.upper())


class Unknown(AliasMixin, int, enum.Enum):
    BPM = TOKEN_OFFSET.BPM.value
    AUDIO_KEY = TOKEN_OFFSET.KEY.value
    TIME_SIGNATURE = TOKEN_OFFSET.TS.value
    PITCH_RANGE = TOKEN_OFFSET.PITCH_RANGE.value
    INST = TOKEN_OFFSET.INST.value
    GENRE = TOKEN_OFFSET.GENRE.value
    VELOCITY = TOKEN_OFFSET.VELOCITY.value
    TRACK_ROLE = TOKEN_OFFSET.TRACK_ROLE.value
    RHYTHM = TOKEN_OFFSET.RHYTHM.value


class Offset(AliasMixin, int, enum.Enum):
    BPM = TOKEN_OFFSET.BPM.value
    AUDIO_KEY = TOKEN_OFFSET.KEY.value + 1
    TIME_SIGNATURE = TOKEN_OFFSET.TS.value + 1
    PITCH_RANGE = TOKEN_OFFSET.PITCH_RANGE.value + 1
    MEASURES_4 = TOKEN_OFFSET.NUM_MEASURES.value
    MEASURES_8 = TOKEN_OFFSET.NUM_MEASURES.value + 1
    MEASURES_16 = TOKEN_OFFSET.NUM_MEASURES.value + 2
    INST = TOKEN_OFFSET.INST.value + 1
    GENRE = TOKEN_OFFSET.GENRE.value + 1
    VELOCITY = TOKEN_OFFSET.VELOCITY.value + 1
    TRACK_ROLE = TOKEN_OFFSET.TRACK_ROLE.value + 1
    RHYTHM = TOKEN_OFFSET.RHYTHM.value + 1


ENCODERS: Dict[str, EncodeFunc] = dict()


def _get_meta_name(func_name: str) -> str:
    return "_".join(func_name.split("_")[1:])


def register_encoder(func):
    ENCODERS[_get_meta_name(func.__name__)] = func
    return func


def inject_args_to_encode_func(encode_func, *args, **kwargs) -> int:
    num_args = len(inspect.getfullargspec(encode_func).args)
    if num_args == 1:
        return encode_func(args[0])
    return encode_func(*args, **kwargs)


def encode_unknown(
    raise_error: bool = False, error_message: str = ErrorMessage.UNPROCESSABLE_MIDI_ERROR.value
):
    def decorator(func: EncodeFunc):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            meta_name = _get_meta_name(func.__name__)
            if args[0] == constants.UNKNOWN:
                if raise_error:
                    raise UnprocessableMidiError(error_message)
                return Unknown.get(meta_name).value
            return inject_args_to_encode_func(func, *args, **kwargs)

        return wrapper

    return decorator


def add_offset(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        meta_name = _get_meta_name(func.__name__).upper()
        offset_value = Offset.get(meta_name).value
        unknown_value = Unknown.get(meta_name).value
        result = inject_args_to_encode_func(func, *args, **kwargs)
        if result == unknown_value:
            return result
        return result + offset_value

    return wrapper


@register_encoder
@add_offset
@encode_unknown()
def encode_bpm(bpm: Union[int, str]) -> int:
    bpm_meta = min(bpm, constants.MAX_BPM) // constants.BPM_INTERVAL
    if bpm_meta == 0:
        bpm_meta = 1
    return bpm_meta


@register_encoder
@add_offset
@encode_unknown()
def encode_audio_key(audio_key: str, encoding_map: Dict[str, int]) -> int:
    try:
        return encoding_map[audio_key]
    except KeyError:
        raise UnprocessableMidiError(f"audio key KeyError: {audio_key}")


@register_encoder
@add_offset
@encode_unknown()
def encode_time_signature(time_signature: str, encoding_map: Dict[str, int]) -> int:
    try:
        return encoding_map[time_signature]
    except KeyError:
        raise UnprocessableMidiError(f"ts KeyError: {time_signature}")


@register_encoder
@add_offset
@encode_unknown()
def encode_pitch_range(pitch_range: str, encoding_map: Dict[str, int]) -> int:
    try:
        return encoding_map[pitch_range]
    except KeyError:
        raise UnprocessableMidiError(f"pitch range KeyError: {pitch_range}")


@register_encoder
@encode_unknown(raise_error=True)
def encode_num_measures(num_measures: Union[float, str]) -> int:
    num_measures = math.floor(num_measures)
    if num_measures == 4:
        return Offset.MEASURES_4.value
    elif num_measures == 5:
        return Offset.MEASURES_4.value
    elif num_measures == 8:
        return Offset.MEASURES_8.value
    elif num_measures == 9:
        return Offset.MEASURES_8.value
    elif num_measures == 16:
        return Offset.MEASURES_16.value
    elif num_measures == 17:
        return Offset.MEASURES_16.value
    else:
        raise UnprocessableMidiError(f"num measures ValueError: {num_measures}")


@register_encoder
@add_offset
@encode_unknown()
def encode_inst(inst: Union[int, str], encoding_map: Dict[str, int]) -> int:
    try:
        return encoding_map[inst]
    except KeyError:
        raise UnprocessableMidiError(f"inst KeyError: {inst}")


@register_encoder
@add_offset
@encode_unknown()
def encode_genre(genre: str, encoding_map: Dict[str, int]) -> int:
    try:
        return encoding_map[genre]
    except KeyError:
        raise UnprocessableMidiError(f"genre KeyError: {genre}")


@register_encoder
@add_offset
@encode_unknown()
def encode_min_velocity(velocity: Union[int, str]):
    return math.floor(velocity / constants.VELOCITY_INTERVAL)


@register_encoder
@add_offset
@encode_unknown()
def encode_max_velocity(velocity: Union[int, str]):
    return math.ceil(velocity / constants.VELOCITY_INTERVAL)


@register_encoder
@add_offset
@encode_unknown()
def encode_track_role(track_role: str, encoding_map: Dict[str, int]) -> int:
    try:
        return encoding_map[track_role]
    except KeyError:
        raise UnprocessableMidiError(f"track role KeyError: {track_role}")


@register_encoder
@add_offset
@encode_unknown()
def encode_rhythm(rhythm: str, encoding_map: Dict[str, int]) -> int:
    try:
        return encoding_map[rhythm]
    except KeyError:
        raise UnprocessableMidiError(f"rhythm KeyError: {rhythm}")


def encode_meta(
    midi_meta: MidiMeta,
) -> List[int]:
    encoding_maps = DEFAULT_ENCODING_MAPS
    result = []
    for meta_name in META_ENCODING_ORDER:
        encoded_meta = inject_args_to_encode_func(
            ENCODERS[meta_name],
            getattr(midi_meta, meta_name),
            encoding_maps.get(meta_name),
        )
        result.append(encoded_meta)
    return result


class MetaEncoder:
    def __init__(self):
        pass

    def encode(self, midi_meta: MidiMeta) -> List[int]:
        return encode_meta(midi_meta)
