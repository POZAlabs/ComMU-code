import copy
import enum
import os
import shutil
import tempfile
from ast import literal_eval
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import miditoolkit
import numpy as np
import pandas as pd
import parmap

from . import augment
from .utils import sync_key_augment
from .utils.exceptions import UnprocessableMidiError
from .encoder import MetaEncoder, EventSequenceEncoder
from .parser import MetaParser

MIDI_EXTENSIONS = (".mid", ".MID", ".midi", ".MIDI")


class OutputSubDirName(str, enum.Enum):
    RAW = "raw"
    ENCODE_NPY = "output_npy"
    ENCODE_TMP = "npy_tmp"


class SubDirName(str, enum.Enum):
    RAW = "raw"
    ENCODE_NPY = "output_npy"
    ENCODE_TMP = "npy_tmp"
    AUGMENTED_TMP = "augmented_tmp"
    AUGMENTED = "augmented"


@dataclass
class OutputSubDirectory:
    encode_npy: Union[str, Path]
    encode_tmp: Union[str, Path]


@dataclass
class SubDirectory:
    raw: Union[str, Path]
    encode_npy: Union[str, Path]
    encode_tmp: Union[str, Path]
    augmented_tmp: Optional[Union[str, Path]] = field(default=None)
    augmented: Optional[Union[str, Path]] = field(default=None)


def get_output_sub_dir(root_dir: Union[str, Path]) -> OutputSubDirectory:
    result = dict()
    for name, member in OutputSubDirName.__members__.items():
        output_dir = root_dir.joinpath(member.value)
        output_dir.mkdir(exist_ok=True, parents=True)
        result[name.lower()] = output_dir
    return OutputSubDirectory(**result)


def get_sub_dir(
        root_dir: Union[str, Path], split: Optional[str]) -> SubDirectory:
    result = dict()
    for name, member in SubDirName.__members__.items():
        if split is None:
            sub_dir = root_dir.joinpath(member.value)
        else:
            sub_dir = root_dir.joinpath(split).joinpath(member.value)
        sub_dir.mkdir(exist_ok=True, parents=True)
        result[name.lower()] = sub_dir
    return SubDirectory(**result)


@dataclass
class EncodingOutput:
    meta: np.ndarray
    event_sequence: np.ndarray


class Preprocessor:
    def __init__(
            self,
            meta_parser: MetaParser,
            meta_encoder: MetaEncoder,
            event_sequence_encoder: EventSequenceEncoder,
            csv_path: str,
    ):
        self.meta_parser = meta_parser
        self.meta_encoder = meta_encoder
        self.event_sequence_encoder = event_sequence_encoder
        self.csv_path = csv_path

    def augment_data(
            self,
            source_dir: Union[str, Path],
            augmented_dir: Union[str, Path],
            augmented_tmp_dir: Union[str, Path],
            num_cores: int,
    ):
        augment.augment_data(
            midi_path=str(source_dir),
            augmented_dir=str(augmented_dir),
            augmented_tmp_dir=str(augmented_tmp_dir),
            num_cores=num_cores,
        )

    def encode_event_sequence(self, midi_path: Union[str, Path], sample_info: Dict) -> np.ndarray:
        with tempfile.NamedTemporaryFile(suffix=Path(midi_path).suffix) as f:
            midi_obj = miditoolkit.MidiFile(midi_path)
            for idx in range(len(midi_obj.instruments)):
                try:
                    if midi_obj.instruments[idx].name == "chord":
                        midi_obj.instruments.pop(idx)
                except IndexError:
                    continue
            midi_obj.dump(f.name)
            event_sequence = np.array(self.event_sequence_encoder.encode(midi_path, sample_info=sample_info))
            return event_sequence

    def preprocess(
            self,
            root_dir: Union[str, Path],
            num_cores: int,
            data_split: Tuple[str] = ("train", "val",),
    ):
        default_sub_dir = get_sub_dir(root_dir, split=None)
        fetched_samples = pd.read_csv(self.csv_path,
                                      converters={"chord_progressions": literal_eval})

        for empty_dir in fields(default_sub_dir):
            if empty_dir.name in ("encode_npy",):
                continue
            else:
                try:
                    os.rmdir(root_dir.joinpath(getattr(SubDirName, empty_dir.name.upper()).value))
                except FileNotFoundError:
                    continue

        for split in data_split:
            split_sub_dir = get_sub_dir(root_dir, split=split)
            self.augment_data(
                source_dir=split_sub_dir.raw,
                augmented_dir=split_sub_dir.augmented,
                augmented_tmp_dir=split_sub_dir.augmented_tmp,
                num_cores=num_cores,
            )

            sample_id_to_path = self._gather_sample_files(
                *(split_sub_dir.raw, split_sub_dir.augmented))

            self.export_encoded_midi(
                fetched_samples=fetched_samples,
                encoded_tmp_dir=split_sub_dir.encode_tmp,
                sample_id_to_path=sample_id_to_path,
                num_cores=num_cores,
            )

            input_npy, target_npy = self.concat_npy(split_sub_dir.encode_tmp)
            np.save(str(default_sub_dir.encode_npy.joinpath(f"input_{split}.npy")), input_npy, allow_pickle=True)
            np.save(str(default_sub_dir.encode_npy.joinpath(f"target_{split}.npy")), target_npy, allow_pickle=True)

            for empty_dir in os.listdir(root_dir.joinpath(split)):
                if empty_dir in ("raw", "npy_tmp", "augmented", "augmented_tmp"):
                    continue
                else:
                    shutil.rmtree(root_dir.joinpath(split).joinpath(empty_dir))

    def export_encoded_midi(
            self,
            fetched_samples: Union[pd.DataFrame, List[Dict[str, Any]]],
            sample_id_to_path: Dict[str, str],
            encoded_tmp_dir: Union[str, Path],
            num_cores: int,
    ) -> None:
        sample_infos_chunk = [
            (idx, arr.tolist())
            for idx, arr in enumerate(np.array_split(np.array(fetched_samples.to_dict('records')), num_cores))
        ]
        parmap.map(
            self._preprocess_midi_chunk,
            sample_infos_chunk,
            sample_id_to_path=sample_id_to_path,
            encode_tmp_dir=encoded_tmp_dir,
            pm_pbar=True,
            pm_processes=num_cores,
        )

    def _preprocess_midi_chunk(
            self,
            idx_sample_infos_chunk: Tuple[int, Iterable[Dict[str, Any]]],
            sample_id_to_path: Dict[str, str],
            encode_tmp_dir: Union[str, Path],
    ):
        idx, sample_infos_chunk = idx_sample_infos_chunk
        copied_sample_infos_chunk = copy.deepcopy(list(sample_infos_chunk))
        parent_sample_ids_to_info = {
            sample_info["id"]: sample_info for sample_info in copied_sample_infos_chunk
        }
        parent_sample_ids = set(parent_sample_ids_to_info.keys())

        copied_sample_infos_chunk.extend(
            [
                {"id": sample_id, "augmented": True}
                for sample_id in sample_id_to_path.keys()
                if sample_id.split("_")[0] in parent_sample_ids
            ]
        )

        encode_tmp_dir = Path(encode_tmp_dir)
        for sample_info_idx, sample_info in enumerate(copied_sample_infos_chunk):
            copied_sample_info = sample_info
            if sample_info.get("augmented", False):
                id_split = copied_sample_info["id"].split("_")
                bpm = copied_sample_info.get("bpm")
                audio_key = copied_sample_info.get("audio_key")
                if len(id_split) > 1:
                    parent_sample_id, audio_key, bpm = id_split
                else:
                    parent_sample_id = id_split[0]

                if bpm is None or audio_key is None:
                    continue

                augmented_midi_path = sample_id_to_path[copied_sample_info["id"]]
                copied_sample_info = copy.deepcopy(parent_sample_ids_to_info[parent_sample_id])
                copied_sample_info["bpm"] = int(bpm)
                # key_origin = copied_sample_info["audio_key"] + copied_sample_info["chord_type"] in ["cmajor", "aminor"]
                # key_origin 값 수정
                key_origin = copied_sample_info["audio_key"] in ["cmajor", "aminor"]

                if not key_origin:
                    continue
                try:
                    copied_sample_info["chord_progressions"] = sync_key_augment(
                        copied_sample_info["chord_progressions"][0],
                        audio_key.replace("minor", "").replace("major", ""),
                        copied_sample_info["audio_key"][0], # audio_key 값 앞쪽으로 할당
                    )
                except IndexError:
                    print(f"chord progression info is unknown: {augmented_midi_path}")
                    continue
                copied_sample_info["audio_key"] = audio_key
                copied_sample_info["rhythm"] = copied_sample_info.get("sample_rhythm")
                # is_incomplete_measure column 추가
                if copied_sample_info["num_measures"]%4==0:
                    copied_sample_info["is_incomplete_measure"] = False
                else:
                    copied_sample_info["is_incomplete_measure"] = True

                midi_path = sample_id_to_path.get(copied_sample_info["id"])
                if midi_path is None:
                    continue
                try:
                    encoding_output = self._preprocess_midi(
                        sample_info=copied_sample_info, midi_path=augmented_midi_path
                    )
                except (IndexError, TypeError) as e:
                    print(f"{e}: {augmented_midi_path}")
                    continue
                except ValueError:
                    print(f"num measures not allowed: {augmented_midi_path}")
                    continue
                output_dir = encode_tmp_dir.joinpath(f"{idx:04d}")
                output_dir.mkdir(exist_ok=True, parents=True)
                try:
                    np.save(
                        os.path.join(output_dir, f"input_{sample_info_idx}"), encoding_output.meta
                    )
                    np.save(
                        os.path.join(output_dir, f"target_{sample_info_idx}"), encoding_output.event_sequence
                    )
                except AttributeError:
                    continue

    def _preprocess_midi(
            self, sample_info: Dict[str, Any], midi_path: Union[str, Path]
    ) -> Optional[EncodingOutput]:
        midi_meta = self.meta_parser.parse(meta_dict=sample_info)
        try:
            encoded_meta: List[Union[int, str]] = self.meta_encoder.encode(midi_meta)
        except UnprocessableMidiError as e:
            print(f"{e}: {midi_path}")
            return None
        encoded_meta: np.ndarray = np.array(encoded_meta, dtype=object)
        encoded_event_sequence = np.array(
            self.encode_event_sequence(midi_path, sample_info), dtype=np.int16
        )
        return EncodingOutput(meta=encoded_meta, event_sequence=encoded_event_sequence)

    @staticmethod
    def _gather_sample_files(*source_dirs: Union[str, Path]) -> Dict[str, str]:
        def _gather(_source_dir):
            return {
                filename.stem: str(filename)
                for filename in Path(_source_dir).rglob("**/*")
                if filename.suffix in MIDI_EXTENSIONS
            }

        result = dict()
        for source_dir in source_dirs:
            result.update(_gather(source_dir))
        return result

    @staticmethod
    def concat_npy(source_dir: Union[str, Path]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        def _gather(_prefix) -> List[str]:
            npy_suffix = ".npy"
            return sorted(
                str(f)
                for f in Path(source_dir).rglob("**/*")
                if f.suffix == npy_suffix and f.stem.startswith(_prefix)
            )

        def _concat(_npy_list: List[str]) -> List[np.ndarray]:
            return [np.load(_p, allow_pickle=True) for _p in _npy_list]

        return _concat(_gather("input")), _concat(_gather("target"))
