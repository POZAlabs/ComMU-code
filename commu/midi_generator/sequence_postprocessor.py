from pathlib import Path
from typing import List

from miditoolkit import MidiFile

from commu.midi_generator.container import TransXlInputData
from commu.preprocessor.encoder import EventSequenceEncoder
from commu.preprocessor.utils.container import MidiInfo


class PostprocessTask:
    def __init__(self):
        pass

    def __call__(self, input_data: TransXlInputData):
        self.input_data = input_data

    def get_output_dir(self) -> Path:
        return self.input_data.output_dir

    def set_output_file_path(self, index: int) -> Path:
        track_role = self.input_data.track_role
        inst = self.input_data.inst
        pitch_range = self.input_data.pitch_range

        output_dir = Path(self.input_data.output_dir).joinpath(
            f"{track_role}_{inst}_{pitch_range}")
        output_dir.mkdir(exist_ok=True, parents=True)

        file_name = f"{track_role}_{inst}_{pitch_range}_{index:03d}.mid"

        return output_dir.joinpath(file_name)

    def decode_event_sequence(
            self,
            generation_result: List[int],
            num_meta: int
    ) -> MidiFile:
        encoded_meta = generation_result[1: num_meta + 1]
        event_sequence = generation_result[num_meta + 2:]
        decoder = EventSequenceEncoder()
        decoded_midi = decoder.decode(
            midi_info=MidiInfo(*encoded_meta, event_seq=event_sequence),
        )

        return decoded_midi

    def execute(self, sequences: List[List[int]], meta_info_len: int) -> Path:
        for idx, seq in enumerate(sequences):
            decoded_midi = self.decode_event_sequence(
                generation_result=seq,
                num_meta=meta_info_len,
            )
            output_file_path = self.set_output_file_path(idx)
            decoded_midi.dump(output_file_path)

        return self.get_output_dir()
