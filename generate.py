import argparse
from typing import Dict

from commu.midi_generator.generate_pipeline import MidiGenerationPipeline
from commu.preprocessor.utils import constants


def parse_args() -> Dict[str, argparse.ArgumentParser]:
    model_arg_parser = argparse.ArgumentParser(description="Model Arguments")
    input_arg_parser = argparse.ArgumentParser(description="Input Arguments")

    # Model Arguments
    model_arg_parser.add_argument("--checkpoint_dir", type=str)

    # Input Arguments
    input_arg_parser.add_argument("--output_dir", type=str, required=True)

    ## Input meta
    input_arg_parser.add_argument("--bpm", type=int)
    input_arg_parser.add_argument("--audio_key", type=str, choices=list(constants.KEY_MAP.keys()))
    input_arg_parser.add_argument("--time_signature", type=str, choices=list(constants.TIME_SIG_MAP.keys()))
    input_arg_parser.add_argument("--pitch_range", type=str, choices=list(constants.PITCH_RANGE_MAP.keys()))
    input_arg_parser.add_argument("--num_measures", type=float)
    input_arg_parser.add_argument(
        "--inst", type=str, choices=list(constants.INST_MAP.keys()),
    )
    input_arg_parser.add_argument(
        "--genre", type=str, default="cinematic", choices=list(constants.GENRE_MAP.keys())
    )
    input_arg_parser.add_argument(
        "--track_role", type=str, choices=list(constants.TRACK_ROLE_MAP.keys())
    )
    input_arg_parser.add_argument(
        "--rhythm", type=str, default="standard", choices=list(constants.RHYTHM_MAP.keys())
    )
    input_arg_parser.add_argument("--min_velocity", type=int, choices=range(1, 128))
    input_arg_parser.add_argument("--max_velocity", type=int, choices=range(1, 128))
    input_arg_parser.add_argument(
        "--chord_progression", type=str, help='Chord progression ex) C-C-E-E-G-G ...'
    )
    # Inference 시 필요 정보
    input_arg_parser.add_argument("--num_generate", type=int)
    input_arg_parser.add_argument("--top_k", type=int, default=32)
    input_arg_parser.add_argument("--temperature", type=float, default=0.95)

    arg_dict = {
        "model_args": model_arg_parser,
        "input_args": input_arg_parser
    }
    return arg_dict


def main(model_args: argparse.Namespace, input_args: argparse.Namespace):
    pipeline = MidiGenerationPipeline(vars(model_args))

    inference_cfg = pipeline.model_initialize_task.inference_cfg
    model = pipeline.model_initialize_task.execute()

    encoded_meta = pipeline.preprocess_task.excecute(vars(input_args))
    input_data = pipeline.preprocess_task.input_data
    meta_info_len = pipeline.preprocess_task.get_meta_info_length()

    pipeline.inference_task(
        model=model,
        input_data=input_data,
        inference_cfg=inference_cfg
    )
    sequences = pipeline.inference_task.execute(encoded_meta)

    pipeline.postprocess_task(input_data=input_data)
    pipeline.postprocess_task.execute(
        sequences=sequences,
        meta_info_len=meta_info_len
    )


if __name__ == "__main__":
    model_args, _ = parse_args()["model_args"].parse_known_args()
    input_args, _ = parse_args()["input_args"].parse_known_args()
    main(model_args, input_args)