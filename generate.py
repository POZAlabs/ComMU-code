import argparse
import copy
import logging
import math
from pathlib import Path
from typing import Any, List, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from commu.preprocessor.encoder import MetaEncoder, EventSequenceEncoder, encoder_utils
from commu.preprocessor.encoder.event_tokens import TOKEN_OFFSET
from commu.preprocessor.utils import sync_key_augment
from commu.preprocessor.utils import constants
from commu.preprocessor.utils.container import MidiInfo, MidiMeta
from commu.model.config_helper import get_default_cfg_inference, get_default_cfg_training
from commu.model.dataset import BaseVocab
from commu.model.model import MemTransformerLM


def _map_chord2token(chord_progression: List, numerator: int, denominator: int) -> Tuple:
    event2word, _ = encoder_utils.mk_remi_map()
    event2word = encoder_utils.add_flat_chord2map(event2word)
    event2word = encoder_utils.abstract_chord_types(event2word)

    norm_chord = sync_key_augment(chord_progression, "c", "c")
    beats_per_bar = int(numerator / denominator * 4)
    chord_idx_lst, unique_cp = encoder_utils.detect_chord(norm_chord[0], beats_per_bar)
    chord_position = []
    for i in chord_idx_lst:
        if isinstance(i, int):
            chord_position.append(
                TOKEN_OFFSET.POSITION.value
            )
        else:
            bit_offset = (float(str(i).split(".")[-1]) * constants.DEFAULT_POSITION_RESOLUTION) / (
                    10 ** len(str(i).split(".")[-1])
            )
            chord_position.append(int(
                TOKEN_OFFSET.POSITION.value
                + bit_offset)
            )

    chord_token = []
    for chord in unique_cp:
        chord = chord.lower()
        chord = "Chord_" + chord.split("/")[0].split("(")[0]
        chord_token.append(event2word[chord])
    return chord_token, chord_position


def parse_meta(**kwargs: Any) -> MidiMeta:
    return MidiMeta(**kwargs)


def encode_meta(meta_encoder: MetaEncoder, midi_meta: MidiMeta) -> List[int]:
    return meta_encoder.encode(midi_meta)


def decode_event_sequence(
        generation_result: List[int],
        num_meta: int,
        meta,
        output_dir: Union[str, Path],
        index: int,
        args: argparse.Namespace,
):
    track_role = args.track_role
    event_sequence = generation_result[num_meta + 1:]
    inst = args.inst
    pitch_range = args.pitch_range
    output_dir = Path(output_dir).joinpath(
            f"{track_role}_{inst}_{pitch_range}")
    output_dir.mkdir(exist_ok=True, parents=True)
    decoder = EventSequenceEncoder()
    decoder.decode(
        output_path=Path(output_dir).joinpath(
            f"{track_role}_{inst}_{pitch_range}_{index:03d}.mid"),
        midi_info=MidiInfo(*meta, event_seq=event_sequence)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="ComMU Transformer Inference")

    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    # Input meta
    parser.add_argument("--bpm", type=int)
    parser.add_argument("--audio_key", type=str, choices=list(constants.KEY_MAP.keys()))
    parser.add_argument("--time_signature", type=str, choices=list(constants.TIME_SIG_MAP.keys()))
    parser.add_argument("--pitch_range", type=str, choices=list(constants.PITCH_RANGE_MAP.keys()))
    parser.add_argument("--num_measures", type=float)
    parser.add_argument(
        "--inst", type=str, choices=list(constants.INST_MAP.keys()),
    )
    parser.add_argument(
        "--genre", type=str, default="cinematic", choices=list(constants.GENRE_MAP.keys())
    )
    parser.add_argument("--min_velocity", type=int, choices=range(1, 128))
    parser.add_argument("--max_velocity", type=int, choices=range(1, 128))
    parser.add_argument(
        "--track_role", type=str, choices=list(constants.TRACK_ROLE_MAP.keys())
    )
    parser.add_argument(
        "--rhythm", type=str, default="standard", choices=list(constants.RHYTHM_MAP.keys())
    )

    parser.add_argument(
        "--chord_progression", type=str, help='Chord progression ex) C-C-E-E-G-G ...'
    )

    # Sampling
    parser.add_argument("--num_generate", type=int, help="생성 개수")

    return parser


def main(inference_cfg, args: argparse.Namespace):
    output_dir = Path(args.output_dir).expanduser()
    checkpoint_dir = Path(args.checkpoint_dir).expanduser()
    chord_progression = args.chord_progression.split("-")

    topk = int(inference_cfg.SAMPLING.threshold)
    temperature = inference_cfg.SAMPLING.temperature

    device = torch.device("cuda" if inference_cfg.MODEL.device else "cpu")
    perform_vocab = BaseVocab()

    cfg = get_default_cfg_training()
    cfg.defrost()
    cfg.MODEL.same_length = True  # Needed for same_length =True during evaluation
    cfg.freeze()

    model = MemTransformerLM(cfg, perform_vocab)
    checkpoint = torch.load(checkpoint_dir)

    model.load_state_dict(checkpoint["model"], strict=False)
    model = model.to(device)
    model.eval()
    model.reset_length(1, inference_cfg.MODEL.memory_length)

    midi_meta = parse_meta(**vars(args), chord_prgression=chord_progression)
    logging.info(f"Generating {args.num_generate} samples using following meta:\n{midi_meta.dict()}")

    meta_encoder = MetaEncoder()
    encoded_meta = encode_meta(
        meta_encoder=meta_encoder, midi_meta=midi_meta
    )
    logging.info("Encoded meta")

    num_conditional_tokens = len(encoded_meta)
    logging.info('* Total number of tokens used for condition is {}'.format(len(encoded_meta)))

    idx = 0
    while idx != args.num_generate:
        seq = [0]
        mems = None
        with torch.no_grad():
            print("Generating the idx: " + str(idx + 1))
            context = np.array(seq + encoded_meta[:num_conditional_tokens - 1], dtype=np.int32)[:, np.newaxis]
            context = torch.from_numpy(context).to(device).type(torch.long)
            ret = model.forward_generate(context, mems)
            _, mems = ret
            seq = seq + encoded_meta[:num_conditional_tokens]
            generation_length = inference_cfg.GENERATION.generation_length

            num_measures = args.num_measures
            numerator, denominator = args.time_signature.split("/")
            chord_token, chord_position = _map_chord2token(
                chord_progression,
                int(numerator),
                int(denominator)
            )

            check_chord_progression = copy.deepcopy(chord_token)
            check_chord_position = copy.deepcopy(chord_position)
            check_inter_chord = []
            for i in chord_position:
                if i == TOKEN_OFFSET.POSITION.value:
                    check_inter_chord.append(0)
                else:
                    check_inter_chord.append(1)

            if (len(check_chord_position) != len(check_chord_progression)) or (
                    len(check_inter_chord) != len(check_chord_progression)
            ):
                raise ValueError("chord progression and position information not alligned")

            is_incomplete = num_measures % 4 != 0
            incomplete_filled = False if is_incomplete else True

            for _ in range(generation_length):
                if seq[-1] == 1:
                    break
                # Create input array
                inp = np.array([seq[-1]], dtype=np.int32)[:, np.newaxis]
                inp = torch.from_numpy(inp).to(device).type(torch.long)

                ret = model.forward_generate(inp, mems)

                all_logits, mems = ret
                # Select last timestep, only batch item
                logits = all_logits[-1, 0]

                logits = logits[1:]

                # Handle temp 0 (argmax) case
                if temperature == 0:
                    probs = torch.zeros_like(logits)
                    probs[logits.argmax()] = 1.0
                else:
                    # Apply temperature spec
                    logits /= temperature

                    # Compute softmax
                    probs = F.softmax(logits, dim=-1)

                # Exclude bos token
                probs = F.pad(probs, [1, 0])


                _, top_idx = torch.topk(probs, topk)
                mask = torch.zeros_like(probs)
                mask[top_idx] = 1.0
                probs *= mask
                probs /= probs.sum()

                # chord progression teacher forcing
                if not incomplete_filled:
                    incomplete_filled = True if seq.count(TOKEN_OFFSET.BAR.value) > 1 else False

                # position 1/128 right after the bar token
                if incomplete_filled:
                    if seq[-1] == TOKEN_OFFSET.BAR.value:
                        nex_token = int(TOKEN_OFFSET.POSITION.value)
                        seq.append(nex_token)
                        continue


                if chord_position != [] and chord_token != []:
                    if incomplete_filled:
                        # one chord per bar
                        if len(check_chord_progression) == int(num_measures//4 * 4):
                            if (

                                    seq[-2] == TOKEN_OFFSET.BAR.value
                                    and seq[-1] == TOKEN_OFFSET.POSITION.value
                            ):
                                next_tokens = int(chord_token.pop(0))
                                seq.append(next_tokens)
                                continue
                        # in case the chord progression change within a bar
                        else:
                            if (
                                    seq[-2] == TOKEN_OFFSET.BAR.value
                                    and seq[-1] == TOKEN_OFFSET.POSITION.value
                            ):
                                next_tokens = int(chord_token.pop(0))
                                seq.append(next_tokens)
                                chord_position.pop(0)
                                check_inter_chord.pop(0)
                                continue

                            elif seq[-1] == chord_position[0]:
                                if check_inter_chord[0] == 1:
                                    next_tokens = int(chord_token.pop(0))
                                    seq.append(next_tokens)
                                    chord_position.pop(0)
                                    check_inter_chord.pop(0)
                                    continue

                    token = torch.multinomial(probs, 1)
                    token = int(token.item())
                    # forced assignment of the skipped chord positionn
                    if check_inter_chord[0] == 1:
                        if (chord_position[0] < token
                                < TOKEN_OFFSET.POSITION.value + constants.DEFAULT_POSITION_RESOLUTION):
                            token = chord_position[0]
                            seq.append(token)
                            continue
                        elif token == TOKEN_OFFSET.BAR.value:
                            token = chord_position[0]
                            seq.append(token)
                            continue
                    # ignore generated chord token
                    if TOKEN_OFFSET.CHORD_START.value <= token <= TOKEN_OFFSET.CHORD_END.value:
                        continue
                    # ignore eos token
                    elif token == TOKEN_OFFSET.EOS.value:
                        token = chord_position[0] if check_inter_chord[0] else TOKEN_OFFSET.BAR.value
                        seq.append(token)
                        continue
                    else:
                        seq.append(token)
                else:
                    token = torch.multinomial(probs, 1)
                    token = int(token.item())
                    if TOKEN_OFFSET.CHORD_START.value <= token <= TOKEN_OFFSET.CHORD_END.value:
                        continue
                    # ignore bar token
                    if token == TOKEN_OFFSET.BAR.value:
                        token = TOKEN_OFFSET.EOS.value
                        seq.append(token)
                    else:
                        seq.append(token)
            num_bar = 0
            for i in seq:
                if i == TOKEN_OFFSET.BAR.value:
                    num_bar += 1
            num_chord = 0
            for i in seq:
                if TOKEN_OFFSET.CHORD_START.value <= i <= TOKEN_OFFSET.CHORD_END.value:
                    num_chord += 1
            if len(chord_token) != 0:
                print("length:", len(chord_token))
                print("error in teacher forcing")
            elif num_bar != int(math.ceil(num_measures)):
                print("length:", num_bar)
                print("error in bar length")
            elif num_chord not in [len(check_chord_progression), len(check_chord_progression) * 2]:
                print("num_chord:", num_chord, f"vs {len(check_chord_progression)}")
                print("error in chord length")
            else:
                print("correct_length:", num_bar)
                decode_event_sequence(
                    generation_result=seq,
                    num_meta=len(encoded_meta),
                    meta=encoded_meta,
                    output_dir=output_dir,
                    index=idx,
                    args=args,
                )
                idx += 1


if __name__ == "__main__":
    known_args, _ = parse_args().parse_known_args()
    inference_cfg = get_default_cfg_inference()
    inference_cfg.freeze()
    # Sanity check to make sure the config is indeed right
    print(inference_cfg)
    main(inference_cfg, known_args)
