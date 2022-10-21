import math
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import yacs.config

from logger import logger
from commu.midi_generator.container import TransXlInputData
from commu.model.model import MemTransformerLM
from commu.preprocessor.encoder import TOKEN_OFFSET
from commu.preprocessor.utils.constants import DEFAULT_POSITION_RESOLUTION


class TeacherForceTask:
    def __init__(self, input_data):
        self.input_data = input_data
        self.next_tokens_forced = []
        self.wrong_tokens = []
        self.no_sequence_appended = False
        self.is_incomplete = input_data.num_measures % 4 != 0
        self.incomplete_filled = not self.is_incomplete

        self.chord_token, self.chord_position = input_data.chord_token_components.values()
        assert len(self.chord_token) == len(self.chord_position), "Wrong Chord Length"
        self.chord_length = len(self.chord_token)
        self.inter_chord_flags = []
        for i in self.chord_position:
            if i == TOKEN_OFFSET.POSITION.value:
                self.inter_chord_flags.append(False)
            else:
                self.inter_chord_flags.append(True)

    def check_first_position(self, seq):
        """
        못갖춘 마디가 아닌 상태에서 bar token 이후 토큰인지 확인
        """
        return self.incomplete_filled and seq[-1] == TOKEN_OFFSET.BAR.value

    def check_remnant_chord(self):
        """
        teacher force 필요한 코드 소진 하였는지 확인, 다 소진 하였다면 False
        """
        return bool(len(self.chord_token) * len(self.chord_position))

    def check_length_fit(self):
        """
        한마디에 코드 하나만 들어가는지 확인, chord 개수 == bar 개수 일치 여부
        """
        return self.chord_length == int(self.input_data.num_measures // 4 * 4)

    def check_position_fit(self, seq):
        """
        코드 정보가 들어올 차례인지 즉 bar token, position 1/128 token 이 직전 token 인지 확인
        """
        return seq[-2] == TOKEN_OFFSET.BAR.value and seq[-1] == TOKEN_OFFSET.POSITION.value

    def check_one_chord_per_bar_case(self, seq):
        """
        한마디에 코드 하나만 들어가는 case 확인
        """
        return (
            self.check_remnant_chord()
            and self.incomplete_filled
            and self.check_length_fit()
            and self.check_position_fit(seq)
        )

    def check_mul_chord_per_bar_case(self, seq):
        """
        마디 중간에 코드 있는 case 확인
        is_first_position_chord: bar token 이후 삽입될 코드인 case
        is_inter_position_chord: 마디 중간에 삽입될 코드인 case
        """
        is_first_position_chord = (
            self.check_remnant_chord()
            and self.incomplete_filled
            and not self.check_length_fit()
            and self.check_position_fit(seq)
        )

        is_inter_position_chord = (
            self.check_remnant_chord()
            and self.incomplete_filled
            and not self.check_length_fit()
            and not self.check_position_fit(seq)
            and seq[-1] == self.chord_position[0]
            and self.inter_chord_flags[0]
        )
        return is_first_position_chord or is_inter_position_chord

    def check_chord_position_passed(self, token):
        """
        다음 코드는 마디 중간에 나타날 코드이고
        position이 지나쳐버리거나 다음 마디가 나오면 True
        """
        if not self.check_remnant_chord():
            return False
        is_position_passed = (
            self.chord_position[0] < token < TOKEN_OFFSET.POSITION.value + DEFAULT_POSITION_RESOLUTION
            or token == TOKEN_OFFSET.BAR.value
        )
        return self.inter_chord_flags[0] and is_position_passed

    def check_wrong_chord_token_generated(self, token):
        """
        코드 token은 전부 티쳐포싱을 통해서만 추가되어야함
        """
        return TOKEN_OFFSET.CHORD_START.value <= token <= TOKEN_OFFSET.CHORD_END.value

    def check_wrong_eos_generated(self, token):
        return self.check_remnant_chord() and token == TOKEN_OFFSET.EOS.value

    def check_wrong_bar_token_generated(self, token):
        return not self.check_remnant_chord() and token == TOKEN_OFFSET.BAR.value

    def teach_first_position(self) -> None:
        """
        마디 시작 마다 코드 추가를 강제하므로 1/128 position을 teacher force
        """
        self.next_tokens_forced.append(int(TOKEN_OFFSET.POSITION.value))

    def teach_chord_token(self):
        """
        코드 토큰 강제할당
        강제할당할 토큰을 저장해둔 chord_token에서 가져와 next_torken_forced에 넣음
        강제할당 완료 예정이므로 wrong_tokens 초기화
        """
        next_chord_tokens = self.chord_token.pop(0)
        self.next_tokens_forced.append(next_chord_tokens)
        self.chord_position.pop(0)
        self.inter_chord_flags.pop(0)
        self.wrong_tokens = []

    def teach_chord_position(self):
        next_position_token = self.chord_position[0]
        self.next_tokens_forced.append(next_position_token)
        self.wrong_tokens = []

    def teach_wrong_chord_token(self, wrong_token):
        self.no_sequence_appended = True
        self.wrong_tokens.append(wrong_token)

    def teach_remnant_chord(self):
        token = self.chord_position[0] if self.inter_chord_flags[0] else TOKEN_OFFSET.BAR.value
        self.next_tokens_forced.append(token)

    def teach_eos(self):
        token = TOKEN_OFFSET.EOS.value
        self.next_tokens_forced.append(token)

    def validate_teacher_forced_sequence(self, seq) -> None:
        def _count_num_chord(seq):
            chord_counter = 0
            for token in seq:
                if TOKEN_OFFSET.CHORD_START.value <= token <= TOKEN_OFFSET.CHORD_END.value:
                    chord_counter += 1
            return chord_counter

        num_bars = seq.count(TOKEN_OFFSET.BAR.value)
        num_chord = _count_num_chord(seq)

        if len(self.chord_token) != 0:
            raise Exception(
                f"remnant chord length: {len(self.chord_token)} \n" "error in teacher forcing"
            )
        elif num_bars != int(math.ceil(self.input_data.num_measures)):
            raise Exception(f"bar length: {num_bars} \n" "error in bar length")
        elif num_chord != self.chord_length:
            raise Exception(
                f"num_chord: {num_chord} vs {self.chord_length} \n" "error in chord length"
            )
        else:
            logger.info(f"correct_length: {num_bars}")
            logger.info(seq)


class InferenceTask:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(
        self,
        model: MemTransformerLM,
        input_data: TransXlInputData,
        inference_cfg: yacs.config.CfgNode,
    ):
        self.model = model
        self.input_data = input_data
        self.inference_cfg = inference_cfg

    def init_seq_and_mems(
        self, encoded_meta: List[int], num_conditional_tokens: int
    ) -> Tuple[List[int], torch.Tensor]:

        seq = [0]
        ctx = np.array(seq + encoded_meta[: num_conditional_tokens - 1], dtype=np.int32)[
            :, np.newaxis
        ]
        context = torch.from_numpy(ctx).to(self.device).type(torch.long)
        _, init_mems = self.model.forward_generate(context, mems=None)
        init_seq = seq + encoded_meta[:num_conditional_tokens]
        return init_seq, init_mems

    def calc_logits_and_mems(
        self, seq: List[int], mems: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inp = np.array([seq[-1]], dtype=np.int32)[:, np.newaxis]
        input_token = torch.from_numpy(inp).to(self.device).type(torch.long)
        ret = self.model.forward_generate(input_token, mems)
        all_logits, mems = ret
        logits = all_logits[-1, 0][1:]
        return logits, mems

    def calc_probs(self, logits):
        # Handle temp 0 (argmax) case
        if self.input_data.temperature == 0:
            probs = torch.zeros_like(logits)
            probs[logits.argmax()] = 1.0
        else:
            # Apply temperature spec
            logits /= self.input_data.temperature
            # Compute softmax
            probs = F.softmax(logits, dim=-1)

        probs = F.pad(probs, [1, 0])
        return probs

    def apply_sampling(self, probs, wrong_tokens):
        _, top_idx = torch.topk(probs, self.input_data.top_k)
        mask = torch.zeros_like(probs)
        mask[top_idx] = 1.0
        if wrong_tokens:
            for w in wrong_tokens:
                mask[w] = 0.0
        probs *= mask
        probs /= torch.sum(probs)
        return probs

    def infer_token(self, probs):
        token = torch.multinomial(probs, 1)
        token = int(token.item())
        return token

    def generate_sequence(self, seq, mems):
        logits = None
        teacher = TeacherForceTask(self.input_data)
        first_loop = True
        for _ in range(self.inference_cfg.GENERATION.generation_length):
            if seq[-1] == 1:
                break

            if teacher.next_tokens_forced:
                next_token = teacher.next_tokens_forced.pop(0)
                seq.append(next_token)
                logits, mems = self.calc_logits_and_mems(seq, mems)
                continue

            if teacher.no_sequence_appended:
                assert logits is not None
                teacher.no_sequence_appended = False
            elif first_loop:
                logits, _ = self.calc_logits_and_mems(seq, mems)
                first_loop = False
            else:
                logits, mems = self.calc_logits_and_mems(seq, mems)

            probs = self.calc_probs(logits)
            probs = self.apply_sampling(probs, teacher.wrong_tokens)

            # 코드 진행 teacher forcing
            # incomplete인 경우 두번째 Bar Token 이후부터 갖춘마디와 같은 방식으로 적용
            if not teacher.incomplete_filled:
                teacher.incomplete_filled = True if seq.count(TOKEN_OFFSET.BAR.value) > 1 else False

            # bar position 뒤에 position 1/128 강제할당
            if teacher.check_first_position(seq):
                teacher.teach_first_position()
                continue

            # 코드 진행 코드 수가 생성 마디수와 같은 경우
            if teacher.check_one_chord_per_bar_case(seq):
                teacher.teach_chord_token()
                continue

            # 코드 진행이 마디 내에서 바뀌는 경우
            if teacher.check_mul_chord_per_bar_case(seq):
                teacher.teach_chord_token()
                continue

            # 토큰 생성 후 토큰이 잘못 되었는지 확인하여 티쳐포싱 진행
            try:
                token = self.infer_token(probs)
            except RuntimeError as e:
                logger.error(f"Sampling Error: {e}")
                seq = None
                break

            # 생성 토큰이 티쳐포싱 되어야할 코드 Position 지나친 경우
            if teacher.check_chord_position_passed(token):
                teacher.teach_chord_position()
                continue

            # 티쳐포싱으로만 입력되어야 할 코드 토큰이 중간에 생성된 경우
            if teacher.check_wrong_chord_token_generated(token):
                teacher.teach_wrong_chord_token(token)
                continue

            # 코드 진행 입력이 다 되지 않았는데 EOS 생성된 경우
            if teacher.check_wrong_eos_generated(token):
                teacher.teach_remnant_chord()
                continue

            # 마지막 마디 생성하는 상황에서 다음 마디 bar token 생성된 경우
            if teacher.check_wrong_bar_token_generated(token):
                teacher.teach_eos()
                continue

            seq.append(token)

        try:
            teacher.validate_teacher_forced_sequence(seq)
        except Exception as error_message:
            logger.error(error_message)
            seq = None
        return seq

    def validate_generated_sequence(self, seq: List[int]) -> bool:
        num_note = 0
        for idx, token in enumerate(seq):
            if idx + 2 > len(seq) - 1:
                break
            if token in range(TOKEN_OFFSET.NOTE_VELOCITY.value, TOKEN_OFFSET.CHORD_START.value):
                if (
                    seq[idx - 1] in range(TOKEN_OFFSET.POSITION.value, TOKEN_OFFSET.BPM.value)
                    and seq[idx + 1]
                    in range(TOKEN_OFFSET.PITCH.value, TOKEN_OFFSET.NOTE_VELOCITY.value)
                    and seq[idx + 2]
                    in range(TOKEN_OFFSET.NOTE_DURATION.value, TOKEN_OFFSET.POSITION.value)
                ):
                    num_note += 1
        return num_note > 0

    def execute(self, encoded_meta) -> List[List[int]]:
        num_conditional_tokens = len(encoded_meta)
        idx = 0
        sequences = []
        while idx != self.input_data.num_generate:
            with torch.no_grad():
                logger.info("Generating the idx: " + str(idx + 1))
                seq, mems = self.init_seq_and_mems(encoded_meta, num_conditional_tokens)
                seq = self.generate_sequence(seq, mems)
                if seq is None:
                    continue
                if not self.validate_generated_sequence(seq):
                    logger.error("Empty sequence generated")
                    continue
            sequences.append(seq)
            idx += 1
        return sequences
