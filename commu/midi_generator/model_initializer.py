from pathlib import Path
from typing import Tuple

import torch
import yacs.config

from commu.midi_generator.container import ModelArguments
from commu.model.config_helper import get_default_cfg_inference, get_default_cfg_training
from commu.model.dataset import BaseVocab
from commu.model.model import MemTransformerLM


class ModelInitializeTask:
    def __init__(self, model_args: ModelArguments, map_location: str, device: torch.device):
        self.model_args = model_args
        self.map_location = map_location
        self.device = device
        self.inference_cfg = self.initialize_inference_config()

    def initialize_inference_config(self) -> yacs.config.CfgNode:
        inference_cfg = get_default_cfg_inference()
        inference_cfg.freeze()
        return inference_cfg

    def load_checkpoint_fp(self) -> Tuple[Path, Path]:
        checkpoint_dir = self.model_args.checkpoint_dir
        if checkpoint_dir:
            model_fp = Path(checkpoint_dir)
            training_cfg_fp = model_fp.parent / "config.yml"
        else:
            model_parent = Path(self.inference_cfg.MODEL.model_directory)
            model_fp = model_parent / self.inference_cfg.MODEL.checkpoint_name
            training_cfg_fp = model_parent / "config.yml"
        return model_fp, training_cfg_fp

    def initialize_training_cfg(self) -> yacs.config.CfgNode:
        cfg = get_default_cfg_training()
        cfg.defrost()
        cfg.MODEL.same_length = True  # Needed for same_length =True during evaluation
        cfg.freeze()
        return cfg

    def initialize_model(self, training_cfg, model_fp):
        perform_vocab = BaseVocab()
        model = MemTransformerLM(training_cfg, perform_vocab)
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint["model"], strict=False)
        model = model.to(self.device)
        model.eval()
        model.reset_length(1, self.inference_cfg.MODEL.memory_length)
        return model

    def execute(self):
        model_fp, training_cfg_fp = self.load_checkpoint_fp()
        training_cfg = self.initialize_training_cfg()
        model = self.initialize_model(training_cfg, model_fp)
        return model