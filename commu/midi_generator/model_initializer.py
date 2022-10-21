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
        inference_cfg.merge_from_file(self.model_args.inference_config)
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

    def initialize_training_cfg(self, config_path: Path) -> yacs.config.CfgNode:
        cfg = get_default_cfg_training()
        # The following try to merge the configurations from yaml file,
        # and since we have "", which is integrated as None and can not be read by yacs,
        # we have the following block "try: except:" to read from list rather than merge from file.
        try:
            cfg.merge_from_file(config_path)
        except Exception as e:
            print("*" * 100)
            print(
                "Note, if you are loading an old config.yml file which includes None inside,\n"
                " please change it to a string 'None' to make sure you can do training_cfg.merge_from_file.\n"
                "e.g. training_cfg.DISCRIMINATOR.type, training_cfg.TRAIN.pad_type "
                "and training_cfg.TRAIN.load_from_previous.\n"
                "and please note DISCRIMINATOR.temperature is DISCRIMINATOR.beta_max\n"
            )
            print("*" * 100)
            raise e

        cfg.defrost()
        cfg.DISCRIMINATOR.type = (
            "Null"  # cnn for cnn distriminator or Null for no discriminator or 'bert' for BERT
        )
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
        training_cfg = self.initialize_training_cfg(training_cfg_fp)
        model = self.initialize_model(training_cfg, model_fp)
        return model