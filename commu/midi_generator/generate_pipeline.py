import torch

from commu.midi_generator.container import ModelArguments
from commu.midi_generator.model_initializer import ModelInitializeTask
from commu.midi_generator.info_preprocessor import PreprocessTask
from commu.midi_generator.midi_inferrer import InferenceTask
from commu.midi_generator.sequence_postprocessor import PostprocessTask


class MidiGenerationPipeline:
    def __init__(self):
        self.map_location = None
        self.device = None
        self.model_args = None
        self.model_initialize_task = None

        self.preprocess_task = None
        self.inference_task = None
        self.postprocess_task = None

    def initialize_model(self, model_arguments: dict):
        self.map_location = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.map_location)
        self.model_args = ModelArguments(**model_arguments)

        self.model_initialize_task = ModelInitializeTask(
            model_args=self.model_args,
            map_location=self.map_location,
            device=self.device
        )

    def initialize_generation(self):
        self.preprocess_task = PreprocessTask()
        self.inference_task = InferenceTask(self.device)
        self.postprocess_task = PostprocessTask()