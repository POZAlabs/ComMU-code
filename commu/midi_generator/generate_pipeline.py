from model_initilizer import ModelInitializeTask
from info_preprocessor import PreprocessTask
from midi_inferrer import InferenceTask
from sequence_postprocessor import PostprocessTask


class MidiGenerationPipeline:
    def __init__(self):
        self.model_initialize_task = ModelInitializeTask()
        self.preprocess_task = PreprocessTask()
        self.inference_task = InferenceTask()
        self.postprocess_task = PostprocessTask()