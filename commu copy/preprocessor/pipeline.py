import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Union

from logger import logger
from .encoder import EventSequenceEncoder, MetaEncoder
from .parser import MetaParser
from .preprocessor import Preprocessor


class PreprocessPipeline:
    def __init__(self):
        pass

    def __call__(
            self,
            root_dir: Union[str, Path],
            csv_path: Union[str, Path],
            num_cores: int = max(4, cpu_count() - 2),
    ):
        meta_parser = MetaParser()
        meta_encoder = MetaEncoder()
        event_sequence_encoder = EventSequenceEncoder()
        preprocessor = Preprocessor(
            meta_parser=meta_parser,
            meta_encoder=meta_encoder,
            event_sequence_encoder=event_sequence_encoder,
            csv_path=csv_path,
        )
        logger.info(f"Initialized preprocessor")
        logger.info("Start preprocessing")
        start_time = time.perf_counter()
        preprocessor.preprocess(
            root_dir=root_dir,
            num_cores=num_cores,
        )
        print("d")
        end_time = time.perf_counter()
        logger.info(f"Finished preprocessing in {end_time - start_time:.3f}s")
