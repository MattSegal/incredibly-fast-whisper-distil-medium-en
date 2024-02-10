import time
from typing import Any

import torch
from cog import BasePredictor, Input, Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class Whisper:
    LARGE_V3 = "openai/whisper-large-v3"
    MEDIUM_EN = "openai/whisper-medium.en"
    SMALL_EN = "openai/whisper-small.en"
    TINY_EN = "openai/whisper-tiny.en"


WHISPER_MODEL = Whisper.LARGE_V3

BATCH_SIZE = 24
CHUNK_LENGTH_S = 30
STRIDE_LENGTH_S = (4, 2)  # Not clear if the 2nd param is required
MAX_NEXT_TOKENS = 512
MODEL_CACHE = "model_cache"


class Predictor(BasePredictor):
    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""
        self.model_cache = MODEL_CACHE
        self.device = "cuda:0"
        torch_dtype = torch.float16
        with Timer(f"loading {WHISPER_MODEL}"):
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                WHISPER_MODEL,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                use_flash_attention_2=True,
                cache_dir=self.model_cache,
            )
            model.to(self.device)
            processor = AutoProcessor.from_pretrained(
                WHISPER_MODEL, cache_dir=self.model_cache
            )
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=MAX_NEXT_TOKENS,
                chunk_length_s=CHUNK_LENGTH_S,
                stride_length_s=STRIDE_LENGTH_S,
                torch_dtype=torch_dtype,
                device=self.device,
                return_timestamps=True,
            )

    def predict(self, audio: Path = Input(description="Audio file")) -> Any:
        """Transcribes and optionally translates a single audio file"""
        with Timer(f"running prediction with {WHISPER_MODEL}"):
            outputs = self.pipe(str(audio), batch_size=BATCH_SIZE)

        return outputs


class Timer:

    def __init__(self, msg: str) -> None:
        self.msg = msg

    def __enter__(self):
        self.start_time = time.time()
        print(f"Started {self.msg}...")

    def __exit__(self, *args, **kwargs):
        end_time = time.time()
        elapsed_time = int(end_time - self.start_time)
        print(f"Finished {self.msg} (took {elapsed_time} seconds)")
