from typing import Any

import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizerFast,
    WhisperForConditionalGeneration,
    pipeline,
)
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""
        self.model_cache = "model_cache"
            # model_id="distil-whisper/distil-medium.en"
        model_id="distil-whisper/distil-large-v2"
        torch_dtype = torch.float16
        self.device = "cuda:0"
        model = WhisperForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            cache_dir=self.model_cache,
        ).to(self.device)
        tokenizer = WhisperTokenizerFast.from_pretrained(
            model_id, cache_dir=self.model_cache
        )
        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_id, cache_dir=self.model_cache
        )
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            model_kwargs={"use_flash_attention_2": True},
            torch_dtype=torch_dtype,
            device=self.device,
        )

    def predict(
        self,
        audio: Path = Input(description="Audio file"),
        batch_size: int = Input(
            default=24,
            description="Number of parallel batches you want to compute. Reduce if you face OOMs. (default: 24).",
        ),
    ) -> Any:
        """Transcribes and optionally translates a single audio file"""
        outputs = self.pipe(
            str(audio),
            chunk_length_s=30,
            batch_size=batch_size,
            return_timestamps=True,
        )
        return outputs
