from typing import Any

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Loads whisper models into memory to make running multiple predictions efficient"""
        self.model_cache = "model_cache"
        # model_id="distil-whisper/distil-small.en"
        # model_id="distil-whisper/distil-medium.en"
        model_id="distil-whisper/distil-large-v2"
        # model_id="openai/whisper-large-v3"
        torch_dtype = torch.float16
        self.device = "cuda:0"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True, cache_dir=self.model_cache,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id, cache_dir=self.model_cache)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            torch_dtype=torch_dtype,
            device=self.device,
            return_timestamps=True,
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
            batch_size=batch_size,
        )
        return outputs
