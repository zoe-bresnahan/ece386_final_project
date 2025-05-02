import sounddevice as sd
import numpy as np
import numpy.typing as npt
import sys
import time
import Jetson.GPIO as GPIO
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline


def build_pipeline(model_id: str, torch_dtype: torch.dtype, device: str) -> Pipeline:
    """Creates a Hugging Face automatic-speech-recognition pipeline on the given device."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        torch_dtype=torch_dtype,
        device=device,
    )
    return pipe


# function for recording, transcribing, llm, and api chain

if __name__ == "__main__":
    # build whisper pipeline
    # Get model as argument, default to "distil-whisper/distil-medium.en" if not given
    model_id = sys.argv[1] if len(sys.argv) > 1 else "distil-whisper/distil-medium.en"
    print("Using model_id {model_id}")
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device {device}.")

    print("Building model pipeline...")
    pipe = build_pipeline(model_id, torch_dtype, device)
    print(type(pipe))
    print("Done")

    # initialize gpio pin29
    #Init as digital input
    my_pin = 29
    GPIO.setmode(GPIO.BOARD)  # Board pin numbering scheme
    GPIO.setup(my_pin, GPIO.IN) # pin is digital input

    print('Starting Demo! Move pin 29 between 0V and 3.3V')

    # wait for rising edge
    while True:
        GPIO.wait_for_edge(my_pin, GPIO.RISING)
        print('UP!')

        weather= function_name(pipe)
        print(weather)
