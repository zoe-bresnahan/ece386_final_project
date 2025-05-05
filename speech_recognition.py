import sounddevice as sd
import numpy as np
import numpy.typing as npt
import sys
import time
import Jetson.GPIO as GPIO
from ollama import Client
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


def record_audio(duration_seconds: int = 10) -> npt.NDArray:
    """Record duration_seconds of audio from default microphone.
    Return a single channel numpy array."""
    sample_rate = 16000  # Hz
    samples = int(duration_seconds * sample_rate)
    # Will use default microphone; on Jetson this is likely a USB WebCam
    audio = sd.rec(samples, samplerate=sample_rate, channels=1, dtype=np.float32)
    # Blocks until recording complete
    sd.wait()
    # Model expects single axis
    return np.squeeze(audio, axis=1)


LLM_MODEL: str = "gemma3:27b"  # Optional, change this to be the model you want
client: Client = Client(
    host="http://ai.dfec.xyz:11434"  # Optional, change this to be the URL of your LLM
)


def llm_parse_for_wttr(prompt: str) -> str:
    response = client.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt},
            {
                "role": "system",
                "content": """#Overview:
         the prompt is a sentence or question that includes a subject of a city, landmark, or airport. Extract the subject from the prompt then reformat it to fit the criteria
         ##Format Criteria
         -make output all lowecase
         -if the subject is a city return the city but replace spaces with plus signs
         -if the subject is a landmark return the name of the landmark with the spaces replaced with plus signs and a tilda at the beginning of the return statement
         -if the subject is an airport return the airport identification code
         ###Examples
         -input: what is the weather los angeles, expected output: los+angeles
         -input: give me the weather at the leaning tower of pisa, expected output: ~leaning+tower+of+pisa
         -input: what is the weather at the los angeles airport, expected output: lax
         -input: what is the weather in paris france, expected output: paris""",
            },
        ],
        # temperature=0
    )
    return response["message"]["content"]


# function for recording, transcribing, llm, and api chain
def weather_function(pipe: Pipeline):

    # record audio
    print("Recording...")
    audio = record_audio()
    print("Done")

    # get transcription
    print("Transcribing...")
    start_time = time.time_ns()
    speech = pipe(audio)
    end_time = time.time_ns()
    print("Done")

    # get location from transcription using llm
    location = llm_parse_for_wttr(speech)

    weather = f"curl wttr.in/{location}"

    return weather


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
    # Init as digital input
    my_pin = 29
    GPIO.setmode(GPIO.BOARD)  # Board pin numbering scheme
    GPIO.setup(my_pin, GPIO.IN)  # pin is digital input

    print("Starting Demo! Move pin 29 between 0V and 3.3V")

    # wait for rising edge
    while True:
        GPIO.wait_for_edge(my_pin, GPIO.RISING)
        print("UP!")

        weather = weather_function(pipe)
        print(weather)
