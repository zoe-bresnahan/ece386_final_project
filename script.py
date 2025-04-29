import sounddevice as sd
import numpy as np
import numpy.typing as npt
import sys
import time
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Pipeline, pipeline
