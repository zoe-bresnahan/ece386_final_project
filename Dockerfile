FROM nvcr.io/nvidia/pytorch:25.02-py3-igpu
WORKDIR /app
RUN apt update && \ 
    apt install -y --no-install-recommends \
    libportaudiocpp0 \
    libportaudio2 \
    gpiod && \
    apt clean
RUN pip install --upgrade --no-cache-dir pip && \ 
    pip install --no-cache-dir \
    transformers==4.49.0 \
    accelerate==1.5.2 \
    sounddevice \
    Jetson.GPIO 
COPY speech_recognition.py .
ENV HF_HOME="/huggingface/"
ENV JETSON_MODEL_NAME=JETSON_ORIN_NANO
ENTRYPOINT ["python", "speech_recognition.py"]  
