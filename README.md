This project's goal was to design and construct a voice assistant that listens for a wake word; records a 
sentence asking for weather at a city, airport, or location; returns the weather at that spot.

To use this voice assistant you need to open the terminal on the Jetson Orin Nano and cd into the file location. Once there run the following command to 
build the Docker container 'sudo docker buildx build . -t whisper'. This command also installs the necessary libraries and sets critical variables like entry point. Then 
once the container is built use the command 'sudo docker run -it --rm --device=/dev/snd --device=/dev/gpiochip0 --runtime=nvidia whisper' to launch the program. 
First the whisper transcription pipeline will be downloaded allowing for a "hot start" once the keyword is said. Then the Jetson is waiting for a rising edge on
pin 29 which occurs when the Arduino detects the keyword being said. Then the program begins recording the users voice using the webcam to get the prompt of where they want the weather.
The audio is then transcribed and sent to the ollama llm for processing automatically. The llm function returns the formatted location and passes it to the wttr.in API to retrieve the weather.
The weather at the location is then printed to the computer screen for the user to see and the program goes back into wait mode until the keyword is said again or the program is exited.

Documentation Statement:
I went to EI with Captain Yarbrough multiple times for the checkpoints and the final implementation. I also used chatGPT to understand certain errors in my code and get direction on what may be causing them.
