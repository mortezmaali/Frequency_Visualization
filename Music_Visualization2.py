# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 13:41:54 2025

@author: Morteza
"""

import pygame
import numpy as np
import scipy.io.wavfile as wav
from pydub import AudioSegment
import os

# Function to convert MP3 to WAV
def mp3_to_wav(mp3_file):
    audio = AudioSegment.from_mp3(mp3_file)
    wav_file = mp3_file.replace(".mp3", ".wav")
    audio.export(wav_file, format="wav")
    return wav_file

# Function to convert frequency to RGB color (smooth mapping across the visible spectrum)
def freq_to_rgb(freq, min_freq, max_freq):
    # Normalize frequency to the range [0, 1]
    normalized_freq = (freq - min_freq) / (max_freq - min_freq)
    
    # Convert normalized frequency to wavelength (in nm)
    wavelength = 380 + (normalized_freq * (740 - 380))
    
    # Mapping of wavelength to RGB using a smoother gradient for the visible spectrum
    if 380 <= wavelength < 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0
        B = 1
    elif 440 <= wavelength < 490:
        R = 0
        G = (wavelength - 440) / (490 - 440)
        B = 1
    elif 490 <= wavelength < 510:
        R = 0
        G = 1
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1
        B = 0
    elif 580 <= wavelength < 645:
        R = 1
        G = -(wavelength - 645) / (645 - 580)
        B = 0
    elif 645 <= wavelength <= 740:
        R = 1
        G = 0
        B = -(wavelength - 740) / (740 - 645)
    else:
        R = 1
        G = 1
        B = 1
    
    # Boost RGB range (multiply by 255) and clip values to the range [0, 255]
    R = min(max(R * 255, 0), 255)
    G = min(max(G * 255, 0), 255)
    B = min(max(B * 255, 0), 255)

    return (R, G, B)

# Function to update the display color based on the frequency
def update_color(frame, audio_data, sampling_rate, max_freq, min_freq, chunk_size=512, color_change_interval=5):
    # Calculate the start index based on the frame number and chunk size
    start = frame * chunk_size
    if start + chunk_size > len(audio_data):
        return (0, 0, 0)  # If we go past the end of the audio, return black
    
    # Only update color every 'color_change_interval' frames
    if frame % color_change_interval != 0:
        return None  # Skip color update for this frame
    
    chunk = audio_data[start:start + chunk_size]  # Take a chunk of audio data
    fft_result = np.fft.fft(chunk)
    freqs = np.fft.fftfreq(len(chunk), 1 / sampling_rate)
    
    # Only keep the positive frequencies (ignore the negative part of FFT result)
    positive_freqs = freqs[:len(freqs) // 2]
    magnitudes = np.abs(fft_result[:len(fft_result) // 2])
    
    # Ensure the maximum magnitude index is within bounds
    max_magnitude_index = np.argmax(magnitudes)
    if max_magnitude_index >= len(positive_freqs):
        max_magnitude_index = len(positive_freqs) - 1  # Cap the index to avoid out of bounds
    
    dominant_freq = positive_freqs[max_magnitude_index]  # Get the corresponding frequency
    
    # Map the dominant frequency to a color
    color = freq_to_rgb(dominant_freq, min_freq, max_freq)
    
    return color

# Load MP3 file and convert it to WAV
mp3_file_path = "C:/Users/Morteza/OneDrive/Desktop/Emily_Music/1_11_2020_Room Tones_Classical_.mp3"
wav_file_path = mp3_to_wav(mp3_file_path)

# Load WAV file
sampling_rate, audio_data = wav.read(wav_file_path)

# Initialize pygame for audio playback and display
pygame.init()
pygame.mixer.init(frequency=sampling_rate)
pygame.mixer.music.load(wav_file_path)
pygame.mixer.music.play()

# Set up the display window to be maximized
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)  # FULLSCREEN mode to maximize window
pygame.display.set_caption("Frequency-to-Color Visualization")

# Get the frequency range (min and max)
fft_result = np.fft.fft(audio_data)
freqs = np.fft.fftfreq(len(audio_data), 1 / sampling_rate)
min_freq = np.min(np.abs(freqs[:len(freqs) // 2]))
max_freq = np.max(np.abs(freqs[:len(freqs) // 2]))

# Main loop to update the display while audio is playing
clock = pygame.time.Clock()
frame = 0
color_change_interval = 10  # Set color change interval (in terms of frames)
while pygame.mixer.music.get_busy():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Get the current frequency and map it to a color (change color every 'color_change_interval' frames)
    color = update_color(frame, audio_data, sampling_rate, max_freq, min_freq, color_change_interval=color_change_interval)
    
    if color:  # Only update the screen if a color is returned
        # Update the screen with the color
        screen.fill(color)  # Fill screen with the color based on the frequency
        pygame.display.flip()
    
    frame += 1
    clock.tick(60)  # Control the frame rate (60 FPS)
    
pygame.quit()
