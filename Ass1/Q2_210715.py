import cv2
import numpy as np
import librosa

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass

    y, sr = librosa.load(audio_path, sr=None)
    n_fft = 2048
    hop_length = 512
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, fmax=22000)

    offset = 1e-10
    spec_db = 10 * np.log10(np.maximum(spec + offset, np.finfo(float).tiny))

    clip_value = 1e-5 
    clipped_spec_db = np.clip(spec_db, clip_value, None)
    value = np.mean(np.abs(clipped_spec_db))

    class_name = 'cardboard' if value < 0.8 else 'metal'

    return class_name
