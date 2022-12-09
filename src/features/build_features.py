import librosa
import numpy as np

from typing import Iterable, List, Tuple
from note_seq import NoteSequence

from src.entities.audio_params import AudioParams


def make_frames(
    audio_signal: Iterable[float], params: AudioParams
) -> Tuple[np.ndarray, List[float]]:
    spectrogram = librosa.feature.melspectrogram(
        y=audio_signal,
        sr=params.sample_rate,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=params.fmax,
        n_fft=params.frame_length,
        hop_length=params.frame_step,
        window=params.window,
    )
    spectrogram = librosa.power_to_db(spectrogram)
    times = [params.frame_time * i for i in range(spectrogram.shape[-1])]
    return spectrogram, times


def tokenize(
    ns: NoteSequence, times: Iterable[float], frame_time: float, single_note=False
) -> List[Tuple[np.ndarray, np.ndarray]]:
    ns_sorted = sorted(ns.notes, key=lambda note: note.start_time)
    ns_iter = 0
    notes = []
    prev_notes = []

    for time in times:
        while (
            ns_iter < len(ns_sorted)
            and ns_sorted[ns_iter].start_time - time < frame_time
        ):
            prev_notes.append(ns_sorted[ns_iter])
            ns_iter += 1

        current_notes = np.full(129, -1)
        current_velocities = np.zeros(128, dtype=int)
        notes_count = 0
        for note in prev_notes[::-1]:
            if note.end_time > time:
                current_notes[notes_count] = note.pitch
                current_velocities[notes_count] = note.velocity
                notes_count += 1

        if current_notes[0] == -1:
            current_notes[0] = 128

        notes.append((np.array(current_notes), np.array(current_velocities)))

        # if single_note and len(notes[-1]) > 1:
        #     notes[-1] = (
        #         np.ndarray(current_notes[0]),
        #         np.ndarray(current_velocities[0]),
        #     )

    assert len(notes) == len(times)
    return notes


def detokenize(
    notes: Iterable[Tuple[np.ndarray, np.ndarray]],
    times: Iterable[float],
    frame_time: float,
) -> NoteSequence:
    ns = NoteSequence()
    for notes_inner, time in zip(notes, times):
        for pitch, velocity in zip(*notes_inner):
            if pitch == 128 or pitch == -1:
                break
            ns.notes.append(
                NoteSequence.Note(
                    pitch=pitch,
                    velocity=velocity,
                    start_time=time,
                    end_time=time + frame_time,
                )
            )

    return ns
