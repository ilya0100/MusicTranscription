import librosa
import numpy as np

from typing import Iterable, List, Tuple
from note_seq import NoteSequence

from src.entities.audio_params import AudioParams
from src.entities.note import Note


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
) -> List[Note]:
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

        notes.append(
            [
                Note(note.pitch, note.velocity)
                for note in prev_notes[::-1]
                if note.end_time > time
            ]
        )

        if single_note and len(notes[-1]) > 1:
            notes[-1] = [notes[-1][0]]

    assert len(notes) == len(times)
    return notes


def detokenize(
    notes: Iterable[Iterable[Note]], times: Iterable[float], frame_time: float
) -> NoteSequence:
    ns = NoteSequence()
    for notes_inner, time in zip(notes, times):
        for note in notes_inner:
            ns.notes.append(
                NoteSequence.Note(
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start_time=time,
                    end_time=time + frame_time,
                )
            )

    return ns
