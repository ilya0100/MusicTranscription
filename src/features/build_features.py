import librosa
import numpy as np

from typing import Iterable, List, Tuple
from note_seq import NoteSequence

from src.entities.audio_params import AudioParams


def make_spectrogram(
    audio_signal: Iterable[float], params: AudioParams, hop_length: int
) -> Tuple[np.ndarray, List[float]]:
    spectrogram = librosa.feature.melspectrogram(
        y=audio_signal,
        sr=params.sample_rate,
        n_mels=params.n_mels,
        fmin=params.fmin,
        fmax=params.fmax,
        n_fft=params.frame_length,
        hop_length=hop_length,
        window=params.window,
    )
    spectrogram = librosa.power_to_db(spectrogram)
    return spectrogram


def split_spectrogram(spectrogram: np.ndarray, width: int) -> np.ndarray:
    if spectrogram.shape[1] % width != 0:
        spectrogram = np.pad(
            spectrogram,
            ((0, 0), (0, width - spectrogram.shape[1] % width)),
            constant_values=0,
        )
    frames = np.array(np.hsplit(spectrogram, spectrogram.shape[1] / width))
    return frames


def tokenize(
    ns: NoteSequence, times: Iterable[float], frame_time: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    ns_sorted = sorted(ns.notes, key=lambda note: note.start_time)
    ns_iter = 0
    notes = []
    prev_notes = []

    for time in times:
        while (
            ns_iter < len(ns_sorted)
            and ns_sorted[ns_iter].start_time - time < frame_time - 1e-6
        ):
            if ns_sorted[ns_iter].end_time - ns_sorted[ns_iter].start_time > 0.02:
                prev_notes.append(ns_sorted[ns_iter])
            ns_iter += 1

        current_notes = np.full(129, -1)
        current_velocities = np.zeros(128, dtype=int)
        notes_count = 0
        for note in prev_notes[::-1]:
            if (
                note.end_time > time
                and np.where(current_notes == note.pitch)[0].size == 0
            ):
                current_notes[notes_count] = note.pitch
                current_velocities[notes_count] = note.velocity
                notes_count += 1

        if current_notes[0] == -1:
            current_notes[0] = 128

        notes.append((np.array(current_notes), np.array(current_velocities)))

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


def join_splitted_notes(
    ns: NoteSequence, recover_miss=False, del_single=False
) -> NoteSequence:
    ns_sorted = sorted(ns.notes, key=lambda note: note.pitch)
    ns_joined = [ns_sorted[0]]
    vels = [ns_sorted[0].velocity]

    eps = 1e-6
    min_time = ns_sorted[0].end_time - ns_sorted[0].start_time + eps
    if recover_miss:
        eps = min_time

    for note in ns_sorted[1:]:
        if (
            ns_joined[-1].pitch == note.pitch
            and note.start_time - ns_joined[-1].end_time < eps
        ):
            ns_joined[-1].end_time = note.end_time
            vels.append(note.velocity)
        else:
            ns_joined[-1].velocity = round(np.mean(vels))
            if (
                del_single
                and ns_joined[-1].end_time - ns_joined[-1].start_time < min_time
            ):
                del ns_joined[-1]

            ns_joined.append(
                NoteSequence.Note(
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start_time=note.start_time,
                    end_time=note.end_time,
                )
            )
            vels = [note.velocity]

    if del_single and ns_joined[-1].end_time - ns_joined[-1].start_time < min_time:
        del ns_joined[-1]
    return NoteSequence(notes=ns_joined)
