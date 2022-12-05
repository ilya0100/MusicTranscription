import tensorflow as tf

from typing import Iterable, List, Tuple
from note_seq import NoteSequence

from src.entities.audio_params import AudioParams
from src.entities.note import Note


def make_frames(
    audio_signal: Iterable[float], params: AudioParams
) -> Tuple[tf.Tensor, List[float]]:
    frames = tf.signal.frame(
        audio_signal,
        frame_length=params.frame_length,
        frame_step=params.frame_step,
        pad_end=params.pad_end,
        pad_value=params.pad_value,
    )
    # frame_time = frame_length / sample_rate
    times = [params.frame_time * i for i in range(frames.shape[0])]
    return frames, times


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
