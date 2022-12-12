import os
import numpy as np
import pandas as pd
import librosa
import note_seq

from typing import Iterable, Tuple
from torch.utils.data import Dataset

from src.entities.dataset_params import DatasetParams
from src.entities.audio_params import AudioParams
from src.features.build_features import make_spectrogram, split_spectrogram, tokenize


class WavMidiDataset(Dataset):
    def __init__(self, params: DatasetParams) -> None:
        super().__init__()

        self._root_path = params.root_path
        self._years = params.years_list
        self._split = params.split
        self._params = params

        self._hop_length = params.audio_params.frame_length // params.overlapping
        self._frame_time = (
            self._hop_length * params.feature_size / params.audio_params.sample_rate
        )

        self._data = []

        metadata_path = os.path.join(self._root_path, params.metadata)
        ds_metadata = pd.read_csv(metadata_path)

        if self._split:
            ds_metadata = ds_metadata[ds_metadata["split"] == self._split]
        if len(self._years) > 0:
            ds_metadata = ds_metadata[
                ds_metadata["year"].map(lambda x: x in self._years)
            ]

        ds_metadata = ds_metadata[["midi_filename", "audio_filename"]]

        self._len = ds_metadata.shape[0]
        self._data = ds_metadata

    def __len__(self):
        return self._len

    def __getitem__(self, idx) -> Tuple[np.ndarray, Tuple[np.ndarray]]:
        midi_filename, audio_filename = self._data.iloc[idx]

        midi_path = os.path.join(self._root_path, midi_filename)
        audio_path = os.path.join(self._root_path, audio_filename)

        frames = self._process_audio(audio_path, self._params.audio_params)
        times = [self._frame_time * i for i in range(frames.shape[0])]

        notes = self._process_midi(midi_path, times)
        assert len(times) == len(notes)

        return frames, notes, times

    def _process_audio(self, audio_path: str, params: AudioParams):
        signal, _ = librosa.load(audio_path, sr=params.sample_rate)
        spectrogram = make_spectrogram(signal, params, self._hop_length)
        frames = split_spectrogram(spectrogram, self._params.feature_size)
        return frames

    def _process_midi(self, midi_path: str, times: Iterable[float]):
        ns = note_seq.midi_file_to_note_sequence(midi_path)
        return tokenize(ns, times, self._frame_time)


class AudioDataset(Dataset):
    def __init__(self, frames: np.ndarray, notes: Tuple[np.ndarray]) -> None:
        super().__init__()

        assert frames.shape[-1] == len(notes)

        self._frames = frames
        self._notes = notes
        self._len = len(notes)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        return self._frames[:, index], self._notes[index]
