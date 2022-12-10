import os
import numpy as np
import pandas as pd
import librosa
import note_seq

from typing import Iterable, Tuple
from torch.utils.data import Dataset

from src.entities.dataset_params import DatasetParams
from src.entities.audio_params import AudioParams
from src.features.build_features import make_frames, tokenize


class WavMidiDataset(Dataset):
    def __init__(self, params: DatasetParams) -> None:
        super().__init__()

        self._root_path = params.root_path
        self._years = params.years_list
        self._split = params.split
        self._audio_params = params.audio_params
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

        frames, times = self._process_audio(audio_path, self._audio_params)
        notes = self._process_midi(midi_path, times)

        assert frames.shape[-1] == len(notes)

        return frames, notes, times

    def _process_audio(self, audio_path: str, params: AudioParams):
        signal, _ = librosa.load(audio_path, sr=params.sample_rate)
        frames, times = make_frames(signal, params)

        assert frames.shape[-1] == len(times)

        return frames, times

    def _process_midi(self, midi_path: str, times: Iterable[float]):
        ns = note_seq.midi_file_to_note_sequence(midi_path)
        return tokenize(ns, times, self._audio_params.frame_time)


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
