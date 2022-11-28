# -*- coding: utf-8 -*-
import os
import pandas as pd

from typing import Tuple
from torch.utils.data import Dataset

from src.entities.dataset_params import DatasetParams


class WavMidiDataset(Dataset):
    def __init__(self, params: DatasetParams) -> None:
        super().__init__()

        self._root_path = params.root_path
        self._years = params.years_list
        self._split = params.split
        self._data = []

        metadata_path = os.path.join(self._root_path, params.metadata)
        ds_metadata = pd.read_csv(metadata_path)

        if self._split:
            ds_metadata = ds_metadata[ds_metadata["split"] == self._split]
        if len(self._years) > 0:
            ds_metadata = ds_metadata[ds_metadata["year"].map(lambda x: x in self._years)]

        ds_metadata = ds_metadata[["midi_filename", "audio_filename"]]

        self._len = ds_metadata.shape[0]
        self._data = ds_metadata

    def _process_audio(self):
        pass

    def _process_midi(self):
        pass

    def __len__(self):
        return self._len

    def __getitem__(self, idx) -> Tuple:
        midi_filename, audio_filename = self._data.iloc[idx]
        
        midi_path = os.path.join(self._root_path, midi_filename)
        audio_path = os.path.join(self._root_path, audio_filename)

        ns = self._process_midi(midi_path)
        audio = self._process_audio(audio_path)
        return audio, ns
    
    def _process_midi(self, midi_path: str):
        pass

    def _process_audio(self, audio_path: str):
        pass