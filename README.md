AutomaticMusicTranscription
==============================

<h2 align="center"><img src="https://github.com/ilya0100/MusicTranscription/blob/72477b5f4db3d18be81c5c386d4c729028486088/readme/1.png"></h2>
<h2 align="center"><img src="https://github.com/ilya0100/MusicTranscription/blob/72477b5f4db3d18be81c5c386d4c729028486088/readme/2.png"></h2>
<h2 align="center"><img src="https://github.com/ilya0100/MusicTranscription/blob/72477b5f4db3d18be81c5c386d4c729028486088/readme/3.png"></h2>
<h2 align="center"><img src="https://github.com/ilya0100/MusicTranscription/blob/72477b5f4db3d18be81c5c386d4c729028486088/readme/4.png"></h2>
<h2 align="center"><img src="https://github.com/ilya0100/MusicTranscription/blob/72477b5f4db3d18be81c5c386d4c729028486088/readme/5.png"></h2>
<h2 align="center"><img src="https://github.com/ilya0100/MusicTranscription/blob/72477b5f4db3d18be81c5c386d4c729028486088/readme/6.png"></h2>

[Predict_v1](https://drive.google.com/file/d/1QX3fVzG-JrFjh37t3Xe0hCk2FIZ98SBx/view?usp=share_link)

Automatic music transcription using neural networks

Installation
------------
**NOTE**: In order to install the `rtmidi` package that we depend on, you may need to install headers for some sound libraries. On Ubuntu Linux, this command should install the necessary packages:

```bash
sudo apt-get install build-essential libasound2-dev libjack-dev portaudio19-dev
```
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python setup.py install
```

Data
------------
- [MusicNet Dataset](https://www.kaggle.com/datasets/imsparsh/musicnet-dataset)
- [The MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
