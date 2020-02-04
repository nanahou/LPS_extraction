# LPS_extraction
============================================

The script is to extract log-power-spectrum (LPS) for speech enhancement and bandwidth extension.

### Requirements

The model is implemented in Tensorflow and Keras and uses several additional libraries. Specifically, we used:

* `pytorch==1.0`
* `python==3.6.8`
* `numpy==1.15.4`
* `scipy==1.2.0`

### Setup

To install this package, simply clone the git repo:

```
git clone https://github.com/nanahou/LPS_extraction.git;
cd LPS_extraction;
```

### Contents

The repository is structured as follows.

* `./data`: some audio samples from dataset[1]
* `audioread.py`: the function to read audios
* `extract_LPS.py`: the main scripts to extract features
* `normhamming.py`: the function to apply a normalized square root hamming periodic window 
* `plot_spectrum.py`: the function to plot the LPS features
* `sigproc.py`: including the functions to frame signals, deframe signals from [2]

### Usage

* If extracting LPS features, you only need to replace the path in `extract_LPS.py` with your own data path and run: 

  ```python extract_LPS.py```

* If plotting your features, you only need to call the function in `plot_spectrum.py`.

```
[1]. Valentini-Botinhao, C., Wang, X., Takaki, S. and Yamagishi, J., 2016. Speech Enhancement for a Noise-Robust Text-to-Speech Synthesis System Using Deep Recurrent Neural Networks. In Interspeech (pp. 352-356).
[2]. https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/sigproc.py
```
