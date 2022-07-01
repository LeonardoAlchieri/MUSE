[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/LeonardoAlchieri/MUSE/graphs/commit-activity)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/LeonardoAlchieri/MUSE/blob/main/LICENSE)
[![PyPI pyversions](https://img.shields.io/badge/Python-3.10-informational)](https://github.com/LeonardoAlchieri/MUSE)

# MUltimodal Stress Estimation
The MUSE project is aimed at estimating, with traditional machine learning, if a person is stressed or not from biometrics data, specifically:
- ElectroCardioGram (ECG)
- Galvanic Skin Response (GSR)
- Skin Temperature (ST)

The current repo contains the code used to run the project in a modular and re-usable way. If something is amiss, please do not hesitate to contact us.

# Structure

The repo is structured like this:
- `src/run` contains all of the scripts used to generated results, in some form.
- All other folders in `src`, i.e., `src/data`, `src/models` and `src/utils` contain code used inside `src/run`. For example, custom classed to handle data, custom methods to load, etc.
- `notebooks` contains some Jupyter notebooks used to analyse the results; most of the notebooks reflect some task, and they were mostly utilized to create visualizations.
- `helper` contains some other scripts, not necessary for a good functioning
- `visualizations` is the folder where the representations created in `notebook` are made

# Usage

Follows a description, in order as used for this work, of the scripts inside `scr/run`. Each "run" script has a config file for it, with the same name, where a series of parameters can be changed. Please see the config file themselves for which values should be changed or not. For any questions regarding this, do not hesitate to contact us.
1. `src/run/run_time_unravelling.py` allows to unravel the labels, for the dataset, for all values in the T-minutes timeseries. It is also possible to change the length of the timeseries considered.
2. `src/run/run_classical_ml.py` is used to make single-sensor ML models trials
3. `src/run/run_dummy.py` allows to make a random baseline for the predictions, following the same code as for the classical uni-sensor approach; in this case, multiple results, apparently one for each sensor, will be given, but it is due just laziness on our part and re-used code.
4. `src/run/run_multimodal_classical_ml.py` allows to, from a selected combination of sensor-models (fixed) and fusion models (can be given a list for multiple trials), to make multimodal predictions.
5. `src/run/run_train_final_model.py` for using the model to make the final prediction over the test set; the configuration files allow to change the combination for the multimodal. At the moment, not unimodal prediction is possible, but it should not be too hard to modify.

# References

The project is part of *The EMBC 2022 Workshop and Challenge on Detection of Stress and Mental Health Using Wearable Sensors*.
For more information, and to use the data:
```latex
@article{yu2022semi,
  title={Semi-Supervised Learning and Data Augmentation in Wearable-based Momentary Stress Detection in the Wild},
  author={Yu, Han and Sano, Akane},
  journal={arXiv preprint arXiv:2202.12935},
  year={2022}
}
```

For referrencing this work, please reference:
```latex
INSERT CITATION
```

@2022, Leonardo Alchieri, Nouran Abdalazim, Lidia Alecci, prof. Silvia Santini, Shkurta Gashi

<sub>People-Centered Computing Lab - Università della Svizzera italiana, Switzerland</sub>
