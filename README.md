# Bi-Input-EEGConv

This is the Repository of Bi-Input-EEGConv for experiment. Bi-Input-EEGConv is a kind of Convolutional Network using for pre-processed EEG datas and graphs, then returning classification results.

## Dataset

- [BCI Competition IV 2a](http://www.bbci.de/competition/iv/)

## Usage

### Requirement

- Python == 3.6
- tensorflow-gpu >= 1.14.0
- scikit-learn >= 0.21.3
- scipy >= 1.3.1
- numpy >= 1.16.5
- pydot >= 1.4.1
- pywavelets >= 1.0.3
- hdf5 >= 1.10.4
- h5py >= 2.9.0
- matplotlib >= 3.1.1
- opencv-contrib-python >= 4.1.1.26
- graphviz >= 2.38

Recommend to use conda environment.

### Coding

```python
import BIEEGConv as bieeg
# TODO: finish the doc
```

## Experiment Plans

1. ~~EEGNet(8,2) with 5-fold average validation~~
2. ~~TSGEEGNet(16,4) with 5-fold average validation~~
3. ~~EEGNet(16,4) with 5-fold average validation~~
4. TSGEEGNet(16,4) with different sparse parameters and 5-fold average validation
5. cropped training
6. add attention method