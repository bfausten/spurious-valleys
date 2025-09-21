# Spurious Valleys in Neural Loss Landscapes
This repo serves as a collection of the code used for plots in my bachelor's thesis about the presence and absence of spurious 
valleys in the loss landscape of neural networks. The thesis is based on the paper *Spurious Valleys in One-hidden-layer Neural Network Optimization Landscapes*
by Venturi et al. (2019) [[paper](https://jmlr.org/papers/v20/18-674.html)]. The code is written in Python and uses PyTorch.

## Requirements
The code was tested with Python 3.11.12. Other required packages are listed in `requirements.txt`. You can install them using pip or conda:
```bash
pip install -r requirements.txt
```
or
```bash
conda install --file requirements.txt
```
## Organization
The code is organized into the following folders:
- `misc`: Contains only first experiments not used in the thesis.
- `theorem8`: Contains code for two experiments related to Theorem 8 in the original paper (positive result, Section 3.1 in my thesis) and their results in `outputs_t8`.
- `theorem13`: Contains code for an experiment related to Theorem 13 in the original paper (first negative result, Section 3.2 in my thesis) and their results in `outputs_t13`.
- `theorem16`: Contains code for an experiment related to Theorem 16 in the original paper (second negative result, Section 3.3 in my thesis) and their results in `outputs_t16`.

## Usage
### Positive Result (Section 3.1, Theorem 8 (Venturi et al. 2019))
To run the experiments related to Theorem 8, navigate to the `theorem8` folder and run `main1()` (Figure 2a) or `main2()` (Figure 2b) in `main.py`.
The results will be saved in the `outputs_t8` folder. All helper functions are contained in `theorem_8.py`.

### Negative Result 1 (Section 3.2, Theorem 13 (Venturi et al. 2019))
To run the experiments related to Theorem 13, navigate to the `theorem13` folder and run `theorem_13.py` (Figure 4).
The results will be saved in the `outputs_t13` folder.

### Negative Result 2 (Section 3.3, Theorem 16 (Venturi et al. 2019))
To run the experiments related to Theorem 16, navigate to the `theorem16` folder and run `theorem_16.py` (Figure 5).
The results will be saved in the `outputs_t16` folder. All helper functions related to the dataset are contained in `th16data.py` and all helper functions for the model architecture and training are in `th16utils.py`.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
