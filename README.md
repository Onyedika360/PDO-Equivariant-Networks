
# Reproducing Group Equivariant CNN's( PDO-Econv) on Mnist dataset

This project aims to reproduce the results of the paper **"Partial Differential Operator Based Equivariant Convolutions"** by Zhengyang Shen


# Introduction
This project reproduces the results of the paper **"Partial Differential Operator Based Equivariant Convolutions"** by Zhengyang Shen, using PyTorch and the MNIST dataset. It explores the role of symmetries in image datasets by implementing convolutional networks that are equivariant to transformations from the dihedral (D4) and cyclic groups. By leveraging these structured symmetries, the model builds more robust and generalizable representations for image classification tasks. This reproduction serves both to validate the paper's findings and to deepen understanding of symmetry-based architectures in deep learning.


## Installation

To get started with the Partial Differential Operator-Based Equivariant CNNs project, follow the steps below:

1. **Clone the repository**
   ```bash
   git clone https://github.com/onyedika360/pdo-equivariant-cnns.git
   cd pdo-equivariant-cnns
   

    
### 🔹 Dataset

The dataset used is a preprocessed MNIST file containing group-equivariant transformations. It is hosted on Hugging Face

- **Repo**: `onyedika360/MNIST_D4_and_P4M_transformation`
- **File**: `P4MUPDATED.pkl`
- **Structure**:
  - `images`: Transformed MNIST image data
  - `labels`: Corresponding labels
- **Split**:
  - 10,000 training samples
  - 10,000 validation samples
  - 50,000 testing samples

📌 **Note**: To avoid misclassification due to symmetry ambiguity, only **translation transformations** were applied to digits **6** and **9**. All other digits were transformed using both dihedral (D4) and cyclic group symmetries.

To load the data:
```python
from huggingface_hub import hf_hub_download
import pickle

# Download the file from Hugging Face Hub
file_path = hf_hub_download(repo_id="onyedika360/MNIST_D4_and_P4M_transformation", filename="P4MUPDATED.pkl")

# Load the data
with open(file_path, "rb") as f:
    dataset = pickle.load(f)

images = dataset["images"]
labels = dataset["labels"]

```

### 🔹 Custom Transformations (Optional)

If you'd like to generate your own dataset with dihedral (D4) and cyclic group transformations, you can use the `DataTransform` module provided in `utils`.

#### Step-by-step:

1. **Import the transformation class and method**:
```python
from utils.DataTransform import MnistP4andP4M

transformer = MnistP4andP4M(data=your_images, label=your_labels)
transformed_data = transformer.P4M() # for the D4 + cyclic transformation
transformed_data = transformer.P4() # for the D4 transformation Only

```

## Features

- 🧠 **Reproduction of PDO-Based Equivariant CNNs**  
  Re-implements the architecture described in *"Partial Differential Operator Based Equivariant Convolutions"* using PyTorch.

- 🔄 **Group Equivariant Convolutions (G-CNNs)**  
  Incorporates symmetries from the dihedral group (D4) and cyclic group to improve model robustness and generalization.

- 🧰 **Custom Data Transformation Support**  
  Includes a reusable `MnistP4andP4M` class for generating D4 + cyclic group transformations on any MNIST-like dataset.

- 🧪 **Careful Dataset Design**  
  To avoid ambiguity between symmetric digits, digits **6** and **9** were only transformed via **translations**, not rotations or flips.

- 📈 **Training and Evaluation Framework**  
  Includes utilities for training, validation, and evaluation using modular PyTorch code and logging support.

- 🧪 **Pre-Transformed Dataset Provided**  
  A `P4MUPDATED.pkl` file containing 70,000 transformed MNIST

- 📓 **Jupyter Notebook Workflow**  
  All experimentation, model setup, and evaluation are documented and executable in `PDO-4.6M.ipynb`.



## Dependencies
pip install -r requirements.txt

## Examples

## License

[MIT](https://choosealicense.com/licenses/mit/)

