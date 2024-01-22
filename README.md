# Experiments of the paper "Kernel Density Matrices for Probabilistic Deep Learning"

This repository contains the code and additional resources for the paper "Kernel Density Matrices for Probabilistic Deep Learning" the paper is available at [arXiv]([https://arxiv.org/abs/2110.00000](https://arxiv.org/abs/2305.18204)). A more recent version of the library can be found at: https://github.com/fagonzalezo/kdm, however, the experiments in the paper were conducted using the code in this repository.

## Contents

This repository includes:

    Code: The implementation code for kernel density matrices (or quantum kernel mixtures, as they are called in this code base) for probabilistic deep learning.
    Examples: Jupyter notebooks demonstrating the usage of the framework for image classification and learning with label proportions.
    Data: Sample datasets used in the examples.
    Documentation: Detailed documentation and instructions for using the code and reproducing the experiments.

## Usage

Examples of code for KDM classification, maximum likelihood, and generative modeling in two moons and MNIST datasets can be found in `kqm_example.ipynb`.

The dependencies are found in the root of this repository on file `pyproject.toml`. You can create an environment an execute `pip install .` to install all the dependencies based on this file. Here is an explanation of each dependency:

    Python: The experimentation requires Python version 3.8 or higher but lower than 3.12. 
    TensorFlow: The experimentation relies on TensorFlow version 2.12.0.
    JAX: The experimentation uses JAX version 0.3.15. 
    Keras: The experimentation utilizes Keras version 2.12.0.
    TensorFlow Probability: The experimentation depends on TensorFlow Probability version 0.19.0. 
    scikit-learn: The experimentation requires scikit-learn version 1.0.1.
    pandas: The experimentation relies on pandas version 1.1.5.
    matplotlib: The experimentation uses matplotlib version 3.4.3.
    seaborn: The experimentation depends on seaborn version 0.11.2.
    pathos: The experimentation uses pathos version 0.3.0. pathos is a library for parallel computing in Python. It provides a high-level interface for executing code in parallel, which can be useful for speeding up certain computations.
    networkx: The experimentation depends on networkx version 3.1. networkx is a library for working with complex networks and graphs. It provides data structures and algorithms for analyzing and manipulating graph data.
    pillow: The experimentation uses pillow version 9.5.0. pillow is a fork of the Python Imaging Library (PIL) and provides a powerful set of image processing capabilities. It is commonly used for loading, manipulating, and saving images in various formats.


# Experimental Setup

In this subsection, we provide a comprehensive description of our experimental setup to ensure reproducibility.

**Hardware Specifications**

Our experiments were conducted using five distinct servers, each with specific configurations detailed below:

    Server A: This machine is equipped with an 8-core Intel 12 2.2 GHz processor, 64 GB of RAM, and an NVIDIA 3080 RTX GPU.
    Server B: Featuring a 40-core Intel Xeon 4114 2.2 GHz processor and 64 GB of RAM, this server does not have a GPU.
    Server C: This computer utilizes a 16-core Intel Xeon E5-2630 V3 2.40 GHz processor, 64 GB of RAM, and an NVIDIA GeForce RTX 2080 GPU.
    Server D: With a robust configuration, this server incorporates a 64-core Intel Xeon Silver 4216 CPU 2.10 GHz processor, 128 GB of RAM, and two NVIDIA RTX A5000 GPUs.
    Server E: The final server comprises a 16-core Intel Xeon E5-2640 V3 2.60 GHz processor, 80 GB of RAM, and an NVIDIA GTX TITAN X GPU.

## Classification with kernel density matrices

Three different models were assessed in this experiment: KDM classification model (QKM), KDM model fined-tuned for a generation with maximum likelihood learning (ML-QKM), and a baseline model with the same encoder coupled with a dense layer. The encoder model utilized for both MNIST and Fashion-MNIST datasets was identical. It consisted of the following components:

    Lambda Layer: The initial layer converted each sample into a 32-float number and subtracted 0.5 from it.
    Convolutional Layers: Two convolutional layers were appended, each with 32 filters, 5-kernel size, same padding, and strides 1 and 2, respectively. Subsequently, two more convolutional layers were added, each with 64 filters, 5-kernel size, same padding, and strides 1 and 2, respectively. Finally, a convolutional layer with 128 filters, 7-kernel size, and stride 1 was included. All convolutional layers employed the Gaussian Error Linear Unit (GELU) activation function.
    Flattened Layer: The output from the previous layer was flattened.
    Dropout Layer: A dropout layer with a dropout rate of 0.2 was introduced.
    Dense Layer: The subsequent dense layer's neuron encoding size was determined using a grid hyperparameter search in the range of $2^i$, where $i \in {1,\cdots,7}$.

This model can be found on `src/kqm/encoder_model.py`.

The decoder architecture consisted of the following components:

    Reshape Layer: The input to the decoder passed through a reshape layer.
    Convolutional Layers: Three convolutional layers were employed, each with 64 filters and kernel sizes of seven, five, and five, respectively. The strides for these layers were set to 1, 1, and 2, while the padding was configured as valid, same, and same, respectively.
    Additional Convolutional Layers: Subsequently, three more convolutional layers were utilized, each with 32 filters, 5-kernel size, and strides 1, 2, and 1, respectively. All these layers utilized the Gelu activation function and had padding set to "same."
    Final Convolutional Layer: The decoder concluded with a 1-filter convolutional layer employing same padding.

Similar to the encoder, all layers in the decoder employed the GELU activation function.

This model can be found on `src/kqm/decoder_model.py`.

For Cifar-10 a different encoder was used for 

     Convolutional Layers: The input tensor is passed through a series of convolutional layers. Each layer applies a 3x3 kernel to extract features from the input. The activation function used is the GELU. Padding is set to 'same'. Batch normalization is applied after each convolutional layer to normalize the activations.
     Max Pooling: After each pair of convolutional layers, max pooling is performed using a 2x2 pool size.
     Encoding Layer: The feature maps obtained from the previous layers are flattened into a vector representation using the Flatten layer.
     Dropout: A dropout layer with a dropout rate of 0.2 is added.
     Hidden Layer: A dense layer of variable size is introduced to further transform the encoded features. The size is search.
     Dropout: Another dropout layer with a dropout rate of 0.2 is added after the hidden layer.

This model can be found on `src/kqm/encoder_model.py`.

The decoder architecture for Cifar-10 consisted of the following components:

    Dense Layer: The input tensor is passed through a dense layer that restores the tensor to its original size.
    Reshape Layer: The tensor is reshaped to match the spatial dimensions of the original input using the Reshape layer.
    Convolutional Transpose Layers: The reshaped tensor undergoes a series of convolutional transpose layers. Each layer applies a 3x3 kernel to upsample the feature maps. The activation function used is 'gelu'. Padding is set to 'same' to maintain the spatial dimensions.
    Batch Normalization: Batch normalization is applied after each convolutional transpose layer to normalize the activations.
    Upsampling: UpSampling2D layers with a size of (2, 2) are used to increase the spatial dimensions of the feature maps.
    Final Convolutional Transpose Layer: The last convolutional transpose layer reconstructs the original number of channels in the input data. Padding is set to 'same'.
    Activation Function: The final output is passed through an activation function, 'sigmoid', to ensure the reconstructed data is within the appropriate range.

The number of components in every dataset for the QKM classification model was searched in 2^i, where i ∈ {2,...,10}. The same for ML-KQM. For the baseline model, the dense layer was searched using the same number of parameters that the QKM classification model generates. The best parameters can be found in Table "best_parameters_experiment_classification".


The models can be found in: https://drive.google.com/file/d/1ieqPQu1YMcBZF2P_5cY-Ym7pqRLddmQ9/view?usp=share_link

After download them, decomprese in /NeurIPS-2023/models/

This model can be found on `src/kqm/decoder_model.py`.

The best hyperparameters for QKM, ML-QKM, and baseline are as follows:

| Dataset        | QKM Components | Encoded Size | QKM-ML Components | Encoded Size | Baseline Dense Layer Components | Encoded Size |
| -------------- | -------------- | ------------ | ----------------- | ------------ | ------------------------------ | ------------- |
| Mnist          | 512            | 128          | 512               | 128          | 8                              | 8             |
| Fashin-Mnist   | 256            | 128          | 256               | 128          | 4                              | 16            |
| Cifar-10       | 1024           | 32           | 1024              | 32           | 4                              | 64            |

All the code assumes that the repository is saved on `/NeurIPS-2023/`. You might create a symbolic link using `ln -s /path_to_repository /NeurIPS-2023/`.  

If you want to reproduce the results of classification based on Mnist, Fashin-Mnist, and Cifar-10 you can run the scripts in the folder `scripts` files `scripts/classification.py` and `scripts/classification_with_density.py`. For running them, execute as follows: 

$ python scripts/classification.py
is_an_initial_test: False
best_config: True
repetitions_for_each_experiment: 1
use_stored_model: True

$ python scripts/classification.py --is_an_initial_test
is_an_initial_test: True
best_config: True
repetitions_for_each_experiment: 1
use_stored_model: True

$ python scripts/classification.py --best_config=False --repetitions_for_each_experiment=5
is_an_initial_test: False
best_config: False
repetitions_for_each_experiment: 5
use_stored_model: True

$ python scripts/classification.py --best_config=False --use_stored_model=False
is_an_initial_test: False
best_config: False
repetitions_for_each_experiment: 5
use_stored_model: False

$ python scripts/classification_with_density.py
is_an_initial_test: False
best_config: True
repetitions_for_each_experiment: 1
use_stored_model: True


If you want to reproduce the results of the generative process using ML-QKM in the folder `scripts` files `scripts/classification.py` and `scripts/classification_with_density.py` are found. For running: 

$ python scripts/classification_with_density.py
is_an_initial_test: False
best_config: True
repetitions_for_each_experiment: 1
use_stored_model: True


If you want to reproduce the results of the learning with label proportions you can run the following script:

$ python scripts/experiment_learning_with_label_proportions.py
is_an_initial_test: False
best_config: True
use_stored_model: True

The best hyperparameters for learning with label proportions are as follows: 

| Dataset, sample | Bag Size | Num. Components | Learning Rate |
|-----------------|----------|-----------------|---------------|
| Adult, $[0, \frac{1}{2}]$ | 8   | 32   | 0.005 |
| Adult, $[0, \frac{1}{2}]$ | 32  | 16   | 0.001 |
| Adult, $[0, \frac{1}{2}]$ | 128 | 512  | 0.001 |
| Adult, $[0, \frac{1}{2}]$ | 512 | 64   | 0.005 |
| Adult, $[\frac{1}{2},1]$ | 8   | 16   | 0.005 |
| Adult, $[\frac{1}{2},1]$ | 32  | 64   | 0.001 |
| Adult, $[\frac{1}{2},1]$ | 128 | 128  | 0.001 |
| Adult, $[\frac{1}{2},1]$ | 512 | 64   | 0.005 |
| MAGIC, $[0, \frac{1}{2}]$ | 8   | 256  | 0.005 |
| MAGIC, $[0, \frac{1}{2}]$ | 32  | 128   | 0.005 |
| MAGIC, $[0, \frac{1}{2}]$ | 128 | 128  | 0.001 |
| MAGIC, $[0, \frac{1}{2}]$ | 512 | 256   | 0.001 |
| MAGIC, $[\frac{1}{2},1]$ | 8   | 16   | 0.005 |
| MAGIC, $[\frac{1}{2},1]$ | 32  | 128  | 0.005 |
| MAGIC, $[\frac{1}{2},1]$ | 128 | 32   | 0.005 |
| MAGIC, $[\frac{1}{2},1]$ | 512 | 128  | 0.001 |

The data for this experiment can be found in: https://drive.google.com/file/d/1G2QKdSqtS9Mb2pzsXitgZUPhqL8IO8mm/view?usp=sharing

    After download them, decompress in /NeurIPS-2023/scripts/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/

The original scripts and repository of the data can be found in: 

    https://github.com/Z-Jianxin/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework

Citation

If you find this work useful in your research, please consider citing:

@misc{gonzalez2023quantum,
      title={Kernel Density Matrices for Probabilistic Deep Learning}, 
      author={Fabio A. González and Raúl Ramos-Pollán and Joseph A. Gallego-Mejia},
      year={2023},
      eprint={2305.18204},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

# License

