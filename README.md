# Noise2Noise: Learning Image Restoration without Clean Data - _Official TensorFlow implementation of the ICML 2018 paper_

**Jaakko Lehtinen**, **Jacob Munkberg**, **Jon Hasselgren**, **Samuli Laine**, **Tero Karras**, **Miika Aittala**, **Timo Aila**

**Abstract**:

_We apply basic statistical reasoning to signal reconstruction by machine learning -- learning to map corrupted observations to clean signals -- with a simple and powerful conclusion: it is possible to learn to restore images by only looking at corrupted examples, at performance at and sometimes exceeding training using clean data, without explicit image priors or likelihood models of the corruption. In practice, we show that a single model learns photographic noise removal, denoising synthetic Monte Carlo images, and reconstruction of undersampled MRI scans -- all corrupted by different processes -- based on noisy data only._

![alt text](img/n2nteaser_1024width.png "Denoising comparison")


## Resources

* [Paper (arXiv)](https://arxiv.org/abs/1803.04189)


## Project Structure

### Core Components
- `network.py`: Implementation of the neural network architecture for Noise2Noise
- `train.py`: Main training script for the general denoising model
- `validation.py`: Tools for validating trained models on test datasets
- `config.py`: Configuration settings and hyperparameters for training

### Dataset Management
- `dataset.py`: Base dataset handling and loading utilities
- `dataset_tool_tf.py`: TensorFlow-specific dataset preprocessing tools
- `dataset_tool_mri.py`: Specialized tools for MRI dataset processing
- `download_kodak.py`: Script to download Kodak dataset for testing

### MRI-Specific Components
- `train_mri.py`: Specialized training script for MRI denoising
- `config_mri.py`: Configuration settings specific to MRI processing

### Utilities
- `util.py`: General utility functions used across the project
- `dnnlib/`: Deep neural network library containing core functionality
  - Network components
  - Training frameworks
  - Utility functions

### Resources
- `img/`: Directory containing images for documentation
- `requirements.txt`: List of Python dependencies
- `LICENSE.txt`: Project license information

## Dependencies

The project requires the following main dependencies (see `requirements.txt` for specific versions):
- TensorFlow
- NumPy
- OpenCV
- PIL (Python Imaging Library)

## Getting started

The below sections detail how to get set up for training the Noise2Noise network using the ImageNet validation dataset.   Noise2Noise MRI denoising instructions are at the end of this document.

### Python requirements

This code is tested with Python 3.6.  We're using [Anaconda 5.2](https://www.anaconda.com/download/) to manage the Python environment.  Here's how to create a clean environment and install library dependencies:

```
conda create -n n2n python=3.6
conda activate n2n
conda install tensorflow-gpu
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This will install TensorFlow and other dependencies used in this project.

### Preparing datasets for training and validation

This section explains how to prepare a dataset into a TFRecords file for use in training the Noise2Noise denoising network. The image denoising results presented in the Noise2Noise paper have been obtained using a network trained with the ImageNet validation set.

**Training dataset for ImageNet**: To generate the ImageNet validation set tfrecords file, run:

```
# This should run through roughly 50K images and output a file called `datasets/imagenet_val_raw.tfrecords`.
python dataset_tool_tf.py --input-dir "<path_to_imagenet>/ILSVRC2012_img_val" --out=datasets/imagenet_val_raw.tfrecords
```

A successful run of dataset_tool_tf.py should print the following upon completion:

```
<...long omitted...>
49997 ./ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_00002873.JPEG
49998 ./ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_00031550.JPEG
49999 ./ImageNet/ILSVRC2012_img_val/ILSVRC2012_val_00009765.JPEG
Dataset statistics:
  Formats:
    RGB: 49100 images
    L: 899 images
    CMYK: 1 images
  width,height buckets:
    >= 256x256: 48627 images
    < 256x256: 1373 images
```

**Training dataset for BSD300**: If you don't have access to the ImageNet training data, you can experiment with this code using the BSD300 dataset -- just note that the results will be slightly worse than with a network trained on ImageNet.

Download the [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz) images to some folder.  The below example assumes the dataset has been uncompressed into the `./datasets` directory.

Convert the images in BSD300 training set into a tfrecords file:

```
python dataset_tool_tf.py --input-dir datasets/BSDS300-images/BSDS300/images/train --out=datasets/bsd300.tfrecords
```

**Kodak validation set**.  Training tests validation loss against the [Kodak Lossless True Color Image Suite](http://r0k.us/graphics/kodak/) dataset.  Here's how to prepare this dataset for use during training:

```
# Download the kodak validation set from http://r0k.us/graphics/kodak/
python download_kodak.py --output-dir=datasets/kodak
```

### Training networks

To train the noise2noise autoencoder on ImageNet:

```
# try python config.py train --help for available options
python config.py --desc='-test' train --train-tfrecords=datasets/imagenet_val_raw.tfrecords --long-train=true --noise=gaussian
```

You can inspect the training process using TensorBoard:

```
cd results
tensorboard --logdir .
```

By default, this invocation will train a Gaussian denoising network using the ImageNet validation set.  This takes roughly 7.5 hours on an NVIDIA Titan V GPU.

Upon completion, the training process produces a file called `network_final.pickle` under the `results/*` directory.

### Validation using a trained network

Once you've trained a network, you can run a validation dataset through the network:

Suppose your training run results were stored under `results/00001-autoencoder-1gpu-L-n2n`.  Here's how to run a set of images through this network:

```
python config.py validate --dataset-dir=datasets/kodak --network-snapshot=results/00001-autoencoder-1gpu-L-n2n/network_final.pickle
```

## Reproducing Noise2Noise paper results

Here's a summary of training options to reproduce results from the Noise2Noise paper:

| Noise    | Noise2Noise | Command line |
| -----    | ----------- |--------------|
| Gaussian | Yes         | python config.py train --noise=gaussian --noise2noise=true --long-train=true --train-tfrecords=datasets/imagenet_val_raw.tfrecords |
| Gaussian | No          | python config.py train --noise=gaussian --noise2noise=false --long-train=true --train-tfrecords=datasets/imagenet_val_raw.tfrecords |
| Poisson  | Yes         | python config.py train --noise=poisson --noise2noise=true --long-train=true --train-tfrecords=datasets/imagenet_val_raw.tfrecords |
| Poisson  | No          | python config.py train --noise=poisson --noise2noise=false --long-train=true --train-tfrecords=datasets/imagenet_val_raw.tfrecords |

To validate against a trained network, use the following options:

| Noise    | Dataset     | Command line | Expected PSNR (dB)|
| -----    | ----------- |--------------|------------------|
| Gaussian | kodak       | python config.py validate --dataset-dir=datasets/kodak --noise=gaussian --network-snapshot=<.../network_final.pickle> | 32.38 (n2c) / 32.39 (n2n) |
| Gaussian | bsd300      | python config.py validate --dataset-dir=datasets/bsd300 --noise=gaussian --network-snapshot=<.../network_final.pickle> | 31.01 (n2c) / 31.02 (n2n) |
| Poisson  | kodak       | python config.py validate --dataset-dir=datasets/kodak --noise=poisson --network-snapshot=<.../network_final.pickle> | 31.66 (n2c) / 31.66 (n2n) |
| Poisson  | bsd300      | python config.py validate --dataset-dir=datasets/bsd300 --noise=poisson --network-snapshot=<.../network_final.pickle> | 30.27 (n2c) / 30.26 (n2n) |

_Note: When running a validation set through the network, you should match the augmentation noise (e.g., Gaussian or Poisson) with the type of noise that was used to train the network._

## Training on Your Own Noisy Dataset

This section provides a step-by-step guide for training the Noise2Noise network on your own dataset of noisy images.

### Step 1: Prepare Your Dataset

1. Create a directory structure for your dataset:
```bash
mkdir -p datasets/custom_dataset
```

2. Organize your noisy images:
   - Place your noisy images in the `datasets/custom_dataset` directory
   - Supported image formats: PNG, JPEG
   - Recommended image size: at least 256x256 pixels (smaller images will be upscaled)
   - Images should be in RGB format

### Step 2: Convert Dataset to TFRecords

1. Create the TFRecords file from your images:
```bash
python dataset_tool_tf.py --input-dir=datasets/custom_dataset --out=datasets/custom_dataset.tfrecords
```

### Step 3: Configure Training Parameters

1. Create a copy of the config file for your custom training:
```bash
copy config.py config_custom.py
```

2. Modify the following parameters in `config_custom.py`:
   - Set `train_tfrecords = 'datasets/custom_dataset.tfrecords'`
   - Adjust `max_epochs` if needed (default is 300)
   - Modify `minibatch_size` based on your GPU memory
   - Set `corrupt_targets = True` since we're training on naturally noisy images

### Step 4: Start Training

1. Run the training:
```bash
python config_custom.py
```

The training progress will be displayed showing:
- Current epoch
- Training loss
- Time per epoch
- Learning rate

Training outputs will be saved in the `results` directory:
- `network_final.pickle`: The final trained model
- Sample images showing the denoising results

### Step 5: Validation

1. To validate your trained model on new images:
```bash
python config_custom.py validate --dataset-dir=path/to/test/images --network-snapshot=results/[YOUR_TRAINING_DIR]/network_final.pickle
```

### Tips for Better Results

1. Dataset Quality:
   - Use high-quality noisy images
   - Ensure consistent noise levels across your dataset
   - Include diverse image content

2. Training Parameters:
   - Start with default parameters
   - If results are unsatisfactory:
     - Try increasing `max_epochs`
     - Adjust learning rate schedule
     - Modify network capacity

3. Memory Management:
   - If you run into memory issues:
     - Reduce `minibatch_size`
     - Use smaller image crops (adjust in config)

4. Monitoring Training:
   - Check the generated sample images periodically
   - Monitor the training loss for convergence
   - Use validation set to prevent overfitting

## MRI denoising

### Preparing the MRI training dataset

Use the `dataset_tool_mri.py` script to generate training and validation datasets for the N2N MRI case.

**Step #1**: Download the IXI-T1 dataset from: https://brain-development.org/ixi-dataset/.  Unpack to some location.

**Step #2**: Convert the IXI-T1 dataset into a set of PNG files:

```
# Assumes you have downloaded and untarred IXI-T1 under ~/Downloads/IXI-T1.

python dataset_tool_mri.py genpng --ixi-dir=~/Downloads/IXI-T1 --outdir=datasets/ixi-png
```

**Step #3**: Convert a subset of the IXI-T1 dataset into training and validation sets:

```
python dataset_tool_mri.py genpkl --png-dir=datasets/ixi-png --pkl-dir=datasets
```
