# DataMatrix Image Reconstruction to Enhance Decodability

This repository implements training and evaluation of neural networks for **enhancing the image quality of DM codes** before decoding them with a code reader.

The trained models remove visual defects, such as blur, reflections, and distortions, which improves decodability. Models are trained on synthetically generated data and evaluated separately on synthetic and real-world photos of DM codes.

This repository is a part of my [Diploma thesis](https://is.muni.cz/th/ppu25/dp-dmcodes-thesis.pdf), which compares various neural network architectures for enhancing DM codes.


## How to use the repository

### Setup

To setup the environment, run the following commands:
```shell
git clone ...
conda env create -f conda-env.yaml
conda activate dm-codes
pip install -r requirements.txt
pip install -e .
```

### How to train

Training data is generated on the fly, and the real-world validation dataset is automatically downloaded from HuggingFace. However, you also need to generate a synthetic validation dataset using this command:

```shell
python dm_codes/scripts/create_synthetic_dataset.py
```

For more details about data, see the [section below](#about-the-data).

To run the training, use:

```shell
python dm_codes/scripts/train.py
```

Training parameters, including model architecture, can be specified in `config.yaml`. The training will be logged into wandb and checkpoints will be saved in the `checkpoints` folder.

### How to evaluate

To evaluate a checkpoint on the real test dataset, run:

```shell
python dm_codes/scripts/measure_accuracy.py <checkpoint_path.ckpt> <saved_accuracies_path.json>
```

The metrics will be saved to `saved_accuracies_path.json`.

## About the Data

### Synthetically generated dataset

The synthetic dataset used for training is created by a data generation and augmentation pipeline. For every synthetically generated DM code, we have a triplet: encoded **string**, clean **target DM code** created by encoding the string, and augmented **corrupted DM code** created by performing augmentations on the target DM code. 

![image](https://github.com/shortery/dp-dm-codes/assets/70586740/4adba6e0-3dd1-4146-818a-37dc9568af0f)


### Real dataset

The real dataset used for evaluation is created by manually collecting and annotating 300 real-world images. The dataset is [published on HuggingFace](https://huggingface.co/datasets/shortery/dm-codes).

![image](https://github.com/shortery/dp-dm-codes/assets/70586740/e5e6bf2f-44b0-4a9d-a02d-d11b1050ffa9)


## Results

The trained neural networks can **significantly improve** the visual quality and decodability of degraded Data Matrix images. Below are several examples showing the input corrupted real-world images and their corresponding enhanced outputs. 

| Corrupted Input \| Enhanced Output | Corrupted Input \| Enhanced Output |
|--------------------------------|--------------------------------|
| ![image](https://github.com/user-attachments/assets/a9d7dd11-c9c0-452a-9ac4-ec87e0cf04ef) | ![image](https://github.com/user-attachments/assets/3be3ef69-7699-4eb9-a8bf-6e9e6b7b2eda) |
| ![image](https://github.com/user-attachments/assets/e71ab24e-c33b-4220-a530-fb79eb8b947a) | ![image](https://github.com/user-attachments/assets/61abc84a-692f-4a6a-bc55-f144a2492974) |
| ![image](https://github.com/user-attachments/assets/c1ddc19a-0468-4111-82fd-b01186171ae3) | ![image](https://github.com/user-attachments/assets/8405c435-d93b-4177-8453-30220f60d85d) |


### Quantitative Findings

In my thesis, I trained **19 models** by combining four architectures (UNet, UNet++, LinkNet, MANet) with four encoder families (ResNet, EfficientNet, DenseNet, MobileNet). 

The decode rate and median inference time on the GPU for different models are shown in the following scatter plot. Bubble sizes are proportional to the number of trainable parameters of a model.

![image](https://github.com/user-attachments/assets/2127e523-706b-4321-a298-2da0c1c74e5d)

All models showed **significant improvements** in decode rates and reductions in misread rates when compared to decoding without image enhancement.

- The UNet and UNet++ architectures demonstrate the highest decode rate on the real test dataset, with an **average of 52.76% compared to 31.66% without the network**, followed by LinkNet architectures, while MANet architectures achieve the lowest decode rates.
- The best-performing model was UNet++ with an EfficientNet-B6 encoder, achieving **57.29% (1.81× improvement over baseline)** decode rate.

For more detailed comparism, see my [Diploma thesis](https://is.muni.cz/th/ppu25/dp-dmcodes-thesis.pdf). 

## Citation

```bibtex
@thesis{dmcodes-thesis,
  author = {Petra Krátká},
  title = {DataMatrix Image Reconstruction to Enhance Decodability},
  address = {Brno},
  year = {2024},
  school = {Masaryk University, Faculty of Informatics},
  type = {Diploma thesis},
  url = {https://is.muni.cz/th/ppu25/dp-dmcodes-thesis.pdf},
}
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
