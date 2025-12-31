# DataMatrix Image Reconstruction to Enhance Decodability

This repository is a part of the Diploma thesis 
<https://is.muni.cz/th/ppu25/dp-dmcodes-thesis.pdf>

#### Abstract

Data Matrix (DM) codes are two-dimensional visual patterns used to encode textual data, widely utilized in industrial applications such as parcel tracking or quality control due to their high data capacity and robust error correction. However, DM codes often face readability challenges due to visual defects like poor image quality, blur, and reflections. This thesis aims to compare various encoder-decoder neural network models to enhance the DM code image quality before decoding it with a code reader. For this purpose, we create two datasets, one synthetically generated for training and the second consisting of photos of DM codes taken from a real setting for evaluation. Our experiments systematically explore different architecture and encoder variants to achieve the highest decode rate and further examine the impact of network size on accuracy. Lastly, we measure the inference time of all trained models. The results provide valuable insights into model selection for DM code image enhancement.

## Synthetically generated dataset

Synthetic dataset used for training is created by a sophisticated data generation and augmentation pipeline. For every synthetically generated DM code we have a triplet: encoded **string**, clean **target DM code** created by encoding the string, augmented **corrupted DM code** created by performing augmentations on the target DM code. 

![image](https://github.com/shortery/dp-dm-codes/assets/70586740/4adba6e0-3dd1-4146-818a-37dc9568af0f)


## Real dataset

Real dataset used for evaluation is created by manually collecting and annotating 300 real-world images. We publish this dataset on HuggingFace <https://huggingface.co/datasets/shortery/dm-codes>.

![image](https://github.com/shortery/dp-dm-codes/assets/70586740/e5e6bf2f-44b0-4a9d-a02d-d11b1050ffa9)

## How to train
Training data is generated on the fly and the real validation dataset is automatically downloaded from HuggingFace. However, you need to generate a synthetic validation dataset using this command:
```
python dm_codes/scripts/create_synthetic_dataset.py
```

To run the training, use:
```
python dm_codes/scripts/train.py
```
Training parameters can be specified in `config.yaml`. The training will be logged in wandb and checkpoints will be saved in the `checkpoints` folder.

## How to evaluate
To evaluate on the real test dataset, run:
```
python dm_codes/scripts/measure_accuracy.py <checkpoint_path.ckpt> <saved_accuracies_path.json>
```
where the metrics will be saved into `saved_accuracies_path.json`.


## Dependencies

Dependencies are specified in requirements.txt


## Citing
```
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
