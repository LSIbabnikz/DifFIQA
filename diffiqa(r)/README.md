
# __DifFIQA(R) - Extended Regression Approach__ 

 - This directory includes training and inference scripts for the regression approach proposed by [DifFIQA: Face Image Quality Assessment Using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2305.05768) as well as the pretrained model weights.

---


## __Table of Contents__

  - [**1. Environment Setup**](#1-environment-setup)

  - [**2. Training**](#2-training)

  - [**3. Inference**](#3-inference)

  - [**4. Model Weights**](#4-model-weights)


---


## __1. Environment Setup__

__Before running training or evaluation you also need to obtain the pretrained model weights for the used FR model.__

- The weights for the CosFace model used in the paper are available at the [official repository](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). 

- The correct backbone is available under the __glint360k\_cosface\_r100\_fp16\_0.1__ directory.

- Place the file under __./model_weights__ and rename it to _"weights.pth"_.

---



## __1. Training__

__To train the DifFIQA(R) regression approach first download the [VGGFace2 quality scores](https://unilj-my.sharepoint.com/:u:/g/personal/zb4290_student_uni-lj_si/EdfMHzcYnDhBjA8IkuExVH0B6XNoiFQmn38dO_J2dTO24Q?e=hxcM0G) used in the paper or calculate your own using the DifFIQA diffusion approach.*.__

- Place the quality score file in "./quality_scores" and name it "vggface2-qs.pkl".

__Then configure the training configuration file *./configs/train_config.yaml*.__

- Alter the save location (_save\_path_) where you wish to save the trained model.

- Alter the dataset image location (_image\_loc_) to point to the cropped VGGFace2 dataset.

- You are free to alter also the rest of the provided configuration parameters, at your own risk.

__Once you have configured the parameters run the training script.__

> python train.py -c ./configs/train_config.yaml

---


## __2. Inference__

__To perform quality score inference you will need to either train your own regression model as described [above](#2-training) or download the pretrained model used in the paper, as described [below](#4-model-weights).__

__Once you have a pretrained regression model you can alter the inference script ./configs/inference_config.yaml.__

- Alter the image folder location (_dataset.loc_) to point to the dataset for which you want to extract quality scores.

- Alter the model weights location (_model.weights_) to point to the pretrained weights of the regression model.

- Alter the save location (_base.save\_path_) to the location where you want to save the extracted quality scores.

__Once you have configured the parameters run the inference script.__

> python inference.py -c ./configs/inference_config.yaml

---

## __3. Model Weights__

__We provide the pretrained regression model weights used in the paper [here](https://unilj-my.sharepoint.com/:u:/g/personal/zb4290_student_uni-lj_si/EdUoWI2DxdFAsgr1qz_Dna4Bz5EEesU407_6yYe7HXYCUQ?e=XRahXp).__

- Download the provided weights and copy them into ./model_weights.

- Configure the inference file to use the downloaded weights.

- Run the inference as described [above](#3-inference).


