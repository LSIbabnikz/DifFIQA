
# __DifFIQA - Base Approach__ 

 - This directory includes training and inference scripts for the base approach proposed by [DifFIQA: Face Image Quality Assessment Using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2305.05768) as well as the pretrained model weights.

---


## __Table of Contents__

  - [**1. Environment Setup**](#1-environment-setup)
 
  - [**2. Training**](#2-training)

  - [**3. Inference**](#3-inference)

  - [**4. Model Weights**](#4-model-weights)

  - [**5. Acknowledgement**](#5-acknowledgement)

---

## __1. Environment Setup__

__Before running training or inference you also need to obtain the pretrained model weights for the used FR model.__

- The weights for the CosFace model used in the paper are available at the [official repository](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch). 

- The correct backbone is available under the __glint360k\_cosface\_r100\_fp16\_0.1__ directory.

- Place the file under __./model_weights__ and rename it to _"weights.pth"_.


---



## __2. Training__

__To train the DifFIQA diffusion approach first configure the training configuration file *./configs/train_config.yaml*.__

- Alter the folder name (_trainin_params.folder_) where the training images can be found.
> The official implementaion uses the CelebA dataset to train the model, but other datasets can easily be used.

- Alter the results folder to where you want to save the results of trainig.

- You are free to alter also the rest of the provided configuration parameters, at your own risk.

__Once you have configured the parameters run the training script.__

> python train.py -c ./configs/train_config.yaml

---


## __3. Inference__

__To perform quality score inference you will need to either train your own diffusion model as described [above](#2-training) or download the pretrained model used in the paper, as described [below](#4-model-weights).__

__Once you have a pretrained diffusion model you can alter the inference script ./configs/inference_config.yaml.__

- Alter the image folder location (_images\_loc_) to point to the dataset for which you want to extract quality scores.

- Alter the model location (_model\_loc_) to point to the pretrained weights of the diffusion model.

- Alter the save location (_save\_loc_) to the location where you want to save the extracted quality scores.

__Additionally, if you did not train your own model, you also need to alter the training script ./configs/train_config.yaml.__

- Alter the image folder location (_trainer_params.folder_) to point to the inference dataset.

__Once you have configured the parameters run the inference script.__

> python inference.py -c ./configs/inference_config.yaml


---

## __4. Model Weights__

__We provide the pretrained diffusion model weights used in the paper [here](https://unilj-my.sharepoint.com/:u:/g/personal/zb4290_student_uni-lj_si/ESPBeGxKtE1Dt-Nh0qL6lWUBz_dFLMOKIrbq4gSRsEHeaw?e=0A7aXQ).__

- Download the provided weights and copy them into ./model_weights.

- Configure the inference file to use the downloaded weights.

- Run the inference as described [above](#3-inference).



---

## __5. Acknowledgement__

 - Special thanks to [lucidrains](https://github.com/lucidrains) for the open-source implementation of [diffusion models](https://github.com/lucidrains/denoising-diffusion-pytorch)!

 - Special thanks to [cszn](https://github.com/cszn) for the open-source implementaion of the [BSRGAN degradation process](https://github.com/cszn/BSRGAN)!

