# Pre-Train Your Loss! High-Performance Transfer Learning with Bayesian Neural Networks and Pre-Trained Priors

This repository contains an easy-to-use PyTorch implementation of methods described in [Pre-Train Your Loss! High-Performance Transfer Learning with Bayesian Neural Networks and Pre-Trained Priors](https://arxiv.org/abs/2205.10279) by [Ravid Shwartz-Ziv](https://www.ravid-shwartz-ziv.com/), [Micah Goldblum](https://goldblum.github.io/), [Hossein Souri](https://hsouri.github.io/), [Sanyam Kapoor](https://sanyamkapoor.com/), [Chen Zhu](https://zhuchen03.github.io/), [Yann LeCun](http://yann.lecun.com/), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).
[![Preview](/loss_surface.png)](https://arxiv.org/abs/2205.10279)

## Summary 

**Idea:** We can transfer much more than an initialization. Knowledge of the source task should affect the **locations** and **shape** of optima on the downstream task.

**Approach:** Infer a posterior on the source task to re-scale as an **informative prior** on the downstream task.

**Results:** Significantly improved performance over standard transfer learning and fine tuning, with minimal overhead.

## Overview

Our Bayesian transfer learning framework transfers knowledge from pre-training to downstream tasks. To up-weight parameter settings consistent with a pre-training loss function, we fit a probability distribution over the parameters of feature extractors to a pre-training loss function and rescale it as a prior. By adopting a learned prior, we alter the downstream loss surface and its optimal locations. By contrast, typical transfer learning methods only use a pre-trained initialization. 

[![Preview](https://github.com/hsouri/BayesianTransferLearning/blob/main/fig.png)](https://arxiv.org/abs/2205.10279)

Our Bayesian transfer learning pipeline uses only easy-to-implement existing tools. In our experiments, Bayesian transfer learning outperforms both SGD-based transfer learning and non-learned Bayesian inference. A schematic of our framework is found below.  
This repo contains the code for extracting your prior parameters and applying them to a downstream task using Bayesian inference. The downstream tasks include both image classification and image segmentation.


### Dependencies:

- torch >= 1.8.1
- torchvision >= 0.9.1
- pytorch-lightning >= 1.4.7

For the complete list of requirements see `requirements.txt`.


### Prepare Datasets:

For your convenience, we have provided the python scripts for downloading and organizing the `Oxford Flowers 102` and `Oxford-IIIT Pet` datasets. The python scripts can be found [here](https://github.com/hsouri/BayesianTransferLearning/tree/main/Prapare%20Data).



### Usage:
Use `prior_run_jobs.py` both to learn priors from pre-trained checkpoints and also to perform inference on downstream tasks. 
```bash

python prior_run_jobs.py --job=<JOB> \
                         --prior_type=<PRIOR_TYPE> \
                         --data_dir=<DATA_DIR> \
                         --train_dataset=<TRAIN_DATASET> \
                         --val_dataset=<VAL_DATASET> \
                         --pytorch_pretrain=<PYTORCH_PRETRAIN> \ 
                         --prior_scale=<PRIOR_SCALE> \ 
                         --num_of_train_examples=<NUM_OF_TRAIN_EXAMPLES> \ 
                         --weights_path=<WEIGHTS_PATH> \ 
                         --number_of_samples_prior=<NUMBER_OF_SAMPLES_PRIOR> \ 
                         --encoder=<ENCODER> \ 

```

Parameters:

* ```JOB``` -  set `prior` to learn a prior or `supervised_bayesian_learning` to perform inference on downstream tasks. 

* ```PRIOR_TYPE``` --type of prior used for inference on a downstream task:

              - `normal` - zero-mean isotropic Gaussian prior
              - `shifted_gaussian` - learned prior
* ```PRIOR_PATH``` - path for the file to load the learned prior. The file should contain mean, variance, and cov_factor fields
* ```DATA_DIR```  -  path which contains the data
* ```TRAIN_DATASET```  - dataset for training
* ```VAL_DATASET```  - dataset for validation
* ```PYTORCH_PRETRAIN```  - if we would like to load the weights from a torchvision pretrained model
* ```PRIOR_SCALE```  - parameter for re-scaling the prior
* ```NUM_OF_TRAIN_EXAMPLES```  - number of training samples on which to train our model
* ```WEIGHTS_PATH```  - path for loading pre-train weights
* ```NUMBER_OF_SAMPLES_PRIOR``` - number of samples for fitting the covariance of the prior
* ```ENCODER``` - base network architecture. The options include most models supported by torchvision.

For the full list of arguments, see `priorBox/options.py`. All optional arguments for Bayesian learning are listed [here](https://github.com/hsouri/BayesianTransferLearning/blob/main/priorBox/bayesian_learning/args.py) and optional arguments for learning a prior are listed [here](https://github.com/hsouri/BayesianTransferLearning/blob/main/priorBox/solo_learn/args/setup.py).


### Our Pre-Trained Priors:
Our learned priors can be found [here](https://drive.google.com/drive/folders/1FbnUsL_CRWORjlTyX8dtHRzcGFeaE4Iz?usp=sharing). The priors include torchvision ResNet-50 and ResNet-101 as well as SimCLR ResNet-50, all trained on ImageNet.  To use these for downstream tasks, pass the argument `--prior_path` along with the path for the prior when running `prior_run_jobs.py`. 

### How to Cite:
```bibtex
@article{shwartz2022pre,
  title={Pre-Train Your Loss: Easy Bayesian Transfer Learning with Informative Priors},
  author={Shwartz-Ziv, Ravid and Goldblum, Micah and Souri, Hossein and Kapoor, Sanyam and Zhu, Chen and LeCun, Yann and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:2205.10279},
  year={2022}
}
```
