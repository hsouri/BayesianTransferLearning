# Pre-Train Your Loss! High-Performance Transfer Learning with Bayesian Neural Networks and Pre-Trained Priors

This repository contains an easy-to-use PyTorch implementation of methods described in [Pre-Train Your Loss! High-Performance Transfer Learning with Bayesian Neural Networks and Pre-Trained Priors](https://github.com/hsouri/BayesianTransferLearning) by [Ravid Shwartz-Ziv](https://www.ravid-shwartz-ziv.com/), [Micah Goldblum](https://goldblum.github.io/), [Hossein Souri](https://hsouri.github.io/), [Sanyam Kapoor](https://sanyamkapoor.com/), [Chen Zhu](https://zhuchen03.github.io/), [Yann Lecun](http://yann.lecun.com/), and [Andrew Gordon Wilson](https://cims.nyu.edu/~andrewgw/).

The Bayesian transfer learning framework modifies the loss surface on downstream tasks, unlike traditional transfer learning approaches which merely adopt a pre-trained initialization.  In our experiments, Bayesian transfer learning outperforms both SGD based transfer learning and also Bayesian inference with non-learned priors.  A schematic of our framework is found below. 

### How to use:
Use `prior_run_jobs.py` both to learn priors from pre-trained checkpoints and also to perform inference on downstream tasks.  For a list of arguments, see `priorBox/options.py`.

### Our Pre-Trained Priors:
Our learned priors can be found [here](https://drive.google.com/drive/folders/1FbnUsL_CRWORjlTyX8dtHRzcGFeaE4Iz?usp=sharing). The priors include torchvision ResNet-50 and ResNet-101 as well as SimCLR ResNet-50, all trained on ImageNet.  To use these for downstream tasks, pass the argument `--prior_path` along with the path for the prior when running `prior_run_jobs.py`. 

[![Preview](https://github.com/hsouri/BayesianTransferLearning/blob/main/fig.png)](https://github.com/hsouri/BayesianTransferLearning)
