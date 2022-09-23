'''Parse bayesian Learning args'''

import argparse
import os

def str2bool(x):
    return (str(x).lower() == 'true')

def parse_args_SBL(parser):
    # Tune arguments
    parser.add_argument('--tune_name', type=str, default='cifar10', help='The name of the run for ray tune')
    parser.add_argument('--use_tune', action='store_true', help='If we want to use ray tune')
    parser.add_argument('--cpu_per_trail', type=int, default=5, help='cpus for tune')
    parser.add_argument('--gpu_per_trail', type=int, default=1, help='gpus for tune')
    parser.add_argument('--num_of_samples', type=int, default=500, help='Number of samples for tune')
    parser.add_argument('--local_dir', type=str, default='/path/to/your/dir', help='local dir for ray tune')
    # Writers args
    parser.add_argument('--ignore_wandb', action='store_true', help='use wandb for logging')
    parser.add_argument('--save_checkpoints', action='store_true', help='save last checkpoints')
    parser.add_argument('--use_mlflow', action='store_true', help='Use mlflow for logging')
    parser.add_argument('--wandb_name', type=str, default='trail', help='Wandb experiment name')
    parser.add_argument('--wandb_entity', type=str, default='', help='Wandb entity')
    parser.add_argument('--wandb_save_dir', type=str, default='./logs/', help='where to save wandb data')
    parser.add_argument('--wandb_project', type=str, default='prior', help='the project name for logging wandb')
    parser.add_argument('--wandb_key', type=str, default=None,
                        help='The personal key for wandb ')
    # pre weights
    parser.add_argument('--weights_path', type=str,
                        default=None,
                        help='The path for loading the weights')
    parser.add_argument('--weights_path_classifier', type=str,
                        default=None,
                        help='The path for loading the weights of the classifier')
    parser.add_argument('--lightning_ckpt_path', type=str,
                        default=None, help='The path for loading the weights of the lighting checkpoint')
    
    # Configuration
    parser.add_argument('--num_of_labels', type=int, default=10, help='num_classes')
    parser.add_argument('--seed', type=int, default=1, help='Fix seed')
    parser.add_argument('--gpus', default=1, help='number of parallel gpus')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for processing the data')
    parser.add_argument('--g_run', type=str, default='f', help='The cluster to run gauss, local or greence')
    parser.add_argument('--g_path', type=str, default='/path/to/your/dir', help='add it to the path')

    # Prior Hyper params
    parser.add_argument('--is_sgld', action='store_true', help='run SGHMC')
    parser.add_argument('--val_size', type=float, default=0, help='split for validation')
    parser.add_argument('--aug', type=str2bool, default=True, help='Do augmentation')
    parser.add_argument('--samples_dir', type=str, default='samples/', help='where to save the samples')

    parser.add_argument('--load_prior', type=str2bool, default=False, help='If we want to load a prior from path.')
    parser.add_argument('--number_of_samples_prior', type=int, default=5, help='the number of samples for the '
                                                                               'covariance of the prior')

    parser.add_argument('--prior_type', type=str, default='shifted_gaussian', help='The type of the prior - '
                                                                                   'shifted_gaussian, normal or laplace')
    parser.add_argument('--prior_path', type=str
                        , default=None,
                        # ,default = 'models/r50_1x_sk0.pth',
                        help='path should fit this format: prior_path_model.pt, prior_path_mean.pt, prior_path_variance.pt, prior_path_covmat.pt')
    parser.add_argument('--prior_scale', type=float, default=1e10, help='variance for the prior')
    parser.add_argument('--scale_low_rank', type=str2bool, default=True,
                        help='if we scale also the low rank cov matrix')
    parser.add_argument('--prior_eps', type=float, default=1e-1, help='adding to the prior variance')
    parser.add_argument('--run_bma', type=str2bool, default=False, help='run bayesian model averaging at the end')

    # Training
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--momentum', type=float, default=0.95, help='momentum')
    parser.add_argument('--n_cycles', type=int, default=4, help='number of lr annealing cycles')
    parser.add_argument('--n_samples', type=int, default=12, help='number of total samples')
    parser.add_argument('--temperature', type=float, default=2e-8, help='temperature of the posterior')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--encoder', type=str, default='resnet50', help='The network architecture')
    parser.add_argument('--pytorch_pretrain', action='store_true', help='load pytorch pretrained weights')
    

    # Data
    parser.add_argument('--train_dataset', type=str, default='cifar10', help='the dataset to load')
    parser.add_argument('--val_dataset', type=str, default='cifar10', help='the dataset to load')
    parser.add_argument('--data_dir', type=str, default=None, help='path where to download/locate the dataset.')
    parser.add_argument('--train_dir', type=str, default='train', help='subpath where the training data is located.')
    parser.add_argument('--val_dir', type=str, default='val', help='subpath where the validation data is located.')
    parser.add_argument('--server_path', type=str2bool, default=False, help='adjust server path')
    
    # LaplaceApproxPrior
    parser.add_argument('--prior_train_dataset', type=str, default=None, help='the dataset to load')
    parser.add_argument('--prior_val_dataset', type=str, default=None, help='the dataset to load')
    parser.add_argument('--prior_train_dir', type=str, default=None, help='validation folder path')
    parser.add_argument('--prior_val_dir', type=str, default=None, help='train folder path')
    parser.add_argument('--prior_data_dir', type=str, default=None, help='where to save')

    args, unknown = parser.parse_known_args()
    args.backbone_args = {}

    return args
    

