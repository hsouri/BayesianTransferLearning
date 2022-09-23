"""The main supervised class for running bayesian deep learning model (including ray tune)"""

import os
from .bayesian_base import _bayesianBase
from .train_supervised import training_function
from .run_tune import run_ray
from ..solo_learn.utils.classification_dataloader import prepare_data



class bayesianSupervised(_bayesianBase):

    def __init__(self, args):
        super(bayesianSupervised, self).__init__(args)


    def learn(self):
        config = self.get_config(self.args)
        train_loader, test_loader, config['N'], config['num_of_batches'] = prepare_data(
            train_dataset=config['train_dataset'],
            val_dataset=config['val_dataset'],
            data_dir=config['data_dir'],
            train_dir=config['train_dir'],
            val_dir=config['val_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
        )

        if self.args.use_tune:
            run_ray(self.args, config, train_loader, test_loader)
        else:
            training_function(config, train_loader, test_loader)




    def get_config(self, args):
        # Set wandb
        if not args.ignore_wandb:
            os.environ['WANDB_API_KEY'] = args.wandb_key
        config = vars(args)
        # Adjust the paths to the server
        if args.server_path:
            paths = ['local_dir', 'data_dir', 'weights_path', 'wandb_save_dir', 'samples_dir', 'prior_path']
            if args.g_run == 'greence':
                args.g_path = '/scratch/rs8020/'
            elif args.g_run == 'gauss':
                args.g_path = '/path/to/your/dir'
            elif args.g_path == 'local':
                args.g_path = '/data2/ssl_priors'
            else:
                args.g_path = '/path/to/your/dir'
            for path in paths:
                config[path] = args.g_path + config[path]

        return config