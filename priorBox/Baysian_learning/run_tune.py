"""Run ray tune for hyper parameters search"""
import tune
from ray.tune import CLIReporter
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from .train_supervised import training_function

def run_ray(args, config, train_loader, test_loader):

    params = {'batch_size': args.batch_size, 'lr': args.lr, 'epochs': args.epochs,
              'momentum': args.momentum,
              'n_cycles': args.n_cycles,
              'n_samples': args.n_samples,
              'temperature': args.temperature,
              'weight_decay': args.weight_decay,
              'prior_scale': args.prior_scale}

    config2 = {'batch_size': tune.choice([32, 64, 128, 256, 512]), 'lr': tune.uniform(0.1, 0.8),
               # 'epochs': tune.choice([200, 300, 600, 800]),
               'epochs': tune.choice([200]),
               'weight_decay': tune.loguniform(1e-6, 1e-2),
               'momentum': tune.uniform(0.85, 0.98),
               'n_cycles': tune.choice([4]),
               'n_samples': tune.choice([12]),
               'temperature': tune.loguniform(1e-5, 10)}

    for name in config2:
        val = config2[name]
        config[name] = val
    # scheduler = ASHAScheduler(
    #    max_t=800,
    #    grace_period=200,
    #    reduction_factor=2)
    callbacks = []
    reporter = CLIReporter(
        parameter_columns=["batch_size", "momentum", "lr", "weight_decay"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])
    # Use bayesian optimisation with TPE implemented by hyperopt
    search_alg = HyperOptSearch(config,
                                metric="loss",
                                mode="min",
                                points_to_evaluate=[params])

    # We limit concurrent trials to 2 since bayesian optimisation doesn't parallelize very well
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    analysis = tune.run(
        tune.with_parameters(
            training_function, train_loader=train_loader, test_loader=test_loader),
        resources_per_trial={
            "cpu": config.cpu_per_trail,
            "gpu": config.gpu_per_trail
        },
        metric="loss",
        mode="min",
        search_alg=search_alg,
        num_samples=args.num_of_samples,
        # scheduler=scheduler,
        progress_reporter=reporter,
        name=args.tune_name,
        local_dir=args.local_dir,
        callbacks=callbacks
    )
