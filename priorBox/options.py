"""Prior general options. More detail options will be set for each specefic prior in their class."""

import argparse
from .solo_learn.args.setup import parse_args_pretrain
from .Baysian_learning.args import parse_args_SBL




def options():
    """Parse args."""
    parser = argparse.ArgumentParser(description='Construct prior options.')

    parser.add_argument('--job', required=True, type=str, 
                        choices=['prior_SSL', 'supervised_bayesian_learning'])

    known_args, _ = parser.parse_known_args()

    if known_args.job == 'prior_SSL':
        # Parse SSL posterior training args
        args = parse_args_pretrain(parser)
    elif known_args.job == 'supervised_bayesian_learning':
        # Parse Bayesian learning args
        args = parse_args_SBL(parser)
    else:
        raise NotImplementedError

    return args

