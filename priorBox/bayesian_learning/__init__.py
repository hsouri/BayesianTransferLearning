"""Interface for bayesian learning experiments."""

from .bayesian_supervised import bayesianSupervised



def bayesian_learner(args):
    if args.job == 'supervised_bayesian_learning':
        return bayesianSupervised(args)
    else:
        raise NotImplementedError('This bayesian learning is not implementd yet!')
    



__all__ = ['bayesian_learner']