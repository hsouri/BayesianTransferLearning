"""Interface for prior experiments."""

from .prior_ssl import PriorSSL


def Prior(args):
    if args.job == 'prior_SSL':
        return PriorSSL(args)
    else:
        raise NotImplementedError('This prior is not implementd yet!')
    






__all__ = ['Prior']