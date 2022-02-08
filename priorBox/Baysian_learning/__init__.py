"""Interface for Baysian learning experiments."""

from .Baysian_supervised import BaysianSupervised



def Baysian_learner(args):
    if args.job == 'supervised_Baysian_lerning':
        return BaysianSupervised(args)
    else:
        raise NotImplementedError('This Baysian learning is not implementd yet!')
    



__all__ = ['Baysian_learner']