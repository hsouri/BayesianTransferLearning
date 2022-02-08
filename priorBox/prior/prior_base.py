"""Prior base class as an interface for all prior jobs"""



class _PriorBase:


    def __init__(self, args) -> None:
        self.args = args

    def learn_prior(self):
        '''learn the prior'''
        raise NotImplementedError()
