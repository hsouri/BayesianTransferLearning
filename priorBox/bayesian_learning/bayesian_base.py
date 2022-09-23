"""bayesian learning base class"""



class _bayesianBase:


    def __init__(self, args):
        self.args = args

    def learn(self):
        '''bayesian learning on a down stream task using the loaded prior'''
        raise NotImplementedError()
