"""Baysian learning base class"""



class _BaysianBase:


    def __init__(self, args):
        self.args = args

    def learn(self):
        '''Baysian learning on a down stream task using the loaded prior'''
        raise NotImplementedError()
