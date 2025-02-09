# Dummy class for warprnnt_pytorch
class RNNTLoss:
    def __init__(self, *args, **kwargs):
        print ("Dummy RNNTLoss constructor called.")
        print ("\targs: ", args)
        print ("\tkwargs: ", kwargs)

    def __call__(self, *args, **kwargs):
        print ("Dummy RNNTLoss function called.")
        print ("\targs: ", args)
        print ("\tkwargs: ", kwargs)
        print ("\treturns: 0.0")
        return 0.0