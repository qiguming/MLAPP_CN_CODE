from PreProcessorApplyToTrain import pre_processor_apply_to_train
class Model():
    def __init__(self):pass
def lin_reg_fit_ebnetlab(X, y, pp):
    pp.addOnes = False
    model = Model()
    model.preproc, x = pre_processor_apply_to_train(pp, X)
    targets = y

    
