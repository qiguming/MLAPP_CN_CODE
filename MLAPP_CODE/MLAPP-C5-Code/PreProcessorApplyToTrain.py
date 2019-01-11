from CenterCols import center_cols
from MakeUnitVariance import make_unit_variance
from RescaleData import rescale_data
from DegExpand import deg_expand
from AddOnes import add_ones
def pre_processor_apply_to_train(preproc, X):
    if preproc.standardizeX:
        X, preproc.Xmu = center_cols(X)
        X, preproc.Xstd = make_unit_variance(X)
    if preproc.rescaleX:
        X, _, _ = rescale_data(X)
    if len(preproc.kernelFn) != 0:
        pass        #  核方法待补充
    if len(preproc.poly) != 0:
        assert(preproc.poly > 0)
        X = deg_expand(X, preproc.poly, addOnes=False)
    if preproc.addOnes:
        X = add_ones(X)
    
    return X,preproc 

    