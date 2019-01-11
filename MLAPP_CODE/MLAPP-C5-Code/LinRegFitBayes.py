from PreProcessorCreate import pre_processor_create
from LinRegFitBayesGaussPrior import lin_reg_fit_bayes_gauss_prior
from LinRegFitEbNetlab import lin_reg_fit_ebnetlab
def line_fit_bayes(X, y, **varargin):
    """实现贝叶斯线性回归模型
    p(y|x)=N(y|w*[1 x],(1/beta)) 式中beta为测量值的精度
    
    Input:
    proir：先验分布，如下几种之一：
        'uninf':  Jeffrey 先验，关于参数w和beta
        'vb':     变分贝叶斯（待定）
        'eb':     经验贝叶斯（证据程序）
        'gauss':  在w上使用N(0, (1/alpha)I)的先验分布，必须制定alpha和beta
        'zellner': 待定
    useArd:
        待定
    preproc:
        预处理器
    
    Output:
    model:         包含后验分布的参数
    logev:         对数边缘似然
    post_summary:  后验分布的总结
    """
    prior = str.lower(varargin.get('prior','uninf'))  # 默认先验分布为均匀先验
    preproc = varargin.get('preproc', pre_processor_create(addOnes=True, standardizeX=False))
    beta = varargin.get('beta', [])
    alpha = varargin.get('alpha', [])
    g = varargin.get('g', [])
    useARD = varargin.get('useARD', False)
    displaySummary = varargin.get('displaySummary', False)
    names = varargin.get('names', [])

    if prior == 'eb':
        prior = 'ebnetlab'
    
    if prior == 'gauss':
        model, logev = lin_reg_fit_bayes_gauss_prior(X, y, alpha, beta, preproc)
    elif prior == 'ebnetlab':
        model, logev = lin_reg_fit_ebnetlab(X, y, preproc)
