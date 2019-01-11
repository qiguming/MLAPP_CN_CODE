class preprocessor(object):
    __slots__ = ('standardizeX','rescaleX','kernelFn','poly','addOnes','Xscale')
    def __init__(self):pass
def pre_processor_create(**varargin):
    """创建预处理器类，该类的每个属性记录着当
    前哪些预处理方法是需要的。
    Input:
    字典形式
    Output:
    预处理器类实例
    """
    pp = preprocessor()
    pp.standardizeX = varargin.get('standardizeX', False)
    pp.rescaleX     = varargin.get('rescaleX ', False)
    pp.Xscale       = [-1, 1]
    pp.kernelFn     = varargin.get('kernelFn', [])
    pp.poly         = varargin.get('poly', [])
    pp.addOnes      = varargin.get('addOnes', False)
    return pp

    