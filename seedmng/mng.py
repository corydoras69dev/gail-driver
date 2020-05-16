class Singleton(object):
    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance

class SeedMng(Singleton):
    _root_seed = 0
    _system_seed = 0
    _tf_graph_seed = 0
    _tf_system_seed = 0
    _np_seed = 0
    _dummy = 0

    def __init__(self):
        _dummy = 0

    def set_root(self, input):
        self._root_seed = input

    def set_iteration(self, n_iter):
        self._system_seed = self._root_seed + n_iter * 110
        self._tf_graph_seed = self._root_seed + n_iter * 130
        self._tf_system_seed = self._root_seed + n_iter * 170
        self._np_seed = self._root_seed + n_iter * 190

    def get_system_seed(self, id = 0):
        return self._system_seed + id

    def get_tf_graph_seed(self, id = 0):
        return self._tf_graph_seed + id

    def get_tf_system_seed(self, id = 0):
        return self._tf_system_seed + id

    def get_np_seed(self, id = 0):
        return self._np_seed + id
