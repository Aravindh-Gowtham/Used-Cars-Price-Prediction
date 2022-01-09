#* Parent class of all components

class DataModeler:
    def __init__(self):
        pass

    def _parametrized(dec):
        def layer(*args, **kwargs):
            def repl(f):
                return dec(f, *args, **kwargs)
            return repl
        return layer

    @staticmethod
    @_parametrized
    def logger(f, job):
        def aux(self, *xs, **kws):
            print("="*50)
            print(job + "...")
            res = f(self, *xs, **kws)
            print(job + " is done")
            print("="*50)
            return res
        return aux