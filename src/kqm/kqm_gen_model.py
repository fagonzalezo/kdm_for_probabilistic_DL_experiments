from tensorflow.keras import Model
from kqm import KQMUnit
from kqm import CosineKernelLayer
from kqm import pure2dm


class KQMGenModel(Model):
    def __init__(self, encoded_size, n_comp):
        super().__init__() 
        self.encoded_size = encoded_size
        self.n_comp = n_comp
        self.kernel = CosineKernelLayer()
        self.kqm_unit = KQMUnit(kernel=self.kernel, dim_x=10, 
                                dim_y=encoded_size, n_comp=n_comp)
        
    def call(self, input):
        rho_y = pure2dm(input)
        rho_x = self.kqm_unit(rho_y)
        return rho_x
