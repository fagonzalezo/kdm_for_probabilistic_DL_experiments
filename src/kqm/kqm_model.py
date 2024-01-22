from kqm import KQMUnit, RBFKernelLayer, pure2dm, dm2discrete, Model
class KQMModel(Model):
    def __init__(self, encoded_size, encoder, n_comp):
        super().__init__() 
        self.encoded_size = encoded_size
        self.encoder = encoder
        self.n_comp = n_comp
        self.kernel = RBFKernelLayer(sigma=0.1, dim=encoded_size)
        self.kqm_unit = KQMUnit(kernel=self.kernel, dim_x=encoded_size, 
                                dim_y=10, n_comp=n_comp)
        
    def call(self, input):
        encoded = self.encoder(input)
        rho_x = pure2dm(encoded)
        rho_y = self.kqm_unit(rho_x)
        probs = dm2discrete(rho_y)
        return probs
