import torch
from torch import nn
from torch.autograd import Variable
from ..module import Module
from ..random_variables import GaussianRandomVariable
from ..lazy import LazyVariable, RootLazyVariable, AddedDiagLazyVariable, DiagLazyVariable


class AbstractVariationalGP(Module):
    def __init__(self, inducing_points, rank=16):
        super(AbstractVariationalGP, self).__init__()
        if not torch.is_tensor(inducing_points):
            raise RuntimeError('inducing_points must be a Tensor')
        n_inducing = inducing_points.size(0)
        self.register_buffer('inducing_points', inducing_points)
        self.register_buffer('variational_params_initialized', torch.zeros(1))
        self.register_parameter('variational_mean', nn.Parameter(torch.zeros(n_inducing)), bounds=(-1e4, 1e4))
        self.register_parameter('root_variational_covar',
                                nn.Parameter(torch.zeros(n_inducing, rank)), bounds=(-100, 100))
        self.register_parameter('log_variational_diag',
                                nn.Parameter(torch.ones(n_inducing).mul_(-1)), bounds=(-100, 100))
        self.register_variational_strategy('inducing_point_strategy')

    def marginal_log_likelihood(self, likelihood, output, target, n_data=None):
        from ..mlls import VariationalMarginalLogLikelihood
        if not hasattr(self, '_has_warned') or not self._has_warned:
            import warnings
            warnings.warn("model.marginal_log_likelihood is now deprecated. "
                          "Please use gpytorch.mll.VariationalMarginalLogLikelihood instead.")
            self._has_warned = True
        if n_data is None:
            n_data = target.size(-1)
        return VariationalMarginalLogLikelihood(likelihood, self, n_data)(output, target)

    def covar_diag(self, inputs):
        if inputs.ndimension() == 1:
            inputs = inputs.unsqueeze(1)
        orig_size = list(inputs.size())

        # Resize inputs so that everything is batch
        inputs = inputs.unsqueeze(-2).view(-1, 1, inputs.size(-1))

        # Get diagonal of covar
        res = super(AbstractVariationalGP, self).__call__(inputs)
        covar_diag = res.covar()
        if isinstance(covar_diag, LazyVariable):
            covar_diag = covar_diag.evaluate()
        covar_diag = covar_diag.view(orig_size[:-1])

        return covar_diag

    def prior_output(self):
        res = super(AbstractVariationalGP, self).__call__(Variable(self.inducing_points))
        if not isinstance(res, GaussianRandomVariable):
            raise RuntimeError('%s.forward must return a GaussianRandomVariable' % self.__class__.__name__)
        return res

    def variational_output(self):
        root_variational_covar = self.root_variational_covar
        variational_covar = AddedDiagLazyVariable(RootLazyVariable(root_variational_covar), DiagLazyVariable(self.log_variational_diag.exp()))
        return GaussianRandomVariable(self.variational_mean, variational_covar)
