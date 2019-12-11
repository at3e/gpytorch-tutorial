"""
Regression with GP
https://gpytorch.readthedocs.io/en/latest/examples/01_Simple_GP_Regression/Simple_GP_Regression.html
"""
import math
import torch
import gpytorch
import visdom

# plot
vis = visdom.Visdom()

# data
train_x = torch.linspace(0, 1, 100)
train_y = torch.sin(train_x * 2 * math.pi) + torch.randn(train_x.size()) * 0.2
vis.line(train_y, train_x, opts={'title': 'train data'})


# Def of Model
class Model(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = Model(train_x, train_y, likelihood)


# Training
model.train()
likelihood.train()
opt = torch.optim.Adam(model.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)  # marginal log likelihood
num_iter = 50
for i in range(num_iter):
    opt.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print(f"Iter {i + 1}/{num_iter} - Loss: {loss.item():.3f}"
          f"  lengthscale: {model.covar_module.base_kernel.lengthscale.item():.3f}"
          f"  noise: {model.likelihood.noise.item():.3f}")
    opt.step()

# Testing
test_x = torch.linspace(0, 2, 200)
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    pred = likelihood(model(test_x))
    vis.line(pred.mean.numpy(), test_x, opts={'title': 'testing'})


lower, upper = pred.confidence_region()
vis.line(X=test_x, Y=torch.stack((pred.mean, lower, upper)).t(),
         opts={'title': 'prediction', 'legend': ['mean', 'lower', 'upper']})
