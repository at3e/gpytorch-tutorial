"""
Classification with GP
https://gpytorch.readthedocs.io/en/latest/examples/02_Simple_GP_Classification/Simple_GP_Classification.html

This works with the latest version.
And the learning is unstable (it may be success if the loss get less than 0.2).
"""
import math
import torch
import gpytorch
import visdom

vis = visdom.Visdom()

# data
train_x = torch.linspace(0, 1, 10)
train_y = torch.sign(torch.cos(train_x * 4 * math.pi)).add(1).div(2)
vis.line(X=train_x, Y=train_y, opts=dict(title='train data', markers=True))


# Def of Model
class Model(gpytorch.models.ApproximateGP):
    def __init__(self, train_x):
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(self, train_x, variational_dist)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred


model = Model(train_x)
likelihood = gpytorch.likelihoods.BernoulliLikelihood()


# Training
model.train()
likelihood.train()
opt = torch.optim.Adam(model.parameters(), lr=0.1)

mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())
num_iter = 50
for i in range(num_iter):
    opt.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print(f"Iter {i + 1}/{num_iter} - Loss: {loss.item():.3f}")
    opt.step()


# Testing
model.eval()
likelihood.eval()

with torch.no_grad():
    test_x = torch.linspace(-1, 2, 301)
    pred = likelihood(model(test_x))
    mean = pred.mean
    labels = pred.mean.ge(0.5).float()
    vis.line(X=test_x, Y=torch.stack((pred.mean, labels)).t(), opts=dict(title='prediction'))
