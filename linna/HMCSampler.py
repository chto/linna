import torch
import numpy as np
from tqdm.auto import tqdm


class HMCSampler:
    def __init__(self, lnP, x0, m, transform=None, device='cpu'):
        self.lnP = lnP
        self.x0 = x0.to(dtype=torch.float32, device=device)
        self.x = self.x0.clone()
        self.m = m.to(dtype=torch.float32, device=device)
        self.device = device

        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform

    def sample(self, num_samps, num_steps, step_size):
        chain = []

        # Generate samples.
        for i in tqdm(range(num_samps)):
            # Create initial positions and momenta.
            x = self.x.detach().clone().requires_grad_()
            p = torch.randn(x.shape, device=self.device) * torch.sqrt(self.m)

            # Calculate the initial Hamiltonian and gradients.
            lnP = self.lnP(x)
            prevLnP = lnP.detach().clone()
            H_init = (0.5 * torch.sum(torch.square(p) / self.m) - lnP).detach().cpu().numpy()
            grad = torch.autograd.grad(lnP, x)[0]

            # Initial leapfrog step.
            p += 0.5 * grad * step_size
            x = x + (p / self.m) * step_size

            # Update the gradient.
            lnP = self.lnP(x)
            grad = torch.autograd.grad(lnP, x)[0]

            # Continue leapfrog steps.
            for i in range(1, num_steps):
                p += grad * step_size
                x = x + (p / self.m) * step_size

                lnP = self.lnP(x)
                grad = torch.autograd.grad(lnP, x)[0]

            # Final half-step in momentum.
            p += 0.5 * grad * step_size

            # Calculate final Hamiltonian.
            H_prime = (0.5 * torch.sum(torch.square(p) / self.m) - lnP).detach().cpu().numpy()

            # Perform Metropolis-Hastings accept/reject.
            accept_ratio = np.exp(np.minimum(H_init - H_prime, 0))
            accept_prob = min(accept_ratio, 1)
            if np.random.uniform() < accept_prob:
                sample = {'x': self.transform(x).detach().cpu().numpy(), 'lnP': lnP.detach().cpu().numpy(),
                          'accpet_ratio': accept_ratio, 'accept_prob': accept_prob, 'accepted': True}
                self.x = x
            else:
                sample = {'x': self.transform(self.x).detach().cpu().numpy(), 'lnP': prevLnP.cpu().numpy(),
                          'accpet_ratio': accept_ratio, 'accept_prob': accept_prob, 'accepted': False}
            chain.append(sample)

        return chain
