import torch


class FlowSampler:
    def __init__(self, model, num_steps, dim):
        self.model = model
        self.num_steps = num_steps
        self.dim = dim
    
    def get_timesteps(self, steps):
        timesteps = torch.linspace(0, 1, steps + 1)
        return timesteps

    def get_velocity(self, x_t, t):
        return self.model.predict_velocity(x_t, t)
    
    def get_x_plus_delta_t(self, x_t, t, delta_t):
        return x_t + delta_t * self.get_velocity(x_t, t)
    
    @torch.no_grad()
    def sample_loop(self):
        timesteps = self.get_timesteps(self.num_steps)
        x_t = torch.randn(self.dim)
        for i, t in enumerate(timesteps[:-1]):
            delta_t = timesteps[i + 1] - timesteps[i]
            x_t = self.get_x_plus_delta_t(x_t, t, delta_t)
        return x_t

