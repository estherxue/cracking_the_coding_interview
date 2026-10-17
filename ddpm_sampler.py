import torch


class DDPMSampler:
    def __init__(self, model, beta_start, beta_end, num_steps, dim):
        self.model = model
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_steps = num_steps
        self.dim = dim
        self.init_alpha_bar_t()
    
    def get_beta_t(self, t):
        beta_t = self.beta_start + (self.beta_end - self.beta_start) * t / self.num_steps
        return torch.as_tensor(beta_t, dtype=torch.float)
    
    def get_estimated_noise(self, x_t, t):
        return self.model.predict_noise(x_t, t)
    
    def get_sigma_t(self, t):
        return torch.sqrt(self.get_beta_t(t))
    
    def init_alpha_bar_t(self):
        self.alpha_bar_t = torch.cumprod(1 - self.get_beta_t(torch.arange(self.num_steps + 1)), dim=0)
    
    def sample_x_t_minus_1(self, t, x_t):
        z_t = torch.randn_like(x_t) if t != 0 else 0
        sigma_t = self.get_sigma_t(t)
        alpha_t = 1 - self.get_beta_t(t)
        estimated_noise = self.get_estimated_noise(x_t, t)
        coeff_noise = (1 - alpha_t) / torch.sqrt(1 - self.alpha_bar_t[t])
        pre_result = 1 / torch.sqrt(alpha_t) * (x_t - coeff_noise * estimated_noise)
        x_t_minus_1 = pre_result + sigma_t * z_t
        return x_t_minus_1

    @torch.no_grad()
    def sample_loop(self):
        x_t = torch.randn(self.dim)
        for t in range(self.num_steps - 1, -1, -1):
            x_t = self.sample_x_t_minus_1(t, x_t)
        return x_t
    
