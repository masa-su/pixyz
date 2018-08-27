import torch
from torch import optim

from Tars.models.model import Model
from Tars.utils import tolist

class VRNN(Model):
    def __init__(self, encoder, decoder, prior,
                 phi_x, phi_z, rnn_cell, 
                 other_distributions=[], regularizer=[],
                 optimizer=optim.Adam, optimizer_params={}):
        super(VRNN, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.phi_x = phi_x
        self.phi_z = phi_z
        self.rnncell = rnn_cell
        self.regularizer = tolist(regularizer)
        
        # set params and optim
        q_params = list(self.encoder.parameters())
        p_params = list(self.decoder.parameters())
        prior_params = list(self.prior.parameters())
        phi_x_params = list(self.phi_x.parameters())
        phi_z_params = list(self.phi_z.parameters())
        rnn_params = list(self.rnncell.parameters())
        params = q_params + p_params + prior_params
        params += phi_x_params + phi_z_params + rnn_params
        
        other_distributions = tolist(other_distributions)
        for distribution in other_distributions:
            params += list(distributions.parameters())
            
        self.optimizer = optimizer(params, **optimizer_params)
                
    def train(self, train_x, coef=1):
        "train_x['x']: [sequence_length, batch_size, dim]"
        self.decoder.train()
        self.encoder.train()
        self.prior.train()
        self.phi_x.train()
        self.phi_z.train()
        self.rnncell.train()
        
        self.optimizer.zero_grad()

        lower_bound, loss = 0, 0
        h = self.init_hidden(train_x["x"].size(1))
        for step in range(train_x["x"].size(0)):
            x = train_x["x"][step]
            h_next, phi_x_t, phi_z_t = self.vrnn_step(x, h)

            _lb, _loss = self._elbo_step(x, h, phi_x_t, phi_z_t)
            lower_bound, loss = self.add_loss(lower_bound, loss, _lb, _loss)
            h = h_next
            
        loss.backward()
        self.optimizer.step()
        
        return lower_bound, loss
    
    def test(self, test_x, coef=1):
        self.decoder.eval()
        self.encoder.eval()
        self.prior.eval()
        self.phi_x.eval()
        self.phi_z.eval()
        self.rnncell.eval()
        
        lower_bound, loss = 0, 0
        with torch.no_grad():
            h = self.init_hidden(test_x["x"].size(1))
            for step in range(test_x["x"].size(0)):
                x = test_x["x"][step]
                h_next, phi_x_t, phi_z_t = self.vrnn_step(x, h)

                _lb, _loss = self._elbo_step(x, h, phi_x_t, phi_z_t)
                lower_bound, loss = self.add_loss(lower_bound, loss, _lb, _loss)
                h = h_next

        return lower_bound, loss
    
    def vrnn_step(self, x ,h):
        phi_z_t = self.phi_z(self.prior.sample({"h": h})["z"])
#         phi_z_t = self.phi_z(self.prior.sample({"h": h}))
#         print(phi_z_t)
        phi_x_t = self.phi_x(x)
        h = self.rnncell(torch.cat((phi_x_t, phi_z_t), dim=-1), h)
        return h, phi_x_t, phi_z_t
    
    def _elbo_step(self, x, h, phi_x, phi_z, reg_coef=[1]):
        reg_coef = tolist(reg_coef)
        
        log_like = self.decoder.log_likelihood({"h": h, "x": x, "z": phi_z})
        
        lower_bound = [log_like]
        reg_loss = 0
        for i, reg in enumerate(self.regularizer):
            _reg = reg.estimate({"h": h, "x": phi_x})
            lower_bound.append(_reg)
            reg_loss += reg_coef[i] * _reg
        
        lower_bound = torch.stack(lower_bound, dim=-1)
        loss = -torch.mean(log_like - reg_loss)
        
        return lower_bound, loss
            
        
    def init_hidden(self, batch_size):
        h = torch.zeros(batch_size, self.rnncell.hidden_size)
        try: return h.to("cuda") # temporal
        except: return h
    
    def add_loss(self, lb, loss, _lb, _loss):
        return lb + _lb, loss + _loss