import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        # Compute reset gate: rt = σ(Wrx · xt + brx + Wrh · ht−1 + brh)
        self.r = self.r_act.forward(self.Wrx @ x + self.brx + self.Wrh @ h_prev_t + self.brh)
        
        # Compute update gate: zt = σ(Wzx · xt + bzx + Wzh · ht−1 + bzh)
        self.z = self.z_act.forward(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        
        # Compute candidate hidden state: nt = tanh(Wnx · xt + bnx + rt ⊙ (Wnh · ht−1 + bnh))
        self.n = self.h_act.forward(self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh))
        
        # Compute final hidden state: ht = (1 − zt) ⊙ nt + zt ⊙ ht−1
        h_t = (1 - self.z) * self.n + self.z * h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # Reshape x and hidden for gradient computation
        x_col = self.x.reshape(-1, 1) 
        h_col = self.hidden.reshape(-1, 1)  
        
        # Gradients from h_t = (1 - z) * n + z * h_prev_t
        dh_dz = -self.n + self.hidden 
        dh_dn = 1 - self.z 
        dh_dh_prev = self.z 
        
        dz = delta * dh_dz 
        dn = delta * dh_dn  
        
        # Gradients from n = tanh(Wnx @ x + bnx + r * (Wnh @ h_prev_t + bnh))
        # Derivative of tanh
        dn_pre_act = dn * (1 - self.n**2)
        
        # Gradients for n
        self.dWnx = np.outer(dn_pre_act, self.x)  
        self.dbnx = dn_pre_act  
        
        # For the term r * (Wnh @ h_prev_t + bnh)
        Wnh_h = self.Wnh @ self.hidden 
        dr = dn_pre_act * (Wnh_h + self.bnh)  
        
        dWnh_h_plus_bnh = dn_pre_act * self.r  
        self.dWnh = np.outer(dWnh_h_plus_bnh, self.hidden)  
        self.dbnh = dWnh_h_plus_bnh 
        
        # Gradients from z = σ(Wzx @ x + bzx + Wzh @ h_prev_t + bzh)
        # Derivative of sigmoid
        dz_pre_act = dz * self.z * (1 - self.z) 
        
        self.dWzx = np.outer(dz_pre_act, self.x)  
        self.dbzx = dz_pre_act  
        self.dWzh = np.outer(dz_pre_act, self.hidden)  
        self.dbzh = dz_pre_act  
        
        # Gradients from r = σ(Wrx @ x + brx + Wrh @ h_prev_t + brh)
        # Derivative of sigmoid
        dr_pre_act = dr * self.r * (1 - self.r) 
        
        self.dWrx = np.outer(dr_pre_act, self.x) 
        self.dbrx = dr_pre_act
        self.dWrh = np.outer(dr_pre_act, self.hidden) 
        self.dbrh = dr_pre_act 
        
        # 5. Gradients wrt inputs (x and h_prev_t)
        dx = self.Wnx.T @ dn_pre_act + self.Wzx.T @ dz_pre_act + self.Wrx.T @ dr_pre_act
        
        dh_prev_t = (delta * dh_dh_prev + 
                     self.Wnh.T @ dWnh_h_plus_bnh + 
                     self.Wzh.T @ dz_pre_act + 
                     self.Wrh.T @ dr_pre_act)

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t