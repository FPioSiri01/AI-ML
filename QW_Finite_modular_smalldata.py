import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
import os


# THEORETICAL TRANSCENDENTAL EQ AND DATASET

def f_even(E, V0, a):
        k = np.sqrt(2 * E)
        alpha = np.sqrt(2 * (V0 - E))
        return alpha * np.cos(k * a) - k * np.sin(k * a)

def f_odd(E, V0, a):
        k = np.sqrt(2 * E)
        alpha = np.sqrt(2 * (V0 - E))
        return alpha * np.sin(k * a) + k * np.cos(k * a)

def trasc_eq_E(V0, a, n_state):
    E_grid = np.linspace(0.001, V0 - 0.001, 2000)
    func = f_even if n_state % 2 == 0 else f_odd
    f = func(E_grid, V0, a)
    sign_changes = np.where(np.diff(np.sign(f)))[0]
    root_idx = n_state // 2
    if root_idx < len(sign_changes):
        E_low = E_grid[sign_changes[root_idx]]
        E_high = E_grid[sign_changes[root_idx] + 1]
        E_th = brentq(func, E_low, E_high, args=(V0, a))
    else: E_th = np.nan

    return E_th

def generate_and_save_finite_noisy_dataset(num_states, L, a, V0, filename, points_per_state, noise_level):
    """
    Generates a noisy dataset for the Finite Potential Well.
    """
    data_dict = {}
    rng = np.random.default_rng(42)

    print("--- Pre-calculating Analytical Solutions for Dataset ---")
    for n in range(num_states):
        E_th = trasc_eq_E(V0, a, n)

        if not np.isnan(E_th):
            
            x_sampled = rng.uniform(-L, L, points_per_state)

            k = np.sqrt(2 * E_th)
            alpha = np.sqrt(2 * (V0 - E_th))
            
            if n % 2 == 0:
                psi_sampled = np.where(np.abs(x_sampled) <= a, 
                                       np.cos(k * x_sampled), 
                                       np.cos(k * a) * np.exp(-alpha * (np.abs(x_sampled) - a)))
            else:
                psi_sampled = np.where(np.abs(x_sampled) <= a, 
                                       np.sin(k * x_sampled), 
                                       np.sign(x_sampled) * np.sin(k * a) * np.exp(-alpha * (np.abs(x_sampled) - a)))
            
            x_norm = np.linspace(-L, L, 2000)
            if n % 2 == 0:
                psi_norm = np.where(np.abs(x_norm) <= a, np.cos(k * x_norm), np.cos(k * a) * np.exp(-alpha * (np.abs(x_norm) - a)))
            else:
                psi_norm = np.where(np.abs(x_norm) <= a, np.sin(k * x_norm), np.sign(x_norm) * np.sin(k * a) * np.exp(-alpha * (np.abs(x_norm) - a)))
            norm_fact = np.sqrt(np.trapezoid(psi_norm**2, x_norm))
            
            psi_sampled /= norm_fact
            
            # Add Gaussian noise to the dataset
            if noise_level > 0.0:
                psi_sampled += rng.normal(0.0, noise_level, size=psi_sampled.shape)
            
            data_dict[f"x_n{n}"] = x_sampled.astype(np.float32)
            data_dict[f"psi_n{n}"] = psi_sampled.astype(np.float32)
            print(f"  State n={n} bound found at E = {E_th:.4f} a.u.")
        else:
            print(f"  State n={n} is unbound for V0={V0}. Filling with fallback dummy values.")
            data_dict[f"x_n{n}"] = rng.uniform(-L, L, points_per_state).astype(np.float32)
            
            # Add noise to fallback dummy values as well
            dummy_psi = np.zeros(points_per_state)
            if noise_level > 0.0:
                dummy_psi += rng.normal(0.0, noise_level, size=dummy_psi.shape)
                
            data_dict[f"psi_n{n}"] = dummy_psi.astype(np.float32)

    np.savez(filename, **data_dict)
    print(f"--> Saved reference data to '{filename}'\n")

# PINN CLASS
class APINN(nn.Module):
    def __init__(self, L, a, V0):
        super(APINN, self).__init__()
        
        self.L = L    
        self.a = a    
        self.V0 = V0  
        
        self.hidden_layer1 = nn.Linear(1, 80)
        self.hidden_layer2 = nn.Linear(80, 80)
        self.hidden_layer3 = nn.Linear(80, 80)
        self.hidden_layer4 = nn.Linear(80, 80)
        self.hidden_layer5 = nn.Linear(80,80)
        self.hidden_layer6 = nn.Linear(80,80)
        self.output_layer = nn.Linear(80, 2)
        
        self.E_net = nn.Linear(1, 1, bias=False)
        #nn.init.constant_(self.E_net.bias, 1.0)
        #self.optimizer = torch.optim.Adam(list(self.parameters()) + list(self.E_net.parameters()), lr=0.01)

        psi_params = (list(self.hidden_layer1.parameters()) + 
                  list(self.hidden_layer2.parameters()) + 
                  list(self.hidden_layer3.parameters()) +
                  list(self.hidden_layer4.parameters()) +
                  list(self.hidden_layer5.parameters()) +
                  list(self.hidden_layer6.parameters()) +
                  list(self.output_layer.parameters()))
    
        E_params = list(self.E_net.parameters())
        
        # Single optimizer with different learning rates
        self.optimizer = torch.optim.Adam([
            {'params': psi_params, 'lr': 0.006},   
            {'params': E_params, 'lr': 0.006}])
        # LR scheduler
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)

        self.mse = nn.MSELoss()
        self.sae = nn.L1Loss(reduction='sum')

    def forward(self, x):
        x_out1 = torch.tanh(self.hidden_layer1(x))
        x_out2 = torch.tanh(self.hidden_layer2(x_out1))
        x_out3 = torch.tanh(self.hidden_layer3(x_out2))
        x_out4 = torch.tanh(self.hidden_layer4(x_out3))
        x_out5 = torch.tanh(self.hidden_layer5(x_out4))
        x_out6 = torch.tanh(self.hidden_layer6(x_out5))
       
        out = self.output_layer(x_out6)
        psi = out[:, 0:1]
        v = out[:, 1:2]
        return psi, v
    
    def get_potential(self, x):
        return torch.where(torch.abs(x) <= self.a, torch.tensor(0.0, device=x.device), torch.tensor(self.V0, device=x.device))
    
    def get_energy(self):
        dummy_input = torch.ones(1, 1)
        return self.E_net(dummy_input).item()

    def diff_eq(self, x_colloc, psi, E, V_x):
        psi_x = torch.autograd.grad(psi, x_colloc, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x, x_colloc, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
        
        pde_residual = -0.5 * psi_xx + (V_x - E) * psi
        loss_pde = self.mse(pde_residual, torch.zeros_like(pde_residual))
        return loss_pde

    def int_norm_bc(self, x_colloc, x_bound_0, x_bound_1, psi, v):
        v_x = torch.autograd.grad(v, x_colloc, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        loss_int = self.mse(v_x - psi**2, torch.zeros_like(psi))
        
        _, v_0 = self.forward(x_bound_0)
        _, v_1 = self.forward(x_bound_1)
        loss_norm = self.sae(v_0, torch.zeros_like(v_0)) + self.sae(v_1, torch.ones_like(v_1))
        
        psi_b0, _ = self.forward(x_bound_0)
        psi_b1, _ = self.forward(x_bound_1)
        loss_bc = self.sae(psi_b0, torch.zeros_like(psi_b0)) + self.sae(psi_b1, torch.zeros_like(psi_b1))
        
        return loss_int, loss_norm, loss_bc
    
    def ortho(self, previous_models, x_colloc, psi):
        loss_ortho = torch.tensor(0.0, device=x_colloc.device)
        for prev_model in previous_models:
            with torch.no_grad():
                psi_prev, _ = prev_model(x_colloc)
            overlap = torch.mean(psi * psi_prev)
            loss_ortho = loss_ortho + self.mse(overlap, torch.zeros_like(overlap))
        return loss_ortho
    
    def symm(self, x_colloc, psi, n_state):
        s = 1.0 if (n_state % 2 == 0) else -1.0
        psi_neg, _ = self.forward(-x_colloc)
        loss_symmetry = self.mse(psi, s * psi_neg)
        return loss_symmetry
    
    def emin(self, E, E_init_val):
        a_param = 0.8
        loss_emin = torch.mean(torch.exp(a_param * (E - E_init_val)))
        return loss_emin

    def data_loss(self, dataset, n_state, x_colloc):
        loss_data = torch.tensor(0.0, device=x_colloc.device)
        if dataset is not None:
            x_d = torch.tensor(dataset[f"x_n{n_state}"], dtype=torch.float32).reshape(-1, 1)
            psi_d = torch.tensor(dataset[f"psi_n{n_state}"], dtype=torch.float32).reshape(-1, 1)
            
            psi_d_pred, _ = self.forward(x_d)
            if torch.sum(psi_d_pred * psi_d) < 0:
                psi_d_pred = -psi_d_pred
                
            loss_data = self.mse(psi_d_pred, psi_d)
        
        return loss_data

    def compute_losses(self, x_colloc, x_bound_0, x_bound_1, epoch, previous_models, E_init_val, n_state, initial_weights_losses, dataset):
        self.optimizer.zero_grad()
        
        psi, v = self.forward(x_colloc)
        E = self.E_net(torch.ones_like(x_colloc))
        V_x = self.get_potential(x_colloc)
        
        loss_pde = self.diff_eq(x_colloc, psi, E, V_x)
        w_pde = initial_weights_losses[0] *20*np.log10(10+1e6*epoch)
        
        loss_int, loss_norm, loss_bc = self.int_norm_bc(x_colloc, x_bound_0, x_bound_1, psi, v)
        w_int = initial_weights_losses[1]  * 70.0
        w_norm = initial_weights_losses[2]  * 50.0
        w_bc = initial_weights_losses[3]  *15.0
        
        loss_ortho = self.ortho(previous_models, x_colloc, psi)
        w_ortho = initial_weights_losses[4] *150.0

        loss_symmetry = self.symm(x_colloc, psi, n_state)
        w_symm = initial_weights_losses[5] #* 5.0

        loss_emin = self.emin(E, E_init_val)
        w_emin = initial_weights_losses[6] * 10*np.exp(-epoch / 1000.0) 

        loss_data = self.data_loss(dataset, n_state, x_colloc)
        w_data = initial_weights_losses[7] #*150 #* np.log10(10+1e8*epoch)
        
        total_loss = (w_pde * loss_pde) + \
                     (w_int * loss_int) + \
                     (w_norm * loss_norm) + \
                     (w_bc * loss_bc) + \
                     (w_ortho * loss_ortho) + \
                     (w_symm * loss_symmetry) + \
                     (w_emin * loss_emin) + \
                     (w_data * loss_data) 
                     
        total_loss.backward()
        self.optimizer.step()
        
        return (total_loss.item(), loss_pde.item(), loss_int.item(), loss_norm.item(), 
                loss_bc.item(), loss_ortho.item(), loss_symmetry.item(), loss_emin.item(), loss_data.item(), E.mean().item())

# SAVE AND LOAD MODELS
def save_model(model, state_n, points_per_state, batch_size, L, a, V0, weights_losses, save_dir="finite_saved_states"):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"FIN_nonfinal_L({L})_a({a})_V0({V0})_PPS({points_per_state})_BS({batch_size})_pde({weights_losses[0]})_int({weights_losses[1]})_norm({weights_losses[2]})_bc({weights_losses[3]})_ortho({weights_losses[4]})_symm({weights_losses[5]})_emin({weights_losses[6]})_data({weights_losses[7]})_{state_n}.pt"    
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Saved state n={state_n} to {filepath}")


def load_previous_models(num_previous, L, a, V0, points_per_state, batch_size, weights_losses, save_dir="finite_saved_states"):
    models = []
    print(f"\nLoading {num_previous} previously trained states")
    for i in range(num_previous):
        filename = f"FIN_nonfinal_L({L})_a({a})_V0({V0})_PPS({points_per_state})_BS({batch_size})_pde({weights_losses[0]})_int({weights_losses[1]})_norm({weights_losses[2]})_bc({weights_losses[3]})_ortho({weights_losses[4]})_symm({weights_losses[5]})_emin({weights_losses[6]})_data({weights_losses[7]})_{i}.pt"        
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find saved model for state {i} at {filepath}")
        
        model = APINN(L, a, V0)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
        print(f"Loaded state n={i} (E ≈ {model.get_energy():.4f})")
    return models


# TRAINING 
def train_spectrum(start_state_idx, target_states, n_epochs, batch_size, L, a, V0, points_per_state, initial_weights_losses, loaded_models=None):
    discovered_models = loaded_models.copy() if loaded_models else []
    all_loss_histories = []
    
    data_file = np.load("finite_quantum_noisy_data.npz") if POINTS_PER_STATE > 0 else None
    
    L_domain = L
    a_well = a
    
    x_b0 = torch.tensor([[-L_domain]], requires_grad=True)
    x_b1 = torch.tensor([[L_domain]], requires_grad=True)
    
    if discovered_models: current_E_init = discovered_models[-1].get_energy() + (start_state_idx+1)**2 - (start_state_idx)**2 - 2.0
    else: current_E_init = 0.0 
        
    n_epochs0 = n_epochs

    for n in range(start_state_idx, target_states):
        print(f" \n TRAINING STATE n={n} (E > {current_E_init:.2f})")
        
        model = APINN(L_domain, a_well, V0)
        n_state = n
        
        names_loss = ["total", "pde", "int", "norm", "bc", "ortho", "symmetry", "emin", "data", "E", "err_E", "fidelity"]
        state_loss = {k: [] for k in names_loss}

        if (n_state == 3): n_epochs = 2*n_epochs0
        elif (n_state >= 4 and n_state <= 6): n_epochs = 4*n_epochs0
        elif n_state > 6: n_epochs = 8*n_epochs0
        else: n_epochs = n_epochs0

        E_th = trasc_eq_E(V0, a_well, n_state)

        for epoch in range(n_epochs):
            x_colloc = (2.0 * L_domain) * torch.rand(size=(batch_size, 1), requires_grad=True, dtype=torch.float32) - L_domain
            '''# More collocation points inside the well
            colloc_frac = 0.6  
            n_well = int(batch_size * colloc_frac)
            n_tail = batch_size - n_well
            x_well = (2.0 * a_well) * torch.rand(n_well, 1, dtype=torch.float32) - a_well
            n_left = n_tail // 2
            n_right = n_tail - n_left
            x_left  = (L_domain - a_well) * torch.rand(n_left,  1, dtype=torch.float32) - L_domain
            x_right = (L_domain - a_well) * torch.rand(n_right, 1, dtype=torch.float32) + a_well
            x_colloc = torch.cat([x_well, x_left, x_right], dim=0)
            x_colloc = x_colloc[torch.randperm(x_colloc.size(0))]
            x_colloc = x_colloc.requires_grad_(True)'''

            losses = model.compute_losses(x_colloc, x_b0, x_b1, epoch, discovered_models, current_E_init, n_state, initial_weights_losses, dataset=data_file)
            model.scheduler.step()

            if (epoch + 1) % 10 == 0:
                with torch.no_grad():
                    x_test = torch.linspace(-L_domain, L_domain, 500).reshape(-1, 1)
                    x_np = x_test.numpy().flatten()
                    
                    psi_pred_test, _ = model(x_test)
                    psi_pred_test = psi_pred_test.numpy().flatten()
                    E_pred_test = model.get_energy()
                    
                    k = np.sqrt(2 * E_th)
                    alpha = np.sqrt(max(0, 2 * (V0 - E_th)))
                    if n_state % 2 == 0:
                        psi_theory = np.where(np.abs(x_np) <= a_well, np.cos(k * x_np), np.cos(k * a_well) * np.exp(-alpha * (np.abs(x_np) - a_well)))
                    else:
                        psi_theory = np.where(np.abs(x_np) <= a_well, np.sin(k * x_np), np.sign(x_np) * np.sin(k * a_well) * np.exp(-alpha * (np.abs(x_np) - a_well)))
                    
                    # Normalization for fidelity
                    norm_const = np.trapezoid(psi_pred_test**2, x_np)
                    if norm_const > 0:
                        psi_pred_test /= np.sqrt(norm_const)
                    psi_theory /= np.sqrt(np.trapezoid(psi_theory**2, x_np))
                    if np.sum(psi_pred_test * psi_theory) < 0:
                        psi_pred_test = -psi_pred_test
                        
                    err_E = (E_th - E_pred_test) / E_th
                    v_pred_norm = psi_pred_test / (np.linalg.norm(psi_pred_test) + 1e-10)
                    v_theory_norm = psi_theory / (np.linalg.norm(psi_theory) + 1e-10)
                    fidelity = np.abs(np.vdot(v_theory_norm, v_pred_norm))**2
                    
                losses_extended = list(losses) + [err_E, fidelity]
                for i, k in enumerate(names_loss):
                    state_loss[k].append(losses_extended[i])
                
                if (losses[0]<0.5 and losses[1]<0.05 and fidelity > 0.98): 
                    print(f"Converged state n={n} at epoch {epoch+1}: total={losses[0]:.4f}, pde={losses[1]:.5f}")
                    break

            if (epoch + 1) % 500 == 0:  
                print(f"Epoch {epoch+1:5d} | E: {losses[9]:.4f} a.u. | "
                    f"Tot: {losses[0]:.3f} | PDE: {losses[1]:.3f} | "
                    f"Int: {losses[2]:.3e} | Norm: {losses[3]:.3e} | "
                    f"BC: {losses[4]:.3e} | Symm: {losses[6]:.3e} | Ortho: {losses[5]} | "
                    f"Emin: {losses[7]:.3e} | Data: {losses[8]:.3e} | Err E: {err_E:.2e} | Fid: {fidelity:.4f}")
                
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        discovered_models.append(model)
        all_loss_histories.append(state_loss)
        
        save_model(model, n_state, points_per_state, batch_size, L_domain, a_well, V0, initial_weights_losses)
        
        current_E_init = losses[9] + (n_state + 1)**2 - n_state**2 - 1.0

    return discovered_models, all_loss_histories

# TEST AND PLOT
def evaluate_and_plot(discovered_models, histories=None, dataset=None):
    num_states = len(discovered_models)
    
    L_domain = discovered_models[0].L
    a_well = discovered_models[0].a
    V0 = discovered_models[0].V0
    x_test = torch.linspace(-L_domain, L_domain, 400).reshape(-1, 1)
    x_np = x_test.numpy().flatten()
    
    fig, axes = plt.subplots(num_states, 1, figsize=(8, 4 * num_states))
    if num_states == 1:
        axes = [axes]
    
    E_theory = []

    for n in range(num_states):
        E_theory.append(trasc_eq_E(V0, a_well, n))
            
    with torch.no_grad():
        for i, model in enumerate(discovered_models):
            psi_pred, _ = model(x_test)
            psi_pred = psi_pred.numpy().flatten()
            E_pred = model.get_energy()
            E_th = E_theory[i]
            V_test = model.get_potential(x_test).numpy()
            
            k = np.sqrt(2 * E_th)
            alpha = np.sqrt(max(0, 2 * (V0 - E_th)))
            
            if i % 2 == 0:
                psi_theory = np.where(np.abs(x_np) <= a_well, 
                                        np.cos(k * x_np), 
                                        np.cos(k * a_well) * np.exp(-alpha * (np.abs(x_np) - a_well)))
            else:
                psi_theory = np.where(np.abs(x_np) <= a_well, np.sin(k * x_np), np.sign(x_np) * np.sin(k * a_well) * np.exp(-alpha * (np.abs(x_np) - a_well)))
            
            psi_theory /= np.sqrt(np.trapezoid(psi_theory**2, x_np))
            
            err_E = (E_th - E_pred) / E_th
            v_pred_norm = psi_pred / (np.linalg.norm(psi_pred) + 1e-10)
            v_theory_norm = psi_theory / (np.linalg.norm(psi_theory) + 1e-10)
            fidelity = np.abs(np.vdot(v_theory_norm, v_pred_norm))**2
            psi_pred /= np.sqrt(np.trapezoid(psi_pred**2, x_np))
            if np.sum(psi_pred * psi_theory) < 0: 
                psi_pred = -psi_pred
                
            ax1 = axes[i]
            color = 'tab:blue'
            ax1.set_xlabel("Space Coordinates $x$ (a.u.)", weight='bold')
            ax1.set_ylabel("Amplitude $\psi(x)$", color=color, weight='bold')
            
            # Plot dataset
            if dataset is not None and f"x_n{i}" in dataset:
                x_d = dataset[f"x_n{i}"]
                psi_d = dataset[f"psi_n{i}"]
                ax1.scatter(x_d, psi_d, color='black', marker='x', s=50, label='Training Data', zorder=5)

            ax1.plot(x_np, psi_pred, color=color, linewidth=2, label=f"APINN $\psi(x)$ (E={E_pred:.2f})", zorder=4)
            if not np.isnan(E_th):
                ax1.plot(x_np, psi_theory, color='black', linestyle='--', alpha=0.7, label=f"Theory $\psi(x)$ (E={E_th:.2f})", zorder=3)
            
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel("Potential $V(x)$", color=color, weight='bold')
            ax2.plot(x_np, V_test, color=color, linestyle='-', alpha=0.3, label=f"$V(x)$")
            ax2.set_ylim(-1, V0 + 5)
            
            ax2.axhline(y=E_pred, color='tab:green', linestyle='-.', label=f"E_pred = {E_pred:.3f}")
            if not np.isnan(E_th):
                ax2.axhline(y=E_th, color='black', linestyle=':', alpha=0.7, label=f"E_th = {E_th:.3f}")

            state_name = "Ground State" if i == 0 else f"Excited State {i}"
            if not np.isnan(E_th):
                ax1.set_title(f"State n={i}: {state_name}\nPred Energy = {E_pred:.4f} (Exact = {E_th:.4f})\nFinal Err E = {err_E:.2e} | $\mathcal{{F}}_{{\psi}}$ = {fidelity:.6f}", fontsize=12, weight='bold')
            else:
                ax1.set_title(f"State n={i}: {state_name}\nPred Energy = {E_pred:.4f} (Exact = Unbound)", fontsize=12, weight='bold')
            
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    
    # PLOT LOSSES AND METRICS
    if histories:
        for i, history in enumerate(histories):
            epochs = np.arange(1, len(history["total"]) + 1) * 10

            # Plot all losses
            plt.figure(figsize=(8, 4))
            for k, v in history.items():
                if k not in ["err_E", "fidelity", "E", "total"] and len(v) > 0:
                    plt.plot(epochs, v, label=k)
            plt.yscale('log')
            plt.title(f"State n={i} Convergence History Profile")
            plt.xlabel("Epoch")
            plt.ylabel("Loss Magnitude")
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.show()

            # Plot Total Loss only
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, history["total"], color='tab:red', linewidth=2, label='Total Loss')
            plt.yscale('log')
            plt.title(f"State n={i} Total Loss Convergence History")
            plt.xlabel("Epoch")
            plt.ylabel("Total Loss Magnitude")
            plt.legend(loc="upper right")
            plt.grid(True, which="both", linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()
            
            # Plot metrics
            fig_metrics, (ax_err, ax_fid) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax_err.plot(epochs, history["err_E"], color='orange', label='$err_E$', linewidth=2)
            ax_err.set_title(f"State n={i} Relative Error of Energy ($err_E$)")
            ax_err.set_xlabel("Epoch")
            ax_err.set_ylabel("$err_E$")
            ax_err.grid(True, alpha=0.5)
            
            ax_fid.plot(epochs, history["fidelity"], color='purple', label='$\mathcal{F}_{\psi}$', linewidth=2)
            ax_fid.set_title(f"State n={i} Fidelity")
            ax_fid.set_xlabel("Epoch")
            ax_fid.set_ylabel("Fidelity")
            ax_fid.grid(True, alpha=0.5)

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    NUM_STATES = 5
    N_EPOCHS = 5000 
    BATCH_SIZE = 600
    L_DOMAIN = 2.5
    A_WELL = 1.0
    V0 = 15.0 
    POINTS_PER_STATE = 0
    INITIAL_WEIGHTS_LOSSES = [2.0, 5000.0, 1000.0, 10.0, 1000.0, 1000.0, 10.0, 50.0]
    
    generate_and_save_finite_noisy_dataset(NUM_STATES, L_DOMAIN, A_WELL, V0, filename="finite_quantum_noisy_data.npz", points_per_state=POINTS_PER_STATE, noise_level=0.1)
    data_file_plot = np.load("finite_quantum_noisy_data.npz") if POINTS_PER_STATE != 0 else None
   
    loaded_models = load_previous_models(num_previous=0, L=L_DOMAIN, a=A_WELL, V0=V0, points_per_state=POINTS_PER_STATE, batch_size=BATCH_SIZE, weights_losses=INITIAL_WEIGHTS_LOSSES) 
    all_models, all_histories = train_spectrum(start_state_idx=0, target_states=4, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, L=L_DOMAIN, a=A_WELL, V0=V0, points_per_state=POINTS_PER_STATE, initial_weights_losses=INITIAL_WEIGHTS_LOSSES, loaded_models=loaded_models)
    
    evaluate_and_plot(all_models, all_histories, dataset=data_file_plot)

