import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1. PHYSICS-INFORMED NEURAL NETWORK MODEL FOR 2D INFINITE WELL
# ==============================================================
class APINN(nn.Module):
    def __init__(self, L):
        super(APINN, self).__init__()
        
        # Physical parameters of the infinite square well
        self.L = L    # Well width, domain is [-L/2, L/2] x [-L/2, L/2]

        # Network structure adapted for 2D inputs (x, y)
        self.hidden_layer1 = nn.Linear(2, 64)
        self.hidden_layer2 = nn.Linear(64, 64)
        self.hidden_layer3 = nn.Linear(64, 64)
        self.hidden_layer4 = nn.Linear(64, 64)
        self.hidden_layer5 = nn.Linear(64, 64)
        self.hidden_layer6 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 2) # Output 0: psi(x,y), Output 1: v(x,y)
        
        # Parallel Network for self-consistent Eigenvalue (E) computation
        self.E_net = nn.Linear(1, 1, bias=False)

        # Joint optimizer using Adam
        self.optimizer = torch.optim.Adam(list(self.parameters()) + list(self.E_net.parameters()), lr=0.001)

        # Loss metrics 
        self.mse = nn.MSELoss()
        self.sae = nn.L1Loss(reduction='sum') # Sum of Absolute Errors for physical constraints

    def forward(self, x, y):
        # Concatenate spatial dimensions for the network
        xy = torch.cat([x, y], dim=1)
        
        # Tanh activation function provides higher order derivatives via AutoGrad
        x_out = torch.tanh(self.hidden_layer1(xy))
        x_out = torch.tanh(self.hidden_layer2(x_out))
        x_out = torch.tanh(self.hidden_layer3(x_out))
        x_out = torch.tanh(self.hidden_layer4(x_out))
        x_out = torch.tanh(self.hidden_layer5(x_out))
        x_out = torch.tanh(self.hidden_layer6(x_out))
       
        out = self.output_layer(x_out)
        psi = out[:, 0:1]
        v = out[:, 1:2]
        return psi, v
    
    def get_energy(self):
        dummy_input = torch.ones(1, 1)
        return self.E_net(dummy_input).item()

    def compute_losses(self, x_colloc, y_colloc, epoch, previous_models, E_init_val, nx, ny):
        self.optimizer.zero_grad()
        
        # Forward passes
        psi, v = self.forward(x_colloc, y_colloc)
        E = self.E_net(torch.ones_like(x_colloc))
        
        # ---------------------------------------------------------
        # A. DIFFERENTIAL EQUATION LOSS (2D Schrödinger Equation)
        # ---------------------------------------------------------
        psi_x = torch.autograd.grad(psi, x_colloc, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x, x_colloc, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
        
        psi_y = torch.autograd.grad(psi, y_colloc, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        psi_yy = torch.autograd.grad(psi_y, y_colloc, grad_outputs=torch.ones_like(psi_y), create_graph=True)[0]
        
        # 2D Infinite Well Hamiltonian inside boundaries: V(x,y) = 0
        pde_residual = -0.5 * (psi_xx + psi_yy) - E * psi
        loss_pde = self.mse(pde_residual, torch.zeros_like(pde_residual))
        
        # ---------------------------------------------------------
        # B. 2D INTEGRAL & NORMALIZATION LOSSES
        # ---------------------------------------------------------
        # Integral loss mapped as mixed partial derivative: v_xy = |psi|^2
        v_x = torch.autograd.grad(v, x_colloc, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xy = torch.autograd.grad(v_x, y_colloc, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        loss_int = self.mse(v_xy, psi**2)
        
        # Normalization constraints
        L_bound = torch.tensor([[self.L / 2.0]], device=x_colloc.device)
        neg_L_bound_x = torch.full_like(x_colloc, -self.L / 2.0)
        neg_L_bound_y = torch.full_like(y_colloc, -self.L / 2.0)
        
        _, v_max = self.forward(L_bound, L_bound)
        _, v_zero_x = self.forward(neg_L_bound_x, y_colloc)
        _, v_zero_y = self.forward(x_colloc, neg_L_bound_y)
        
        loss_norm = self.sae(v_max, torch.ones_like(v_max)) + \
                    self.sae(v_zero_x, torch.zeros_like(v_zero_x)) + \
                    self.sae(v_zero_y, torch.zeros_like(v_zero_y))
        
        # ---------------------------------------------------------
        # C. BOUNDARY CONDITIONS LOSS (4 edges)
        # ---------------------------------------------------------
        pos_L_bound_x = torch.full_like(x_colloc, self.L / 2.0)
        pos_L_bound_y = torch.full_like(y_colloc, self.L / 2.0)
        
        psi_left, _ = self.forward(neg_L_bound_x, y_colloc)
        psi_right, _ = self.forward(pos_L_bound_x, y_colloc)
        psi_bottom, _ = self.forward(x_colloc, neg_L_bound_y)
        psi_top, _ = self.forward(x_colloc, pos_L_bound_y)
        
        loss_bc = self.sae(psi_left, torch.zeros_like(psi_left)) + \
                  self.sae(psi_right, torch.zeros_like(psi_right)) + \
                  self.sae(psi_bottom, torch.zeros_like(psi_bottom)) + \
                  self.sae(psi_top, torch.zeros_like(psi_top))
        
        # ---------------------------------------------------------
        # D. ORTHOGONALITY LOSS
        # ---------------------------------------------------------
        loss_ortho = torch.tensor(0.0, device=x_colloc.device)
        for prev_model in previous_models:
            with torch.no_grad():
                psi_prev, _ = prev_model(x_colloc, y_colloc)
            overlap = torch.mean(psi * psi_prev)
            loss_ortho = loss_ortho + self.mse(overlap, torch.zeros_like(overlap))
        
        # ---------------------------------------------------------
        # E. 2D SYMMETRY LOSS (Inductive Bias - Breaks Degeneracy)
        # ---------------------------------------------------------
        s_x = 1.0 if (nx % 2 == 1) else -1.0
        s_y = 1.0 if (ny % 2 == 1) else -1.0
        
        psi_negx, _ = self.forward(-x_colloc, y_colloc)
        psi_negy, _ = self.forward(x_colloc, -y_colloc)
        
        loss_symmetry = self.mse(psi - s_x * psi_negx, torch.zeros_like(psi)) + \
                        self.mse(psi - s_y * psi_negy, torch.zeros_like(psi))

        # ---------------------------------------------------------
        # F. ENERGY MINIMIZATION LOSS
        # ---------------------------------------------------------
        a_param = 0.8
        w_emin = 10.0 * np.exp(-epoch / 2000.0) 
        loss_emin = torch.mean(torch.exp(a_param * (E - E_init_val)))
        
        total_loss = (1.0 * loss_pde) + \
                     (5000.0 * loss_int) + \
                     (1000.0 * loss_norm) + \
                     (10.0 * loss_bc) + \
                     (1000.0 * loss_ortho) + \
                     (1000.0 * loss_symmetry) + \
                     (w_emin * loss_emin)
                     
        total_loss.backward()
        self.optimizer.step()
        
        return (total_loss.item(), loss_pde.item(), loss_int.item(), loss_norm.item(), loss_bc.item(), 
               loss_ortho.item(), loss_symmetry.item(), loss_emin.item(), E.mean().item())

# ==============================================================
# 2. SEQUENTIAL TRAINING ALGORITHM (2D)
# ==============================================================
def train_spectrum(states_to_find, n_epochs, batch_size, L):
    discovered_models = []
    all_loss_histories = [] 
    
    current_E_init = 0.0 

    # Building 2D foundational mesh for normal distribution sampling
    N_per_dim = int(np.sqrt(batch_size))
    base_mesh_1d = np.linspace(-L / 2.0, L / 2.0, N_per_dim)
    xx, yy = np.meshgrid(base_mesh_1d, base_mesh_1d)
    base_mesh_x = xx.flatten()
    base_mesh_y = yy.flatten()
    std_dev = (L / N_per_dim) * 0.5 

    for state_idx, (nx, ny) in enumerate(states_to_find):
        print(f"\n=======================================================")
        print(f" TRAINING STATE (nx={nx}, ny={ny}) (Targeting E > {current_E_init:.2f})")
        print(f"=======================================================")
        
        model = APINN(L)
        keys = ["total", "pde", "int", "norm", "bc", "ortho", "symmetry", "E"]
        state_loss = {k: [] for k in keys}

        for epoch in range(n_epochs):
            # Dynamic choice of training points in 2D
            sampled_x = np.random.normal(loc=base_mesh_x, scale=std_dev)
            sampled_y = np.random.normal(loc=base_mesh_y, scale=std_dev)
            sampled_x = np.clip(sampled_x, -L / 2.0, L / 2.0)
            sampled_y = np.clip(sampled_y, -L / 2.0, L / 2.0)
            
            x_colloc = torch.tensor(sampled_x, dtype=torch.float32).reshape(-1, 1)
            y_colloc = torch.tensor(sampled_y, dtype=torch.float32).reshape(-1, 1)
            x_colloc.requires_grad = True
            y_colloc.requires_grad = True
            
            losses = model.compute_losses(x_colloc, y_colloc, epoch, discovered_models, current_E_init, nx, ny)
            
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1:5d} | E: {losses[8]:.4f} eV | "
                    f"Tot: {losses[0]:.3f} | PDE: {losses[1]:.3f} | "
                    f"Int: {losses[2]:.2e} | Symm: {losses[6]:.2e}")
                for i, k in enumerate(keys):
                    state_loss[k].append(losses[i])

        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        discovered_models.append(model)
        all_loss_histories.append(state_loss)
        current_E_init = losses[8] + 0.1

    return discovered_models, all_loss_histories

# ==============================================================
# 3. 2D EVALUATION & PLOTTING
# ==============================================================
def evaluate_and_plot(discovered_models, states_to_find):
    L = discovered_models[0].L
    
    # High resolution grid for plotting
    grid_size = 60
    x_1d = np.linspace(-L / 2.0, L / 2.0, grid_size)
    y_1d = np.linspace(-L / 2.0, L / 2.0, grid_size)
    X, Y = np.meshgrid(x_1d, y_1d)
    
    x_test = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1)
    y_test = torch.tensor(Y.flatten(), dtype=torch.float32).reshape(-1, 1)
    
    print("\n--- Generating Metric Comparisons & 2D Graphical Plots ---")
    
    with torch.no_grad():
        for i, model in enumerate(discovered_models):
            nx, ny = states_to_find[i]
            
            psi_pred, _ = model(x_test, y_test)
            psi_pred = psi_pred.numpy().reshape(grid_size, grid_size)
            E_pred = model.get_energy()
            
            # Factual Analytic Solutions for 2D Square Well
            E_th = (nx**2 + ny**2) * np.pi**2 / (2.0 * L**2)
            psi_th = (2.0 / L) * np.sin(nx * np.pi / L * (X + L / 2.0)) * np.sin(ny * np.pi / L * (Y + L / 2.0))
            
            # Normalize APINN prediction via 2D trapezoidal rule execution
            norm_factor = np.sqrt(np.trapz(np.trapz(psi_pred**2, y_1d, axis=0), x_1d))
            psi_pred /= norm_factor
            
            # Phase Alignment
            if np.sum(psi_pred * psi_th) < 0: 
                psi_pred = -psi_pred
            
            # Metric performance displays
            err_E = (E_th - E_pred) / E_th
            fidelity = np.abs(np.trapz(np.trapz(psi_pred * psi_th, y_1d, axis=0), x_1d))**2
            print(f"State (nx={nx}, ny={ny}) | Exact E: {E_th:.5f} | Pred E: {E_pred:.5f} | Rel Err E: {err_E:.2e} | Fidelity: {fidelity:.6f}")
                
            # --- Rendering Output Plots ---
            fig = plt.figure(figsize=(14, 5))
            fig.suptitle(f"State (nx={nx}, ny={ny}) - Predicted Energy: {E_pred:.4f} (Exact: {E_th:.4f})", weight='bold', fontsize=14)
            
            # Subplot 1: Heatmap
            ax1 = fig.add_subplot(121)
            c = ax1.pcolormesh(X, Y, psi_pred, cmap='RdBu_r', shading='auto')
            fig.colorbar(c, ax=ax1)
            ax1.set_title("Wavefunction Heatmap $\psi(x,y)$")
            ax1.set_xlabel("x (natural units)")
            ax1.set_ylabel("y (natural units)")
            
            # Subplot 2: 3D Scatter (Sampling points to retain visibility)
            ax2 = fig.add_subplot(122, projection='3d')
            
            # Sampling logic to represent height with 3D scatter
            sample_indices = np.random.choice(len(X.flatten()), size=1000, replace=False)
            x_samp = X.flatten()[sample_indices]
            y_samp = Y.flatten()[sample_indices]
            z_samp = psi_pred.flatten()[sample_indices]
            
            scatter = ax2.scatter(x_samp, y_samp, z_samp, c=z_samp, cmap='RdBu_r', s=10, alpha=0.8)
            ax2.set_title("3D Wavefunction Scatter Projection")
            ax2.set_xlabel("x")
            ax2.set_ylabel("y")
            ax2.set_zlabel("$\psi(x,y)$")
            
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Specify the target quantum states [nx, ny] to compute
    # Note: Using symmetry biases breaks the degeneracy effectively for (1,2) vs (2,1)
    TARGET_STATES = [(1, 1), (1, 2), (2, 1)] 
    
    N_EPOCHS = 5000     
    BATCH_SIZE = 512    
    L_WELL = 3.0        
    
    models, histories = train_spectrum(TARGET_STATES, N_EPOCHS, BATCH_SIZE, L_WELL)
    evaluate_and_plot(models, TARGET_STATES)