import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os


# DATASET
def generate_and_save_noisy_dataset(num_states, L, filename, points_per_state, noise_level):
    """
    Generates a noisy dataset for the Infinite Potential Well.
    """
    data_dict = {}
    rng = np.random.default_rng(42)
    for n in range(1, num_states + 1):
        x_sampled = rng.uniform(-L / 2.0, L / 2.0, points_per_state)
        psi_sampled = np.sqrt(2.0 / L) * np.sin(n * np.pi / L * (x_sampled + L / 2.0))
        # Gaussian noise 
        if noise_level > 0.0:
            psi_sampled += rng.normal(0.0, noise_level, size=psi_sampled.shape)

        data_dict[f"x_n{n}"] = x_sampled.astype(np.float32)
        data_dict[f"psi_n{n}"] = psi_sampled.astype(np.float32)

    np.savez(filename, **data_dict)
    print(f"--> Successfully created reference dataset: '{filename}' ({points_per_state} points per state with noise level {noise_level})\n")


# PINN CLASS
class APINN(nn.Module):
    def __init__(self, L):
        super(APINN, self).__init__()
        self.L = L

        self.hidden_layer1 = nn.Linear(1, 64)
        self.hidden_layer2 = nn.Linear(64, 64)
        self.hidden_layer3 = nn.Linear(64, 64)
        self.hidden_layer4 = nn.Linear(64, 64)
        self.hidden_layer5 = nn.Linear(64, 64)
        self.hidden_layer6 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 2)

        self.E_net = nn.Linear(1, 1, bias=False)
        #nn.init.constant_(self.E_net.bias, 8.5)

        self.optimizer = torch.optim.Adam(list(self.parameters()) + list(self.E_net.parameters()), lr=0.01)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9995)

        self.mse = nn.MSELoss()
        self.sae = nn.L1Loss(reduction='sum')

    def forward(self, x):
        x_out = torch.tanh(self.hidden_layer1(x))
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

    def diff_eq(self, x_colloc, psi, E):
        psi_x = torch.autograd.grad(psi, x_colloc, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x, x_colloc, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]

        pde_residual = -0.5 * psi_xx - E * psi
        loss_pde = self.mse(pde_residual, torch.zeros_like(pde_residual))
        return loss_pde

    def int_norm(self, x_colloc, x_bound_0, x_bound_1, psi, v):
        v_x = torch.autograd.grad(v, x_colloc, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        loss_int = self.mse(v_x, psi**2)

        _, v_0 = self.forward(x_bound_0)
        _, v_1 = self.forward(x_bound_1)
        loss_norm = self.sae(v_0, torch.zeros_like(v_0)) + self.sae(v_1, torch.ones_like(v_1))
        return loss_int, loss_norm

    def bc(self, x_bound_0, x_bound_1):
        psi_b0, _ = self.forward(x_bound_0)
        psi_b1, _ = self.forward(x_bound_1)
        loss_bc = self.sae(psi_b0, torch.zeros_like(psi_b0)) + self.sae(psi_b1, torch.zeros_like(psi_b1))
        return loss_bc

    def ortho(self, previous_models, x_colloc, psi):
        loss_ortho = torch.tensor(0.0, device=x_colloc.device)
        for prev_model in previous_models:
            with torch.no_grad():
                psi_prev, _ = prev_model(x_colloc)
            overlap = torch.mean(psi * psi_prev)
            loss_ortho = loss_ortho + self.mse(overlap, torch.zeros_like(overlap))
        return loss_ortho

    def symm(self, x_colloc, psi, n_state):
        s = 1.0 if (n_state % 2 == 1) else -1.0
        psi_neg, _ = self.forward(-x_colloc)
        loss_symmetry = self.mse(psi - s * psi_neg, torch.zeros_like(psi))
        return loss_symmetry

    def emin(self, E, E_init_val):
        loss_emin = torch.mean(torch.exp(0.8 * (E - E_init_val)))
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

        loss_pde = self.diff_eq(x_colloc, psi, E)
        w_pde = initial_weights_losses[0] * np.log10(10+1e2*epoch)

        loss_int, loss_norm = self.int_norm(x_colloc, x_bound_0, x_bound_1, psi, v)
        w_int = initial_weights_losses[1]
        w_norm = initial_weights_losses[2]

        loss_bc = self.bc(x_bound_0, x_bound_1)
        w_bc = initial_weights_losses[3]

        loss_ortho = self.ortho(previous_models, x_colloc, psi)
        w_ortho = initial_weights_losses[4]

        loss_symmetry = self.symm(x_colloc, psi, n_state)
        w_symm = initial_weights_losses[5]

        loss_emin = self.emin(E, E_init_val)
        w_emin = initial_weights_losses[6] * np.exp(-epoch/1000)

        loss_data = self.data_loss(dataset, n_state, x_colloc)
        w_data = initial_weights_losses[7] 

        total_loss = w_pde*loss_pde + w_int*loss_int + w_norm*loss_norm + w_bc*loss_bc + w_ortho*loss_ortho + w_symm*loss_symmetry + w_emin*loss_emin + w_data*loss_data

        total_loss.backward()
        self.optimizer.step()

        return (total_loss.item(), loss_pde.item(), loss_int.item(), loss_norm.item(), loss_bc.item(),
               loss_ortho.item(), loss_symmetry.item(), loss_emin.item(), loss_data.item(), E.mean().item())

# SAVE AND LOAD MODELS
def save_model(model, state_n, points_per_state, batch_size, L, weights_losses, save_dir="Progetto/saved_states"):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"INF_MOD_final_L({L})_PPS({points_per_state})_BS({batch_size})_pde({weights_losses[0]})_int({weights_losses[1]})_norm({weights_losses[2]})_bc({weights_losses[3]})_ortho({weights_losses[4]})_symm({weights_losses[5]})_emin({weights_losses[6]})_data({weights_losses[7]})_{state_n}.pt"
    filepath = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), filepath)
    print(f"Saved state n={state_n} to {filepath}")

def load_previous_models(num_previous, L, points_per_state, batch_size, weights_losses, save_dir="Progetto/saved_states"):
    models = []
    print(f"\n--- Loading {num_previous} previously trained states ---")
    for i in range(1, num_previous + 1):
        filename = f"INF_MOD_final_L({L})_PPS({points_per_state})_BS({batch_size})_pde({weights_losses[0]})_int({weights_losses[1]})_norm({weights_losses[2]})_bc({weights_losses[3]})_ortho({weights_losses[4]})_symm({weights_losses[5]})_emin({weights_losses[6]})_data({weights_losses[7]})_{i}.pt"
        filepath = os.path.join(save_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find saved model for state {i} at {filepath}")

        model = APINN(L)
        model.load_state_dict(torch.load(filepath))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
        print(f"Loaded state n={i} (E ≈ {model.get_energy():.4f})")
    return models

# TRAINING
def train_spectrum(start_state_idx, target_states, n_epochs, batch_size, L, points_per_state, initial_weights_losses, loaded_models):
    discovered_models = loaded_models.copy() if loaded_models else []
    all_loss_histories = []

    data_file = np.load("quantum_noisy_data.npz") if points_per_state != 0 else None

    x_b0 = torch.tensor([[-L / 2.0]], requires_grad=True)
    x_b1 = torch.tensor([[L / 2.0]], requires_grad=True)

    if discovered_models:
        current_E_init = discovered_models[-1].get_energy() + (start_state_idx+1)**2 - (start_state_idx)**2 - 2.0
    else:
        current_E_init = 0.0

    base_mesh = np.linspace(-L / 2.0, L / 2.0, batch_size)
    std_dev = (L / batch_size) * 0.5

    for state_idx in range(start_state_idx, target_states):
        n_level = state_idx + 1
        print(f"\n=======================================================")
        print(f" TRAINING STATE n={n_level} (Targeting E > {current_E_init:.2f})")
        print(f"=======================================================")

        model = APINN(L)
        keys = ["total", "pde", "int", "norm", "bc", "ortho", "symmetry", "emin", "data", "E", "err_E", "fidelity"]
        state_loss = {k: [] for k in keys}

        if (n_level >= 4 and n_level <= 6): n_epochs_curr = 2*n_epochs
        elif (n_level > 6 and n_level <= 9): n_epochs_curr = 5*n_epochs
        else: n_epochs_curr = n_epochs

        for epoch in range(n_epochs_curr):
            sampled_points = np.random.normal(loc=base_mesh, scale=std_dev)
            sampled_points = np.clip(sampled_points, -L / 2.0, L / 2.0)

            x_colloc = torch.tensor(sampled_points, dtype=torch.float32).reshape(-1, 1)
            x_colloc.requires_grad = True

            losses = model.compute_losses(x_colloc, x_b0, x_b1, epoch, discovered_models, current_E_init, n_level, initial_weights_losses, dataset=data_file)
            model.scheduler.step()

            if (epoch + 1) % 10 == 0:
                # Calculate losses and metrics
                with torch.no_grad():
                    x_test = torch.linspace(-L / 2.0, L / 2.0, 500).reshape(-1, 1)
                    x_np = x_test.numpy().flatten()

                    psi_pred_test, _ = model(x_test)
                    psi_pred_test = psi_pred_test.numpy().flatten()
                    E_pred_test = model.get_energy()

                    E_th = (n_level**2 * np.pi**2) / (2.0 * L**2)
                    psi_theory = np.sqrt(2.0 / L) * np.sin(n_level * np.pi / L * (x_np + L / 2.0))

                    # Normalize for fidelity step
                    norm_const = np.trapezoid(psi_pred_test**2, x=x_np)
                    if norm_const > 0:
                        psi_pred_test /= np.sqrt(norm_const)
                    if np.sum(psi_pred_test * psi_theory) < 0:
                        psi_pred_test = -psi_pred_test

                    err_E = (E_th - E_pred_test) / E_th

                    v_pred_norm = psi_pred_test / (np.linalg.norm(psi_pred_test) + 1e-10)
                    v_theory_norm = psi_theory / (np.linalg.norm(psi_theory) + 1e-10)
                    fidelity = np.abs(np.vdot(v_theory_norm, v_pred_norm))**2

                # Combine losses and metrics for tracking
                losses_extended = list(losses) + [err_E, fidelity]
                for i, k in enumerate(keys):
                    state_loss[k].append(losses_extended[i])

            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch+1:5d} | E: {losses[9]:.4f} | "
                    f"Tot: {losses[0]:.3f} | PDE: {losses[1]:.3f} | "
                    f"Int: {losses[2]:.2e} | Norm: {losses[3]:.2e} | BC: {losses[4]:.2e} | "
                    f"Symm: {losses[6]:.2e} | Ortho: {losses[5]} | Emin: {losses[7]:.2e} | Data: {losses[8]} | "
                    f"Err E: {err_E:.2e} | Fid: {fidelity:.4f}")

            

        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        discovered_models.append(model)
        all_loss_histories.append(state_loss)

        save_model(model, n_level, points_per_state, batch_size, L, initial_weights_losses)
        current_E_init = losses[9] + (n_level + 1)**2 - n_level**2 - 2.0

    return discovered_models, all_loss_histories

# TEST AND PLOT
def evaluate_and_plot(discovered_models, histories, dataset=None):
    num_states = len(discovered_models)
    L = discovered_models[0].L
    x_test = torch.linspace(-L / 2.0, L / 2.0, 500).reshape(-1, 1)
    x_np = x_test.numpy().flatten()

    fig, axes = plt.subplots(num_states, 1, figsize=(8, 3.5 * num_states))
    if num_states == 1: axes = [axes]

    print("\n--- Generating Metric Comparisons & Graphical Plots ---")

    with torch.no_grad():
        for i, model in enumerate(discovered_models):
            n_level = i + 1
            psi_pred, _ = model(x_test)
            psi_pred = psi_pred.numpy().flatten()
            E_pred = model.get_energy()

            E_th = (n_level**2 * np.pi**2) / (2.0 * L**2)
            psi_theory = np.sqrt(2.0 / L) * np.sin(n_level * np.pi / L * (x_np + L / 2.0))

            # normalization for plotting
            psi_pred /= np.sqrt(np.trapezoid(psi_pred**2, x=x_np))
            if np.sum(psi_pred * psi_theory) < 0: psi_pred = -psi_pred
            # metrics
            err_E = (E_th - E_pred) / E_th
            v_pred_norm = psi_pred / np.linalg.norm(psi_pred)
            v_theory_norm = psi_theory / np.linalg.norm(psi_theory)
            fidelity = np.abs(np.vdot(v_theory_norm, v_pred_norm))**2

            # Plot Final Wavefunction
            ax = axes[i]

            # Plot dataset
            if dataset is not None and f"x_n{n_level}" in dataset:
                x_d = dataset[f"x_n{n_level}"]
                psi_d = dataset[f"psi_n{n_level}"]
                ax.scatter(x_d, psi_d, color='black', marker='o', s=50, label='Training Data', zorder=5)

            ax.plot(x_np, psi_pred, color='red', label=f"PINN", markevery=10, linestyle='-')
            ax.plot(x_np, psi_theory, color='blue', linestyle='--', alpha=0.7, label=f"Ground Truth")
            ax.axvline(x=-L/2.0, color='green', linestyle='-', linewidth=2)
            ax.axvline(x=L/2.0, color='green', linestyle='-')
            ax.set_title(f"State n = {n_level} \n Pred Energy = {E_pred:.4f} (Exact = {E_th:.4f}) \n Final Err E = {err_E:.2e} | $\mathcal{{F}}_{{\psi}}$ = {fidelity:.6f}")
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    plt.show()

    # plot loss and metrics
    if histories is not None:

        for i, history in enumerate(histories):
            epochs = np.arange(1, len(history["total"]) + 1) * 10

            # plot all Losses
            plt.figure(figsize=(8, 5))
            for k, v in history.items():
                if k not in ["err_E", "fidelity", "E"]:
                    plt.plot(epochs, v, label=k)
            plt.yscale('log')
            plt.title(f"State n={i+1} Loss Components")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.show()
            #plot total loss only
            plt.figure(figsize=(8, 5))
            for k, v in history.items():
                if k in ["total"]:
                    plt.plot(epochs, v, label=k)
            plt.yscale('log')
            plt.title(f"State n={i+1} Loss Components")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid(True, alpha=0.2)
            plt.show()

            # plot error and fidelity
            fig_metrics, (ax_err, ax_fid) = plt.subplots(1, 2, figsize=(12, 4))
            ax_err.plot(epochs, history["err_E"], color='orange', label='$err_E$', linewidth=2)
            ax_err.set_title(f"State n={i+1} Relative Error of Energy ($err_E$)")
            ax_err.set_xlabel("Epoch")
            ax_err.set_ylabel("$err_E$")
            ax_err.grid(True, alpha=0.5)

            ax_fid.plot(epochs, history["fidelity"], color='purple', label='$\mathcal{F}_{\psi}$', linewidth=2)
            ax_fid.set_title(f"State n={i+1} Fidelity$")
            ax_fid.set_xlabel("Epoch")
            ax_fid.set_ylabel("Fidelity")
            ax_fid.grid(True, alpha=0.5)

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    N_EPOCHS = 8000
    BATCH_SIZE = 512
    L_WELL = 3.0
    POINTS_PER_STATE = 5

    generate_and_save_noisy_dataset(num_states=15, L=L_WELL, points_per_state = POINTS_PER_STATE, noise_level=1.0, filename="quantum_noisy_data.npz")

    # Load dataset to pass it for plotting
    data_file_plot = np.load("quantum_noisy_data.npz") if POINTS_PER_STATE != 0 else None

    INITIAL_WEIGHTS_LOSSES = [2.0, 5000.0, 1000.0, 10.0, 1000.0, 1000.0, 10.0, 1000.0]
    loaded_models = load_previous_models(num_previous=0, L=L_WELL, points_per_state=POINTS_PER_STATE, batch_size=BATCH_SIZE, weights_losses=INITIAL_WEIGHTS_LOSSES)

    all_models, all_histories = train_spectrum(start_state_idx=0, target_states=4, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, L=L_WELL, points_per_state=POINTS_PER_STATE, initial_weights_losses=INITIAL_WEIGHTS_LOSSES, loaded_models=loaded_models)

    evaluate_and_plot(all_models, all_histories, dataset=data_file_plot)