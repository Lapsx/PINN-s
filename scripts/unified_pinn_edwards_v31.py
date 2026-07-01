import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.special import erfc
import argparse
import time
import json

# Configurações de Hardware e Semente
torch.manual_seed(42)
np.random.seed(42)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Parâmetros Físicos (Cherstvy & Winkler 2011)
params = {
    'delta_min': 0.0001, 'delta_max': 3000.0,
    'ka_min': 0.01, 'ka_max': 10.0,
    'u_max_rel': 12.0,
    'kappa': 0.1,
}

# Constante para transformação logarítmica do u_rel
U_LOG_SCALE = 100.0  # Compressão: espalha a região perto da superfície no espaço de input
U_LOG_NORM = np.log1p(params['u_max_rel'] * U_LOG_SCALE)  # Normalização

def get_args():
    parser = argparse.ArgumentParser(description='PINN Edwards v31 - Refined Density Profiles')
    parser.add_argument('--epochs', type=int, default=60000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--pretrain', type=int, default=50000)
    parser.add_argument('--neurons', type=int, default=256)
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--lbfgs-rounds', type=int, default=50)
    return parser.parse_args()

try: args = get_args()
except:
    class Args: epochs=60000; lr=3e-4; neurons=256; layers=6; pretrain=50000; lbfgs_rounds=50
    args = Args()

# WKB Formula for sphere critical adsorption
def get_wkb_delta_c(ka_val):
    C = 0.973
    if isinstance(ka_val, torch.Tensor):
        ka_np = ka_val.detach().cpu().numpy()
    else:
        ka_np = np.asarray(ka_val, dtype=np.float64)
    num = 6 * ka_np * (1.0 + ka_np) * (C**2)
    den = 2 * np.pi * np.exp(ka_np) * (erfc(np.sqrt(ka_np/2.0))**2)
    return num / (den + 1e-30)

# 2. Solver Numérico Benchmark (resolução adaptativa)
def solve_edwards_numerical(delta_val, ka_val, n_grid=3000):
    """Solver com resolução adaptativa: grade mais fina para picos estreitos."""
    kappa = params['kappa']
    a = ka_val / kappa
    u_min = ka_val
    u_max = u_min + params['u_max_rel'] * 1.5  # Reduzido de 2.0 para melhor resolução
    u = np.linspace(u_min, u_max, n_grid); du = u[1] - u[0]
    pre_factor = (delta_val / (kappa * a)) * (np.exp(ka_val) / (1.0 + ka_val))
    V = - pre_factor * (np.exp(-u) / (u + 1e-12))
    
    # Hamiltoniana H = -d^2/du^2 + V
    H = sp.diags([-1.0/du**2, 2.0/du**2 + V, -1.0/du**2], [-1, 0, 1], shape=(n_grid, n_grid), format='csc')
    try:
        vals, vecs = eigsh(H[1:-1, 1:-1], k=1, which='SA', tol=1e-6)
        mu_val = -vals[0]
        if mu_val <= 1e-6: return u, np.zeros_like(u), 1e-6
        phi = np.zeros(n_grid); phi[1:-1] = np.abs(vecs[:, 0])
        norm = np.trapz(phi**2, u)
        phi = phi / (np.sqrt(norm) + 1e-12) * np.sqrt(kappa)
        return u, phi, mu_val
    except: return u, np.zeros_like(u), 1e-6

# 3. Arquitetura v31 (Log-scale u input + Capped Envelope)
class RFFEncoding(nn.Module):
    """Random Fourier Features com múltiplas escalas."""
    def __init__(self, in_features, out_features, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features) * sigma, requires_grad=False)
    def forward(self, x):
        v = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(v), torch.cos(v)], dim=-1)

class ModifiedMLP(nn.Module):
    def __init__(self, in_f, h_f, layers=6):
        super().__init__()
        self.enc = RFFEncoding(in_f, 128)
        self.U = nn.Linear(256, h_f)
        self.V = nn.Linear(256, h_f)
        self.hidden = nn.ModuleList([nn.Linear(h_f, h_f) for _ in range(layers)])
        self.out = nn.Linear(h_f, 1)
        self.act = nn.Tanh()
        
    def forward(self, x):
        e = self.enc(x)
        u_gate = self.act(self.U(e))
        v_gate = self.act(self.V(e))
        h = u_gate
        for l in self.hidden:
            z = torch.sigmoid(l(h))
            h = (1 - z) * u_gate + z * v_gate
        return self.out(h)

class UnifiedPINNv31(nn.Module):
    def __init__(self, neurons=256, layers=6):
        super().__init__()
        self.phi_net = ModifiedMLP(3, neurons, layers=layers)
        self.mu_net = nn.Sequential(
            nn.Linear(2, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        
    def forward(self, u_abs, d, ka):
        if d.shape[0] != u_abs.shape[0]: d = d.expand(u_abs.shape[0], -1)
        if ka.shape[0] != u_abs.shape[0]: ka = ka.expand(u_abs.shape[0], -1)
        
        u_rel = torch.clamp(u_abs - ka, min=0.0)
        
        # Transformação logarítmica do u_rel -> resolve problema de resolução perto da superfície
        # u_rel = 0.001 -> un ≈ 0.014 (antes: 0.00008)
        # u_rel = 0.01  -> un ≈ 0.095 (antes: 0.0008)
        # u_rel = 0.1   -> un ≈ 0.338 (antes: 0.008)
        # u_rel = 1.0   -> un ≈ 0.651 (antes: 0.083)
        un = torch.log1p(u_rel * U_LOG_SCALE) / U_LOG_NORM
        un = torch.clamp(un, 0.0, 1.0)
        
        # Normalização logarítmica de delta e ka
        dn = (torch.log10(torch.clamp(d, min=params['delta_min'])) - np.log10(params['delta_min'])) / (np.log10(params['delta_max']) - np.log10(params['delta_min']))
        kan = (torch.log10(torch.clamp(ka, min=params['ka_min'])) - np.log10(params['ka_min'])) / (np.log10(params['ka_max']) - np.log10(params['ka_min']))
        
        # Rede mu: prediz log10(mu)
        log_mu = self.mu_net(torch.cat([dn, kan], dim=1))
        mu = torch.pow(10.0, torch.clamp(log_mu, -6.0, 8.0))
        
        # Rede phi: prediz amplitude crua (positiva via softplus)
        p_raw = self.phi_net(torch.cat([un, dn, kan], dim=1))
        phi_raw = F.softplus(p_raw)
        
        # Envelope com alpha LIMITADO para não suprimir picos estreitos
        # Alpha > 3 matava completamente os picos para delta grande / ka pequeno
        alpha = torch.clamp(torch.sqrt(mu.detach() + 0.1), max=3.0)
        bc = 1.0 - torch.exp(-torch.clamp(50.0 * u_rel, max=30.0))  # BC mais íngreme (50 vs 20)
        decay = torch.exp(-torch.clamp(alpha * u_rel, max=30.0))
        env = bc * decay
        phi = env * phi_raw
        return phi, mu

# 4. Motor de Treinamento Refinado
def train():
    model = UnifiedPINNv31(neurons=args.neurons, layers=args.layers).to(device)
    
    # Histórico de Loss
    history = {'epoch': [], 'total': [], 'pde': [], 'norm': [], 'anchor': [], 'unad': []}

    print(f"Pre-calculando âncoras (Grade 10x10 com resolução 5k)...")
    anchors = []
    grid_ka = np.logspace(-2, 1, 10)
    grid_delta = np.logspace(-2, 3.7, 10)
    for ka in grid_ka:
        for d in grid_delta:
            u_n, phi_n, mu_n = solve_edwards_numerical(d, ka)
            mask = (u_n - ka) < 8.0
            anchors.append({
                'ka': torch.tensor([[ka]]).float().to(device),
                'delta': torch.tensor([[d]]).float().to(device),
                'u': torch.tensor(u_n[mask]).float().view(-1, 1).to(device),
                'phi': torch.tensor(phi_n[mask]).float().view(-1, 1).to(device),
                'mu': torch.tensor([[mu_n]]).float().to(device)
            })

    def sample_collocation():
        """Amostragem adaptativa com bias perto da superfície."""
        d_val = 10**(np.random.uniform(-2.0, 3.7, 10))
        ka_val = 10**(np.random.uniform(-2.0, 1.0, 10))
        d_b = torch.tensor(d_val).float().to(device).view(-1, 1).repeat(1, 150).view(-1, 1)
        ka_b = torch.tensor(ka_val).float().to(device).view(-1, 1).repeat(1, 150).view(-1, 1)
        
        # 50% pontos muito perto da superfície (u_rel in [0, 2.0])
        # 50% pontos no domínio completo (u_rel in [0, u_max_rel])
        n_half = 750
        u_near = ka_b[:n_half] + 2.0 * (torch.rand(n_half, 1).to(device)**2.0)
        u_far = ka_b[n_half:] + params['u_max_rel'] * (torch.rand(n_half, 1).to(device)**2.0)
        u_b = torch.cat([u_near, u_far], dim=0)
        return u_b, d_b, ka_b

    def compute_loss(u, d, ka, sampled_anchors=None):
        u.requires_grad_(True)
        phi, mu = model(u, d, ka)
        
        # 1. Resíduo da PDE: phi_uu - (mu + V) * phi = 0
        phi_u = torch.autograd.grad(phi, u, torch.ones_like(phi), create_graph=True)[0]
        phi_uu = torch.autograd.grad(phi_u, u, torch.ones_like(phi_u), create_graph=True)[0]
        
        a = ka / params['kappa']
        pre_factor = (d / (params['kappa'] * a)) * (torch.exp(ka) / (1.0 + ka))
        V_u = - pre_factor * (torch.exp(-torch.clamp(u, min=1e-9)) / (torch.clamp(u, min=1e-9)))
        
        res_raw = phi_uu - (mu + V_u) * phi
        residual = res_raw / (1.0 + torch.log1p(torch.abs(V_u).detach()))
        l_pde = torch.mean(residual**2)
        
        # 2. Normalização Global condicional
        d_unique = d.view(-1, 150)[:, 0:1]
        ka_unique = ka.view(-1, 150)[:, 0:1]
        u_grid = (torch.linspace(0, 1, 500).to(device)**2.0) * params['u_max_rel']
        n_batch = d_unique.shape[0]
        
        ui_batch = ka_unique + u_grid.view(1, 500)
        di_batch = d_unique.expand(n_batch, 500)
        kai_batch = ka_unique.expand(n_batch, 500)
        
        phi_flat, _ = model(ui_batch.reshape(-1, 1), di_batch.reshape(-1, 1), kai_batch.reshape(-1, 1))
        phi_batch = phi_flat.view(n_batch, 500)
        
        phi_sq = phi_batch**2
        du = ui_batch[:, 1:] - ui_batch[:, :-1]
        phi_mid = (phi_sq[:, 1:] + phi_sq[:, :-1]) / 2.0
        integrals = torch.sum(phi_mid * du, dim=1)
        
        ka_unique_np = ka_unique.detach().cpu().numpy().flatten()
        d_unique_np = d_unique.detach().cpu().numpy().flatten()
        dc_unique = get_wkb_delta_c(ka_unique_np)
        
        targets = np.where(d_unique_np > dc_unique, params['kappa'], 0.0)
        targets_t = torch.tensor(targets).float().to(device)
        
        l_norm = torch.mean((integrals - targets_t)**2)
        
        # 3. Penalização para região não-adsorvida
        dc_col = torch.tensor(get_wkb_delta_c(ka.detach().cpu().numpy())).float().to(device)
        is_adsorbed = (d > dc_col).float()
        l_unad_phi = torch.mean(((1.0 - is_adsorbed) * phi)**2)
        l_unad_mu = torch.mean(((1.0 - is_adsorbed) * (torch.log10(mu + 1e-8) - (-6.0)))**2)
        l_unad = 10000.0 * l_unad_phi + 100.0 * l_unad_mu
        
        # 4. Âncoras com loss relativo para melhor matching de amplitude
        l_anchor = 0
        target_anchors = sampled_anchors if sampled_anchors is not None else anchors
        for anc in target_anchors:
            p_a, m_a = model(anc['u'], anc['delta'], anc['ka'])
            # Loss absoluto do perfil
            l_anchor += 50000.0 * F.mse_loss(p_a, anc['phi'])
            # Loss relativo do perfil para capturar forma (não apenas amplitude)
            phi_max = torch.clamp(anc['phi'].max(), min=1e-6)
            l_anchor += 10000.0 * F.mse_loss(p_a / phi_max, anc['phi'] / phi_max)
            # Loss do autovalor (em log)
            l_anchor += 200.0 * F.mse_loss(torch.log10(m_a[:1] + 1e-7), torch.log10(anc['mu'] + 1e-7))
            
        return l_pde, 20000.0 * l_norm, l_anchor / len(target_anchors), l_unad

    # Estágio 1: Adam com cosine annealing
    print(f"Estágio 1: Adam Training ({args.pretrain} épocas)...")
    opt_adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_adam, T_0=10000, T_mult=2, eta_min=1e-5)
    
    t0 = time.time()
    for epoch in range(args.pretrain):
        opt_adam.zero_grad()
        u_b, d_b, ka_b = sample_collocation()
        
        sampled = np.random.choice(anchors, 8, replace=False)
        lp, ln, la, lu = compute_loss(u_b, d_b, ka_b, sampled_anchors=sampled)
        loss = lp + ln + la + lu
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt_adam.step()
        scheduler.step()
        
        if epoch % 1000 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (epoch + 1) * (args.pretrain - epoch - 1)
            lr_now = opt_adam.param_groups[0]['lr']
            print(f"E {epoch:5} | Loss: {loss.item():.4f} | PDE: {lp.item():.2e} | Norm: {ln.item():.2e} | "
                  f"Anc: {la.item():.2e} | Unad: {lu.item():.2e} | lr: {lr_now:.2e} | ETA: {eta/60:.0f}min")
            history['epoch'].append(epoch)
            history['total'].append(loss.item())
            history['pde'].append(lp.item())
            history['norm'].append(ln.item())
            history['anchor'].append(la.item())
            history['unad'].append(lu.item())
            if epoch % 20000 == 0 and epoch > 0: 
                torch.save(model.state_dict(), f"unified_pinn_v31_checkpoint_{epoch}.pt")

    # Estágio 2: L-BFGS em rondas com reamostragem
    print(f"Estágio 2: L-BFGS em {args.lbfgs_rounds} rondas...")
    for round_idx in range(args.lbfgs_rounds):
        # Reamostragem fresca a cada ronda para evitar estagnação
        u_b, d_b, ka_b = sample_collocation()
        
        # Novo otimizador L-BFGS para cada ronda (curvatura limpa)
        opt_lbfgs = torch.optim.LBFGS(model.parameters(), lr=0.005, max_iter=20, 
                                        line_search_fn='strong_wolfe', history_size=50)
        
        def closure():
            opt_lbfgs.zero_grad()
            lp, ln, la, lu = compute_loss(u_b, d_b, ka_b, sampled_anchors=anchors)
            l = lp + ln + la + lu
            l.backward()
            return l
        
        l_val = opt_lbfgs.step(closure)
        
        if round_idx % 5 == 0:
            print(f"L-BFGS Round {round_idx:3}/{args.lbfgs_rounds} | Loss: {l_val.item():.4f}")
            history['epoch'].append(args.pretrain + round_idx * 20)
            history['total'].append(l_val.item())
        
        if torch.isnan(l_val): 
            print("NaN detectado, parando L-BFGS.")
            break

    torch.save(model.state_dict(), "unified_pinn_v31_model.pt")
    with open("training_history_v31.json", "w") as f:
        json.dump(history, f)
    return model, history

# 5. Visualização e Gráficos
def generate_all_plots(model, history):
    model.eval()
    
    # 5.1 Gráfico de Histórico de Treino
    if len(history.get('epoch', [])) > 0:
        plt.figure(figsize=(10, 6))
        plt.semilogy(history['epoch'], history['total'], label='Total Loss', lw=1.5)
        if 'pde' in history and len(history['pde']) > 0:
            n = min(len(history['epoch']), len(history['pde']))
            plt.semilogy(history['epoch'][:n], history['pde'][:n], alpha=0.5, label='PDE')
            plt.semilogy(history['epoch'][:n], history['norm'][:n], alpha=0.5, label='Norm')
            plt.semilogy(history['epoch'][:n], history['anchor'][:n], alpha=0.5, label='Anchor')
        plt.title('Training Convergence History (v31)')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.2)
        plt.savefig('training_loss_v31.png', dpi=300)
        plt.close()

    # 5.2 Gráficos de Perfil de Densidade
    print("Gerando gráficos de densidade...")
    for ka_val in [0.1, 1.0, 5.0]:
        plt.figure(figsize=(10, 6))
        a = ka_val / 0.1
        for d in [30, 100, 300, 1000, 3000]:
            u_n, p_n, mu_n = solve_edwards_numerical(d, ka_val)
            dist_surface = (u_n - ka_val) / 0.1
            u_t = torch.tensor(u_n).float().view(-1, 1).to(device)
            p_p, mu_p = model(u_t, torch.tensor([[d]]).float().to(device), torch.tensor([[ka_val]]).float().to(device))
            p_pinn = p_p.detach().cpu().numpy().flatten()
            plt.plot(dist_surface, p_n**2, ':', color='gray', alpha=0.4, linewidth=1.5)
            line, = plt.plot(dist_surface, p_pinn**2, label=f'$\\delta$={d} (μ={mu_p[0].item():.1f})')
        plt.title(f'Density Profiles | $\\kappa a$={ka_val} (a={a:.0f}Å)')
        plt.xlabel('Distance from surface $r-a$ (Å)')
        plt.ylabel('P(r)')
        plt.xlim(0, 25); plt.legend(fontsize=9); plt.grid(alpha=0.1)
        plt.tight_layout()
        plt.savefig(f'hf_pinn_ka_{ka_val}_v31.png', dpi=300)
        plt.close()

    # 5.3 Mapa de Fase (delta de 1e-4 a 1e5)
    print("Gerando mapa de fase...")
    plt.figure(figsize=(10, 8))
    ka_range = np.logspace(-2, 1, 50)
    delta_range = np.logspace(-4, 5, 60)  # Agora de 1e-4 a 1e5
    KA, DELTA = np.meshgrid(ka_range, delta_range)
    MU = np.zeros_like(KA)
    with torch.no_grad():
        for i in range(len(delta_range)):
            for j in range(len(ka_range)):
                d_t = torch.tensor([[DELTA[i,j]]]).float().to(device)
                k_t = torch.tensor([[KA[i,j]]]).float().to(device)
                _, mu_p = model(k_t + 0.1, d_t, k_t)
                MU[i,j] = mu_p.item()
    
    cp = plt.contourf(KA, DELTA, np.log10(MU + 1e-6), levels=30, cmap='viridis')
    plt.colorbar(cp, label='$\\log_{10}(\\mu)$')
    
    # Linha de transição delta_c
    dc_pinn = []
    for k in ka_range:
        ka_t = torch.tensor([[k]]).float().to(device)
        dl, dh = 1e-4, 1e6
        for _ in range(30):
            dm = np.sqrt(dl * dh)
            with torch.no_grad():
                _, mu_val = model(ka_t + 0.1, torch.tensor([[dm]]).float().to(device), ka_t)
            if mu_val.item() > 1e-4: dh = dm 
            else: dl = dm
        dc_pinn.append(dm)
    plt.plot(ka_range, dc_pinn, 'w--', lw=2.5, label='PINN $\\delta_c$')
    dc_wkb = [get_wkb_delta_c(k) for k in ka_range]
    plt.plot(ka_range, dc_wkb, 'r:', lw=2, label='Theory (WKB)')
    plt.xscale('log'); plt.yscale('log')
    plt.xlim(1e-2, 1e1); plt.ylim(1e-4, 1e5)
    plt.title('Phase Map: Adsorption Transition (v31)')
    plt.xlabel('$\\kappa a$'); plt.ylabel('$\\delta$'); plt.legend()
    plt.tight_layout()
    plt.savefig('phase_map_v31.png', dpi=300)
    plt.close()

    # 5.4 Adsorção Crítica (delta de 1e-4 a 1e5)
    print("Gerando curva de adsorção crítica...")
    plt.figure(figsize=(8, 6))
    plt.loglog(ka_range, dc_pinn, 'ro-', markersize=3, label='PINN $\\delta_c$')
    plt.loglog(ka_range, dc_wkb, 'k--', alpha=0.6, label='Theoretical Trend (WKB)')
    plt.xlim(1e-2, 1e1); plt.ylim(1e-4, 1e5)
    plt.title('Critical Adsorption $\\delta_c$ vs $\\kappa a$ (v31)')
    plt.xlabel('$\\kappa a$'); plt.ylabel('$\\delta_c$')
    plt.grid(True, which="both", ls="-", alpha=0.1); plt.legend()
    plt.tight_layout()
    plt.savefig('critical_adsorption_v31.png', dpi=300)
    plt.close()

    # 5.5 NOVO: Autovalor mu vs kappa*a para diferentes deltas
    print("Gerando gráfico de autovalor mu vs ka...")
    plt.figure(figsize=(10, 7))
    ka_plot = np.logspace(-2, 1, 40)
    delta_values = [10, 30, 100, 300, 1000, 3000]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(delta_values)))
    
    for idx, d in enumerate(delta_values):
        mu_pinn_list = []
        mu_num_list = []
        ka_valid = []
        
        for k in ka_plot:
            # PINN
            with torch.no_grad():
                ka_t = torch.tensor([[k]]).float().to(device)
                d_t = torch.tensor([[d]]).float().to(device)
                _, mu_p = model(ka_t + 0.1, d_t, ka_t)
                mu_pinn_val = mu_p.item()
            
            # Numérico
            _, _, mu_n = solve_edwards_numerical(d, k)
            
            # Só plota se está na região adsorvida
            dc = get_wkb_delta_c(k)
            if d > dc * 0.5:  # Incluir perto da transição
                ka_valid.append(k)
                mu_pinn_list.append(mu_pinn_val)
                mu_num_list.append(mu_n)
        
        if len(ka_valid) > 0:
            plt.semilogy(ka_valid, mu_pinn_list, '-', color=colors[idx], lw=2, 
                        label=f'PINN $\\delta$={d}')
            plt.semilogy(ka_valid, mu_num_list, 'o', color=colors[idx], markersize=4, 
                        alpha=0.5, markerfacecolor='none')
    
    # Legenda combinada
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='gray', lw=2, label='PINN (linhas)'),
                       Line2D([0], [0], marker='o', color='gray', markersize=5, 
                              markerfacecolor='none', linestyle='None', label='Numérico (símbolos)')]
    for idx, d in enumerate(delta_values):
        legend_elements.append(Line2D([0], [0], color=colors[idx], lw=2, label=f'$\\delta$={d}'))
    
    plt.legend(handles=legend_elements, fontsize=8, ncol=2, loc='upper left')
    plt.xlabel('$\\kappa a$'); plt.ylabel('$\\mu$ (autovalor)')
    plt.title('Eigenvalue $\\mu$ vs $\\kappa a$ for Different $\\delta$ (v31)')
    plt.grid(True, which="both", ls="-", alpha=0.1)
    plt.xlim(1e-2, 1e1)
    plt.tight_layout()
    plt.savefig('eigenvalue_vs_ka_v31.png', dpi=300)
    plt.close()
    
    # 5.6 Gráfico de comparação quantitativa mu PINN vs mu numérico
    print("Gerando gráfico de paridade mu...")
    plt.figure(figsize=(7, 7))
    mu_pinn_all, mu_num_all = [], []
    for ka in np.logspace(-2, 1, 10):
        for d in np.logspace(-1, 3.5, 10):
            dc = get_wkb_delta_c(ka)
            if d > dc:
                with torch.no_grad():
                    ka_t = torch.tensor([[ka]]).float().to(device)
                    d_t = torch.tensor([[d]]).float().to(device)
                    _, mu_p = model(ka_t + 0.1, d_t, ka_t)
                _, _, mu_n = solve_edwards_numerical(d, ka)
                if mu_n > 1e-4:
                    mu_pinn_all.append(mu_p.item())
                    mu_num_all.append(mu_n)
    
    mu_pinn_arr = np.array(mu_pinn_all)
    mu_num_arr = np.array(mu_num_all)
    plt.loglog(mu_num_arr, mu_pinn_arr, 'o', markersize=4, alpha=0.5)
    lims = [min(mu_num_arr.min(), mu_pinn_arr.min()) * 0.5, max(mu_num_arr.max(), mu_pinn_arr.max()) * 2]
    plt.loglog(lims, lims, 'k--', alpha=0.5, label='Paridade')
    plt.xlabel('$\\mu$ Numérico'); plt.ylabel('$\\mu$ PINN')
    plt.title('Parity Plot: $\\mu_{PINN}$ vs $\\mu_{numerical}$ (v31)')
    plt.legend(); plt.grid(True, alpha=0.1); plt.axis('equal')
    plt.xlim(lims); plt.ylim(lims)
    plt.tight_layout()
    plt.savefig('parity_mu_v31.png', dpi=300)
    plt.close()
    
    print("Todos os gráficos v31 gerados!")

if __name__ == "__main__":
    if os.path.exists("unified_pinn_v31_model.pt") and args.epochs < 10:
        model = UnifiedPINNv31(neurons=args.neurons, layers=args.layers).to(device)
        model.load_state_dict(torch.load("unified_pinn_v31_model.pt", map_location=device))
        print("Modelo v31 carregado.")
        generate_all_plots(model, {'epoch': [], 'total': []})
    else:
        model, history = train()
        generate_all_plots(model, history)
