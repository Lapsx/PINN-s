import torch
import numpy as np
import matplotlib.pyplot as plt
from unified_pinn_edwards_v31 import UnifiedPINNv31, get_wkb_delta_c

def main():
    # 1. Configurar device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizando dispositivo: {device}")

    # 2. Instanciar a arquitetura do modelo v31 (os hiperparâmetros devem bater com os do treino)
    # Por padrão, no arquivo da v31: u_max_rel=100.0, delta_min=1e-4, delta_max=1e5, ka_min=0.01, ka_max=10.0
    model = UnifiedPINNv31().to(device)
    
    # 3. Carregar os pesos treinados
    model_path = "models/unified_pinn_v31_model.pt"
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modelo {model_path} carregado com sucesso!")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{model_path}' não encontrado. Certifique-se de que o caminho está correto.")
        return

    model.eval()  # Colocar a rede em modo de inferência (desliga dropout/batchnorm se houvesse)

    # 4. Parâmetros de Resolução da Malha (SINTA-SE LIVRE PARA MUDAR AQUI)
    print("Gerando malha para o mapa de fase...")
    res_ka = 50     # Resolução do eixo x (kappa*a)
    res_delta = 60  # Resolução do eixo y (delta)
    
    ka_range = np.logspace(-4, 5, res_ka)
    delta_range = np.logspace(-4, 5, res_delta) 
    
    KA, DELTA = np.meshgrid(ka_range, delta_range)
    MU = np.zeros_like(KA)

    # 5. Avaliação do modelo em Batch (ou ponto a ponto)
    print("Avaliando a rede neural...")
    with torch.no_grad():
        for i in range(len(delta_range)):
            for j in range(len(ka_range)):
                d_t = torch.tensor([[DELTA[i,j]]]).float().to(device)
                k_t = torch.tensor([[KA[i,j]]]).float().to(device)
                
                # A função forward do modelo precisa de u_r (posição), delta e ka.
                # Aqui colocamos um u_r fictício (k_t + 0.1) apenas porque a arquitetura requer, 
                # mas o 'mu_p' (autovalor predito) na v31 depende apenas de delta e ka.
                _, mu_p = model(k_t + 0.1, d_t, k_t)
                MU[i,j] = mu_p.item()

    # 6. Plotagem customizável do Mapa de Fases
    print("Desenhando o gráfico...")
    plt.figure(figsize=(10, 8))
    
    # Você pode mudar o 'cmap' (ex: 'plasma', 'magma', 'inferno', 'coolwarm')
    # Pode mudar os levels também.
    cp = plt.contourf(KA, DELTA, np.log10(MU + 1e-6), levels=30, cmap='viridis')
    plt.colorbar(cp, label='$\\log_{10}(\\mu)$')
    
    # 7. Reconstrução da Linha de Transição PINN
    dc_pinn = []
    print("Calculando a curva crítica da PINN (Busca Binária)...")
    for k in ka_range:
        ka_t = torch.tensor([[k]]).float().to(device)
        dl, dh = 1e-4, 1e6
        for _ in range(30):
            dm = np.sqrt(dl * dh) # busca geométrica
            with torch.no_grad():
                _, mu_val = model(ka_t + 0.1, torch.tensor([[dm]]).float().to(device), ka_t)
            
            # Limiar de transição empírico escolhido (pode alterar se quiser mais restritivo)
            if mu_val.item() > 1e-4: 
                dh = dm 
            else: 
                dl = dm
        dc_pinn.append(dm)
        
    # Plotando as linhas
    plt.plot(ka_range, dc_pinn, 'w--', lw=2.5, label='PINN $\\delta_c$')
    
    # Curva Teórica (WKB)
    dc_wkb = [get_wkb_delta_c(k) for k in ka_range]
    plt.plot(ka_range, dc_wkb, 'r:', lw=2, label='Theory (WKB)')
    
    # Configurações estéticas dos eixos
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-2, 1e1)
    plt.ylim(1e-4, 1e5)
    plt.title('Custom Phase Map: Adsorption Transition')
    plt.xlabel('$\\kappa a$')
    plt.ylabel('$\\delta$')
    plt.legend()
    plt.tight_layout()
    
    # Salvando a imagem
    output_filename = 'phase_map_v31_custom.png'
    plt.savefig(output_filename, dpi=300)
    print(f"Processo concluído! Gráfico salvo como '{output_filename}'.")
    plt.show() # Tenta mostrar na tela se tiver display ativo

if __name__ == "__main__":
    main()
