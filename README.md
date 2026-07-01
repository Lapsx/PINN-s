# PINN-s (Parametric PINNs for Polymer Physics)

*Read this in other languages: [English](#english) | [Português](#português)*

---

## English

> **First Stage of the Master's Project** 
> *Modeling Polymers on Homogeneously Charged Surfaces using Physics-Informed Neural Networks (PINNs)*

### About the Project
This repository contains the **v31** version of the Parametric Physics-Informed Neural Network (Parametric PINN) model. This stage of the research focused on solving the **Edwards Equation** to map the thermodynamic behavior of polymer chains around macro-ions (proteins/nanoparticles) that possess a **homogeneous surface charge**.

The great advantage of this model is its capability for parametric generalization: instead of training the network for a single solvent configuration, salinity, or delta, the PINN v31 learned the continuous solution of the entire **Phase Space**.

### The Phase Space and Adsorption Transition
Using the v31 network, we were able to calculate and extract the critical phase transition curves, mapping the **Adsorption Parameter** ($\delta_c$ or $\delta$) against the **Electrostatic Screening Parameter** ($\kappa \times a$).

This allowed us to accurately identify the thermodynamic boundaries where the polymer transitions from the _Coil_ state (unfolded/solvated) to the _Adsorption_ state on the surface of the macromolecule, depending on the ionic strength of the solvent and the surface affinity.

### Automatic Differentiation and Continuous Solution (Autograd)
The major physical advantage of using a PINN instead of a traditional numerical solver (such as Finite Differences) is that **Neural Networks are analytically differentiable**. 

Instead of discretizing the space into a mesh, we use PyTorch's `torch.autograd` to calculate the exact derivatives of the Edwards Equation with respect to the spatial coordinates. This means the model learns a **continuous and mesh-free** solution. Since the residual of the differential equation is embedded directly into the network's Loss Function, the model is strictly forced to respect the laws of thermodynamics at any point in the continuous space.

### Repository Structure
* `/scripts/`: 
  * `unified_pinn_edwards_v31.py`: Main source code of the Parametric PINN V31, containing the formulation of the Edwards Equation physical residual loss and the optimized training logic (L-BFGS).
  * `plot_phase_map_v31.py`: Numerical script to sweep the parametric domain of the trained network and extract the adsorption Phase Space matrix.
* `/images/`:
  * `phase_map_v31.png`: The thermodynamic transition map extracted by the neural network (Adsorption Parameter vs $\kappa \times a$).
  * Other structural polymer density predictions for specific configurations.

### Demonstration (Phase Map)
The phase map extracted by the parameterized PINN illustrating the polymer transition:

![Phase Map v31](images/phase_map_v31.png)

#### Density Profiles (Continuous Solution)
Thanks to the network's differentiability, the polymer density predictions around the sphere are perfectly continuous curves, without the typical spatial discretization artifacts. *(Density profile image to be added)*

---

## Português

> **Primeira Etapa do Projeto de Mestrado** 
> *Modelagem de Polímeros em Superfícies de Carga Homogênea usando Redes Neurais Informadas por Física (PINNs)*

### Sobre o Projeto
Este repositório contém a versão **v31** do modelo de Redes Neurais Informadas por Física Paramétrica (PINN Paramétrica). Esta etapa da pesquisa focou em resolver a **Equação de Edwards** para mapear o comportamento termodinâmico de cadeias poliméricas ao redor de macro-íons (proteínas/nanopartículas) que possuem uma **superfície de carga homogênea**.

O grande diferencial deste modelo é a sua capacidade de generalização paramétrica: em vez de treinar a rede para uma única configuração de solvente, salinidade ou delta, a PINN v31 aprendeu a solução contínua de todo o **Espaço de Fase**.

### O Espaço de Fase e a Transição de Adsorção
Usando a rede v31, nós conseguimos calcular e extrair as curvas críticas de transição de fase, mapeando o **Parâmetro de Adsorção** ($\delta_c$ ou $\delta$) contra o **Parâmetro de Blindagem Eletrostática** ($\kappa \times a$).

Isso nos permitiu identificar exatamente as fronteiras termodinâmicas onde o polímero transiciona do estado _Coil_ (desenovelado/solvatação) para o estado de _Adsorção_ na superfície da macromolécula, dependendo da força iônica do solvente e da afinidade da superfície.

### Diferenciabilidade Automática e Solução Contínua (Autograd)
A grande vantagem física de usar uma PINN em vez de um solucionador numérico tradicional (como Diferenças Finitas) é que as **Redes Neurais são analiticamente diferenciáveis**. 

Em vez de discretizar o espaço numa malha, usamos o `torch.autograd` do PyTorch para calcular as derivadas exatas da Equação de Edwards em relação às coordenadas espaciais. Isso significa que o modelo aprende uma solução **contínua e livre de malha (mesh-free)**. Como o resíduo da equação diferencial é embutido diretamente na Função de Custo (Loss) da rede, o modelo é forçado a respeitar estritamente as leis da termodinâmica em qualquer ponto do espaço contínuo.

### Estrutura do Repositório
* `/scripts/`: 
  * `unified_pinn_edwards_v31.py`: Código-fonte principal da PINN Paramétrica V31, contendo a formulação da loss de resíduo físico da Equação de Edwards e as lógicas de treinamento otimizado (L-BFGS).
  * `plot_phase_map_v31.py`: Script numérico para varrer o domínio paramétrico da rede treinada e extrair a matriz do Espaço de Fase de adsorção.
* `/images/`:
  * `phase_map_v31.png`: O mapa termodinâmico de transição extraído pela rede neural (Parâmetro de Adsorção vs $\kappa \times a$).
  * Outras predições estruturais de densidade polimérica para configurações pontuais.

### Demonstração (Mapa de Fase)
O mapa de fase extraído pela PINN parametrizada ilustrando a transição do polímero:

![Phase Map v31](images/phase_map_v31.png)

#### Perfis de Densidade (Solução Contínua)
Graças à diferenciabilidade da rede, as predições de densidade do polímero ao redor da esfera são curvas perfeitamente contínuas, sem os artefatos típicos de discretização espacial. *(Imagem do perfil a ser adicionada)*

---
*Statistical Polymer Physics informed by Machine Learning.*
