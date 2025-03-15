import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import matplotlib.animation as animation
from matplotlib.patches import Circle

class SimuladorMecanismo:
    """Simula um mecanismo de ligação de Jansen."""
    
    def __init__(self, comprimentos_links=None):
        # Usa os comprimentos ótimos de Jansen como padrão se nenhum for fornecido
        if comprimentos_links is None:
            self.comprimentos_links = {
                'a': 38,    # Comprimento da manivela
                'b': 41.5,  # Conexão da manivela
                'c': 39.3,  # Lado do triângulo
                'd': 40.1,  # Triângulo ao pé
                'e': 55.8,  # Triângulo ao link superior
                'f': 39.4,  # Comprimento do link superior
                'g': 36.7,  # Link superior ao pivô
                'h': 65.7,  # Distância do pivô fixo
                'i': 49.0,  # Distância do segundo pivô
                'j': 50.0,  # Conexão ao segundo pivô
                'k': 61.9,  # Conexão ao pé
            }
        else:
            self.comprimentos_links = comprimentos_links
            
        # Pontos fixos (pivôs)
        self.ponto_fixo1 = np.array([0, 0])  # Origem (centro da manivela)
        self.ponto_fixo2 = np.array([self.comprimentos_links['h'], 0])  # Primeiro pivô fixo
        self.ponto_fixo3 = np.array([self.comprimentos_links['i'], 0])  # Segundo pivô fixo
        
        # Outros pontos serão calculados durante a simulação
        self.pontos = {}
        
    def resolver_mecanismo(self, angulo_manivela):
        """Calcula todas as posições das juntas para um determinado ângulo da manivela."""
        a = self.comprimentos_links['a']
        b = self.comprimentos_links['b']
        c = self.comprimentos_links['c']
        d = self.comprimentos_links['d']
        e = self.comprimentos_links['e']
        f = self.comprimentos_links['f']
        g = self.comprimentos_links['g']
        j = self.comprimentos_links['j']
        k = self.comprimentos_links['k']
        
        # Converte graus para radianos
        theta = np.radians(angulo_manivela)
        
        # Calcula a posição final da manivela (ponto 2)
        p2 = self.ponto_fixo1 + np.array([a * np.cos(theta), a * np.sin(theta)])
        
        # Resolve para o ponto 3 (onde os links b e c se encontram)
        # Isso requer resolver um problema de interseção de círculos
        # Por simplicidade, usaremos uma aproximação
        # Em um modelo completo, você resolveria a interseção exata
        
        # Cálculo aproximado para o ponto 3
        # Na realidade, isso seria calculado encontrando a interseção
        # de círculos com raios b e c centrados em p2 e p4
        angulo_bc = theta + np.pi/6  # Esta é uma simplificação
        p3 = p2 + np.array([b * np.cos(angulo_bc), b * np.sin(angulo_bc)])
        
        # Continuaríamos calculando todas as posições das juntas
        # Esta é uma versão simplificada focando na posição do pé
        
        # Para demonstração, posição aproximada do pé
        angulo_pe = theta - np.pi/4  # Ângulo simplificado
        pe = p3 + np.array([d * np.cos(angulo_pe), d * np.sin(angulo_pe)])
        
        # Armazena todos os pontos calculados
        self.pontos = {
            'p1': self.ponto_fixo1,  # Centro da manivela
            'p2': p2,                # Fim da manivela
            'p3': p3,                # Junção do link b-c
            'pe': pe                 # Posição do pé
        }
        
        return self.pontos
    
    def simular_rotacao_completa(self, num_passos=360):
        """Simula uma rotação completa da manivela e rastreia o caminho do pé."""
        angulos = np.linspace(0, 360, num_passos)
        caminho_pe = []
        
        for angulo in angulos:
            pontos = self.resolver_mecanismo(angulo)
            caminho_pe.append(pontos['pe'])
            
        return np.array(caminho_pe)
    
    def avaliar_caminho_pe(self):
        """
        Avalia a qualidade do caminho do pé.
        Retorna uma pontuação onde menor é melhor.
        """
        # Obtém o caminho do pé
        caminho_pe = self.simular_rotacao_completa()
        
        # Critérios para avaliar:
        # 1. Quão plana é a parte inferior do caminho?
        # 2. Quanto movimento vertical do corpo isso causaria?
        # 3. O caminho está livre de loops ou cruzamentos?
        
        # Por simplicidade, vamos focar na planicidade da parte inferior
        # Encontra pontos na metade inferior do caminho
        valores_y = caminho_pe[:, 1]
        mediana_y = np.median(valores_y)
        pontos_inferiores = caminho_pe[valores_y < mediana_y]
        
        if len(pontos_inferiores) == 0:
            return 1000  # Pontuação ruim se não houver pontos inferiores
        
        # Calcula a variância dos valores y na parte inferior
        # Menor variância significa caminho mais plano onde toca o chão
        pontuacao_planicidade = np.var(pontos_inferiores[:, 1]) * 10
        
        # Calcula o comprimento total do caminho (métrica de eficiência)
        comprimento_caminho = 0
        for i in range(len(caminho_pe) - 1):
            comprimento_caminho += np.linalg.norm(caminho_pe[i+1] - caminho_pe[i])
        
        # Combina métricas (menor é melhor)
        pontuacao = pontuacao_planicidade + comprimento_caminho * 0.01
        
        return pontuacao


class OtimizadorGenetico:
    """Otimiza as proporções do mecanismo usando um algoritmo genético."""
    
    def __init__(self, tamanho_populacao=50, taxa_mutacao=0.1):
        self.tamanho_populacao = tamanho_populacao
        self.taxa_mutacao = taxa_mutacao
        self.populacao = []
        self.melhor_individuo = None
        self.melhor_pontuacao = float('inf')
        
        # Limites de intervalo para cada link (min, max)
        self.intervalos_links = {
            'a': (20, 60),    # Comprimento da manivela
            'b': (30, 70),    # Conexão da manivela
            'c': (30, 70),    # Lado do triângulo
            'd': (30, 70),    # Triângulo ao pé
            'e': (40, 80),    # Triângulo ao link superior
            'f': (30, 70),    # Comprimento do link superior
            'g': (30, 70),    # Link superior ao pivô
            'h': (50, 90),    # Distância do pivô fixo
            'i': (40, 80),    # Distância do segundo pivô
            'j': (40, 80),    # Conexão ao segundo pivô
            'k': (40, 90),    # Conexão ao pé
        }
        
    def inicializar_populacao(self):
        """Cria população inicial aleatória."""
        self.populacao = []
        for _ in range(self.tamanho_populacao):
            individuo = {}
            for link, (val_min, val_max) in self.intervalos_links.items():
                individuo[link] = random.uniform(val_min, val_max)
            self.populacao.append(individuo)
    
    def avaliar_individuo(self, individuo):
        """Avalia a aptidão de um indivíduo (menor é melhor)."""
        simulador = SimuladorMecanismo(individuo)
        try:
            pontuacao = simulador.avaliar_caminho_pe()
            return pontuacao
        except:
            # Se a simulação falhar (ex: impossibilidade mecânica)
            return float('inf')
    
    def selecionar_pais(self, pontuacoes_aptidao):
        """Seleciona pais com probabilidade inversamente proporcional à aptidão."""
        # Converte pontuações para probabilidades de seleção (pontuação menor = probabilidade maior)
        pontuacao_max = max(pontuacoes_aptidao) + 1  # Adiciona 1 para evitar divisão por zero
        probs_selecao = [(pontuacao_max - pontuacao) / pontuacao_max for pontuacao in pontuacoes_aptidao]
        
        # Normaliza probabilidades
        total = sum(probs_selecao)
        if total == 0:  # Se todos os indivíduos tiverem a mesma pontuação
            probs_selecao = [1/len(pontuacoes_aptidao)] * len(pontuacoes_aptidao)
        else:
            probs_selecao = [p / total for p in probs_selecao]
        
        # Seleciona dois pais
        indices_pais = np.random.choice(
            range(len(self.populacao)), 
            size=2, 
            p=probs_selecao, 
            replace=False
        )
        return [self.populacao[i] for i in indices_pais]
    
    def cruzamento(self, pai1, pai2):
        """Cria filho combinando os genes dos pais."""
        filho = {}
        for link in self.intervalos_links.keys():
            # 50% de chance de herdar de cada pai
            if random.random() < 0.5:
                filho[link] = pai1[link]
            else:
                filho[link] = pai2[link]
        return filho
    
    def mutar(self, individuo):
        """Muta genes aleatoriamente com probabilidade taxa_mutacao."""
        mutado = individuo.copy()
        for link, (val_min, val_max) in self.intervalos_links.items():
            if random.random() < self.taxa_mutacao:
                # Perturba o valor dentro de um intervalo
                delta = (val_max - val_min) * 0.1  # Permite 10% de mudança
                mutado[link] += random.uniform(-delta, delta)
                # Garante que esteja dentro dos limites
                mutado[link] = max(val_min, min(val_max, mutado[link]))
        return mutado
    
    def evoluir_geracao(self):
        """Evolui uma geração da população."""
        # Avalia aptidão para todos os indivíduos
        pontuacoes_aptidao = [self.avaliar_individuo(ind) for ind in self.populacao]
        
        # Rastreia o melhor indivíduo
        idx_pontuacao_min = np.argmin(pontuacoes_aptidao)
        if pontuacoes_aptidao[idx_pontuacao_min] < self.melhor_pontuacao:
            self.melhor_pontuacao = pontuacoes_aptidao[idx_pontuacao_min]
            self.melhor_individuo = self.populacao[idx_pontuacao_min].copy()
        
        # Cria nova geração
        nova_populacao = []
        
        # Elitismo: mantém o melhor indivíduo
        nova_populacao.append(self.populacao[idx_pontuacao_min].copy())
        
        # Cria o resto da população
        while len(nova_populacao) < self.tamanho_populacao:
            # Seleciona pais
            pais = self.selecionar_pais(pontuacoes_aptidao)
            
            # Cria filho
            filho = self.cruzamento(pais[0], pais[1])
            
            # Muta
            filho = self.mutar(filho)
            
            # Adiciona à nova população
            nova_populacao.append(filho)
        
        self.populacao = nova_populacao
        
        return self.melhor_pontuacao
    
    def executar_evolucao(self, geracoes=50):
        """Executa o algoritmo genético por múltiplas gerações."""
        self.inicializar_populacao()
        
        melhores_pontuacoes = []
        
        for ger in range(geracoes):
            melhor_pontuacao = self.evoluir_geracao()
            melhores_pontuacoes.append(melhor_pontuacao)
            
            print(f"Geração {ger+1}/{geracoes}: Melhor Pontuação = {melhor_pontuacao:.4f}")
        
        print("\nEvolução completa!")
        print(f"Melhores proporções de mecanismo encontradas:")
        for link, valor in self.melhor_individuo.items():
            print(f"{link}: {valor:.2f}")
            
        return melhores_pontuacoes, self.melhor_individuo
    
    def visualizar_evolucao(self, pontuacoes):
        """Plota a evolução das melhores pontuações ao longo das gerações."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pontuacoes)+1), pontuacoes)
        plt.title('Evolução da Melhor Pontuação')
        plt.xlabel('Geração')
        plt.ylabel('Pontuação (menor é melhor)')
        plt.grid(True)
        plt.show()
    
    def visualizar_melhor_mecanismo(self):
        """Visualiza o mecanismo com as melhores proporções encontradas."""
        if self.melhor_individuo is None:
            print("Nenhum melhor indivíduo encontrado ainda. Execute a evolução primeiro.")
            return
        
        simulador = SimuladorMecanismo(self.melhor_individuo)
        
        # Cria figura e eixos
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_aspect('equal')
        ax.grid(True)
        
        # Plota o caminho do pé
        caminho_pe = simulador.simular_rotacao_completa()
        ax.plot(caminho_pe[:, 0], caminho_pe[:, 1], 'b-', alpha=0.5, label='Caminho do Pé')
        
        # Cria um gráfico de dispersão para as juntas que serão atualizadas
        dispersao, = ax.plot([], [], 'ro', markersize=8)
        
        # Cria segmentos de linha para os links
        linhas = [ax.plot([], [], 'k-', linewidth=2)[0] for _ in range(4)]
        
        # Pontos fixos
        ax.plot(simulador.ponto_fixo1[0], simulador.ponto_fixo1[1], 'ko', markersize=10)
        ax.plot(simulador.ponto_fixo2[0], simulador.ponto_fixo2[1], 'ko', markersize=10)
        ax.plot(simulador.ponto_fixo3[0], simulador.ponto_fixo3[1], 'ko', markersize=10)
        
        # Define limites
        ax.set_xlim(-100, 150)
        ax.set_ylim(-100, 100)
        
        # Define rótulos
        ax.set_title('Mecanismo de Jansen Otimizado')
        ax.set_xlabel('Posição X')
        ax.set_ylabel('Posição Y')
        ax.legend()
        
        # Função de atualização da animação
        def atualizar(frame):
            angulo = frame
            pontos = simulador.resolver_mecanismo(angulo)
            
            # Atualiza posições das juntas
            dados_x = [pontos['p1'][0], pontos['p2'][0], pontos['p3'][0], pontos['pe'][0]]
            dados_y = [pontos['p1'][1], pontos['p2'][1], pontos['p3'][1], pontos['pe'][1]]
            dispersao.set_data(dados_x, dados_y)
            
            # Atualiza linhas dos links
            linhas[0].set_data([pontos['p1'][0], pontos['p2'][0]], [pontos['p1'][1], pontos['p2'][1]])
            linhas[1].set_data([pontos['p2'][0], pontos['p3'][0]], [pontos['p2'][1], pontos['p3'][1]])
            linhas[2].set_data([pontos['p3'][0], pontos['pe'][0]], [pontos['p3'][1], pontos['pe'][1]])
            
            return [dispersao] + linhas
        
        # Cria animação
        ani = animation.FuncAnimation(fig, atualizar, frames=np.linspace(0, 360, 72),
                                     interval=50, blit=True)
        
        plt.tight_layout()
        plt.show()
        
        return ani


# Exemplo de uso
if __name__ == "__main__":
    # Compara o design original de Jansen com um novo otimizado
    print("Iniciando otimização genética para encontrar proporções ótimas do Strandbeest...")
    otimizador = OtimizadorGenetico(tamanho_populacao=100, taxa_mutacao=0.1)
    pontuacoes, melhor_individuo = otimizador.executar_evolucao(geracoes=20)
    
    # Visualiza o processo de evolução
    otimizador.visualizar_evolucao(pontuacoes)
    
    # Visualiza o melhor mecanismo
    otimizador.visualizar_melhor_mecanismo()
    
    # Compara com as proporções originais de Jansen
    jansen_original = SimuladorMecanismo()  # Usa proporções padrão de Jansen
    pontuacao_jansen = jansen_original.avaliar_caminho_pe()
    
    otimizado = SimuladorMecanismo(melhor_individuo)
    pontuacao_otimizada = otimizado.avaliar_caminho_pe()
    
    print("\nComparação:")
    print(f"Pontuação original de Jansen: {pontuacao_jansen:.4f}")
    print(f"Pontuação otimizada: {pontuacao_otimizada:.4f}")
    print(f"Melhoria: {(pontuacao_jansen - pontuacao_otimizada) / pontuacao_jansen * 100:.2f}%")
    
    # Visualiza ambos os caminhos do pé para comparação
    plt.figure(figsize=(12, 8))
    
    # Original de Jansen
    caminho_pe_original = jansen_original.simular_rotacao_completa()
    plt.plot(caminho_pe_original[:, 0], caminho_pe_original[:, 1], 
             'b-', linewidth=2, label="Design Original de Jansen")
    
    # Design otimizado
    caminho_pe_otimizado = otimizado.simular_rotacao_completa()
    plt.plot(caminho_pe_otimizado[:, 0], caminho_pe_otimizado[:, 1], 
             'r-', linewidth=2, label="Design Geneticamente Otimizado")
    
    plt.title('Comparação dos Caminhos do Pé')
    plt.xlabel('Posição X')
    plt.ylabel('Posição Y')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()
