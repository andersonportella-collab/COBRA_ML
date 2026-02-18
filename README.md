# 🐍 COBRA Machine Learning (COBRA-ML)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Academic-green)]()

## 📋 Descrição

O **COBRA-ML** é um método inovador de tomada de decisão multicritério que integra:

- ✅ **COBRA (Comprehensive Distance Based Ranking)**: Método base de ranking multicritério
- ✅ **PSI (Preference Selection Index)**: Cálculo objetivo de pesos baseado em variação de dados
- ✅ **Balanceamento Gaussiano**: Ajuste parametrizado de pesos com função gaussiana
- ✅ **Machine Learning**: Predição automática de parâmetros ótimos via Random Forest
- ✅ **Interface Bilíngue**: Suporte completo para Português e Inglês
- ✅ **Análise com IA**: Integração opcional com OpenAI GPT-4 para insights avançados

Desenvolvido para aplicações em Pesquisa Operacional, Engenharia de Produção e Tomada de Decisão Multicritério.

---

## 👨‍💻 Autores

- **Anderson Portella** - Universidade Federal Fluminense (UFF)
- **Prof. Dr. Marcos dos Santos** - Escola Naval
- **Prof. Dr. Carlos Francisco Simões Gomes** - Universidade Federal Fluminense (UFF)

### Afiliações
- **Universidade Federal Fluminense (UFF)** - Programa de Pós-Graduação em Engenharia de Produção
- **Escola Naval** - Departamento de Engenharia Naval

---

## 🚀 Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Passo 1: Clonar o repositório

```bash
git clone https://github.com/seu-usuario/cobra-ml.git
cd cobra-ml
```

### Passo 2: Instalar dependências

```bash
pip install -r requirements.txt
```

### Passo 3: (Opcional) Configurar OpenAI para análise com IA

Crie um arquivo `code.env` na raiz do projeto:

```bash
OPENAI_API_KEY=sua_chave_api_aqui
```

Ou configure diretamente pela interface web após iniciar o aplicativo.

---

## 💻 Como Usar

### Executar a Aplicação Web

```bash
streamlit run cobra_ml_app_integrated_FIXED4.py
```

A aplicação será aberta automaticamente no navegador em `http://localhost:8501`

### Recursos Principais

#### 1. Seleção de Idioma 🌐
- Interface completa em **Português** ou **Inglês**
- Tradução de todos os elementos (menus, botões, mensagens, gráficos)

#### 2. Fonte de Dados 📊

**Opção A: Exemplos Pré-carregados**
- ⚡ Energia Renovável
- ✈️ Aviões de Combate  
- 🏭 Fornecedores

**Opção B: Upload de Arquivo**
- Formatos: Excel (.xlsx) ou CSV (.csv)
- Primeira coluna: nomes das alternativas
- Demais colunas: critérios de avaliação

**Opção C: Problema Customizado**
- Defina número de alternativas e critérios
- Configure tipos (Maximizar/Minimizar)
- Edite dados diretamente na interface

#### 3. Configurações de Machine Learning 🤖

- **Predição Automática**: ML prediz λ, μ, σ baseado no problema
- **Treinamento**: Gere dados sintéticos ou use seus próprios
- **Parâmetros Manuais**: Ajuste fino de λ, μ, σ se preferir controle total

#### 4. Análise e Visualizações 📈

- **Ranking Final**: Posições e scores de cada alternativa
- **Análise de Pesos**: Comparação PSI vs Finais
- **Matrizes**: Normalizada e Ponderada com formatação condicional
- **Gráficos Interativos**: Barras, heatmaps, radar charts
- **Função Gaussiana**: Visualização do balanceamento

#### 5. Análise com IA (Opcional) 🤖

Requer chave OpenAI configurada:

- **Geração Automática**: Insights sobre resultados
- **Chat Interativo**: Perguntas e respostas sobre a análise
- **Interpretação**: Explicação dos pesos e ranking

#### 6. Exportação 📥

- **CSV**: Resultados tabulares
- **Excel**: Múltiplas abas (Resultados, Matriz Normalizada, Matriz Ponderada)
- **JSON**: Estrutura completa com metadados

---

## 📐 Metodologia

### Fluxo do Método COBRA-ML

```
Entrada (Matriz X)
    ↓
1. Normalização Vetorial (R)
    ↓
2. Cálculo PSI (w^PSI)
    ↓
3. Predição ML (λ, μ, σ) ← [Novidade!]
    ↓
4. Balanceamento Gaussiano (w^FINAL)
    ↓
5. Matriz Ponderada (V)
    ↓
6. Distâncias COBRA (D+, D-)
    ↓
7. Scores e Ranking
```

### Principais Inovações

#### 1. Normalização Vetorial (vs COBRA Original)

O COBRA-ML substitui a normalização max-min do COBRA original por **normalização vetorial**:

**Para critérios de benefício:**
```
r_ij = x_ij / √(Σ x_kj²)
```

**Para critérios de custo:**
```
r_ij = (1/x_ij) / √(Σ (1/x_kj)²)
```

**Vantagens:**
- Maior estabilidade numérica
- Menor sensibilidade a outliers
- Preservação de proporções relativas
- Consistência com fundamentos de álgebra linear

#### 2. Predição Automática de Parâmetros

Machine Learning (Random Forest) prediz automaticamente:

- **λ (lambda)**: Balanceamento PSI vs Subjetivo [0, 1]
- **μ (mu)**: Centro da gaussiana [0, n-1]
- **σ (sigma)**: Dispersão da gaussiana [0.5, n]

**Features extraídas:**
- Dimensionalidade (m, n)
- Variância dos pesos PSI
- Entropia dos pesos
- Estatísticas dos dados (média, desvio)
- Proporção critérios benefício/custo

---

## 📊 Exemplo de Uso (Python)

```python
from cobra_ml_app_integrated_FIXED4 import COBRA_ML
import numpy as np

# 1. Criar instância
cobra = COBRA_ML()

# 2. Definir problema
alternatives = ['Solar PV', 'Eólica', 'Hidrelétrica']
criteria = ['LCOE ($/MWh)', 'Capacidade (%)', 'Emissões (g/kWh)']
criteria_types = ['cost', 'benefit', 'cost']

data = np.array([
    [45, 25, 50],
    [38, 35, 12],
    [55, 50, 24],
])

# 3. Executar análise
results = cobra.run_cobra_ml(
    X=data,
    criteria_types=criteria_types,
    use_ml=False  # ou True para usar ML
)

# 4. Ver resultados
print("Ranking:", results['ranking'])
print("Scores:", results['scores'])
print("Pesos Finais:", results['weights_final'])
```

**Saída esperada:**
```
Ranking: [1 2 0]  # Eólica (1º), Hidrelétrica (2º), Solar (3º)
Scores: [0.4123 0.7856 0.5234]
Pesos Finais: [0.3289 0.3892 0.2819]
```

---

## 🔧 Estrutura do Projeto

```
cobra-ml/
│
├── cobra_ml_app_integrated_FIXED4.py   # Aplicação principal
├── requirements.txt                    # Dependências Python
├── code.env                           # Configuração OpenAI (opcional)
├── README.md                          # Este arquivo
├── .gitignore                         # Arquivos ignorados pelo Git
│
├── UFF_EN_brasao.png                  # Logo institucional (opcional)
│
└── docs/                              # Documentação adicional
    ├── FORMULACAO_MATEMATICA.py       # Equações completas
    └── exemplo_numerico.py            # Tutorial passo-a-passo
```

---

## 📚 Aplicações

### 1. Energia Renovável
Seleção de fontes de energia baseada em:
- LCOE (Custo Nivelado)
- Fator de Capacidade
- Emissões de CO₂
- Área necessária
- Maturidade tecnológica

### 2. Equipamentos Militares
Avaliação de aeronaves considerando:
- Custo de aquisição
- Performance operacional
- Capacidade Stealth
- Manutenibilidade
- Alcance

### 3. Seleção de Fornecedores
Decisão baseada em:
- Preço
- Qualidade
- Prazo de entrega
- Sustentabilidade
- Confiabilidade

### 4. Outros Domínios
- Localização de instalações
- Avaliação de projetos
- Seleção de tecnologias
- Gestão de portfólio
- Priorização de investimentos

---

## 🎓 Citação Acadêmica

Se utilizar o COBRA-ML em sua pesquisa, por favor cite:

**Formato ABNT:**
```
PORTELLA, Anderson; SANTOS, Marcos dos; GOMES, Carlos Francisco Simões. 
COBRA-ML: Integração de Machine Learning com Métodos Multicritério para 
Tomada de Decisão. Universidade Federal Fluminense, 2024.
```

**Formato APA:**
```
Portella, A., Santos, M. dos, & Gomes, C. F. S. (2024). COBRA-ML: Integration 
of Machine Learning with Multi-Criteria Methods for Decision Making. 
Federal Fluminense University.
```

**BibTeX:**
```bibtex
@techreport{portella2024cobraml,
  author = {Portella, Anderson and Santos, Marcos dos and Gomes, Carlos Francisco Sim{\~o}es},
  title = {COBRA-ML: Integra{\c{c}}{\~a}o de Machine Learning com M{\'e}todos Multicrit{\'e}rio},
  institution = {Universidade Federal Fluminense},
  year = {2024}
}
```

---

## 📄 Licença

Este projeto é desenvolvido para fins **acadêmicos e educacionais** em Engenharia de Produção e Pesquisa Operacional.

### Permissões:
- ✅ Uso acadêmico e educacional
- ✅ Modificação e adaptação para pesquisa
- ✅ Distribuição com atribuição aos autores

### Restrições:
- ❌ Uso comercial sem autorização dos autores
- ❌ Remoção de créditos
- ❌ Distribuição sem referência à fonte

Para uso comercial ou colaborações, contate os autores.

---

## 🔗 Links Úteis

- **SBPO (Simpósio Brasileiro de Pesquisa Operacional)**: [https://www.sobrapo.org.br/](https://www.sobrapo.org.br/)
- **UFF - Programa de Pós-Graduação em Engenharia de Produção**: [http://www.producao.uff.br/](http://www.producao.uff.br/)
- **Escola Naval**: [https://www.marinha.mil.br/en/](https://www.marinha.mil.br/en/)

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

### Áreas de Interesse para Contribuição:
- Novos métodos de normalização
- Algoritmos ML alternativos
- Extensões para dados fuzzy/intervalares
- Visualizações adicionais
- Otimizações de performance
- Novos domínios de aplicação
- Traduções para outros idiomas

---

## 🐛 Reportar Problemas

Encontrou um bug? Por favor abra uma [Issue](https://github.com/seu-usuario/cobra-ml/issues) incluindo:

1. Descrição detalhada do problema
2. Passos para reproduzir
3. Dados de entrada (se possível)
4. Mensagem de erro completa
5. Versão do Python e bibliotecas

---

## 📞 Contato

### Autores
- **Anderson Portella**
  - Email: anderson.portella@example.com
  - LinkedIn: [linkedin.com/in/andersonportella](https://linkedin.com/in/andersonportella)

- **Prof. Dr. Marcos dos Santos**
  - Escola Naval
  - Email: marcos.santos@marinha.mil.br

- **Prof. Dr. Carlos Francisco Simões Gomes**
  - Universidade Federal Fluminense
  - Email: cfsg@id.uff.br

### Institucional
- **UFF - Programa de Pós-Graduação em Engenharia de Produção**
  - Website: [http://www.producao.uff.br/](http://www.producao.uff.br/)
  
- **Escola Naval**
  - Website: [https://www.marinha.mil.br/en/](https://www.marinha.mil.br/en/)

---

## 🙏 Agradecimentos

- Universidade Federal Fluminense (UFF) - Suporte institucional
- Escola Naval - Colaboração e validação
- Comunidade MCDM - Fundamentos teóricos
- Desenvolvedores das bibliotecas open source utilizadas

---

## 📈 Roadmap

### Versão 2.0 (Futuro)
- [ ] Suporte para critérios fuzzy
- [ ] Análise de robustez automática
- [ ] Comparação com múltiplos métodos MCDM
- [ ] Interface mobile responsiva
- [ ] Relatórios em PDF automáticos
- [ ] API REST para integração

### Versão 3.0 (Visão)
- [ ] Integração com bancos de dados SQL
- [ ] Análise de incerteza (intervalos)
- [ ] Suporte para decisões em grupo
- [ ] Dashboard executivo
- [ ] Otimização multi-objetivo

---

## 📝 Changelog

### Versão 1.0.0 (2024-02-XX)
- ✨ Lançamento inicial do COBRA-ML
- ✨ Interface bilíngue (PT/EN)
- ✨ Integração com OpenAI GPT-4
- ✨ Upload de arquivos Excel/CSV
- ✨ Exportação múltipla (CSV/Excel/JSON)
- ✨ Normalização vetorial
- ✨ Machine Learning com Random Forest
- ✨ Visualizações interativas com Plotly

---

## ⭐ Se este projeto foi útil, considere dar uma estrela!

[![Star on GitHub](https://img.shields.io/github/stars/seu-usuario/cobra-ml?style=social)](https://github.com/seu-usuario/cobra-ml)

---

**Desenvolvido com 🐍 Python e ❤️ para pesquisa em Tomada de Decisão Multicritério**

*Última atualização: Fevereiro 2024*
