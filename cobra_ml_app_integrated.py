# cobra_ml_app_integrated_FIXED.py
"""
COBRA Machine Learning — Método Multicritério com Machine Learning
Aplicativo Streamlit Bilíngue (Português/Inglês)

Integra COBRA (Comprehensive Distance Based Ranking) com:
- PSI (Preference Selection Index)
- Balanceamento Gaussiano
- Machine Learning para predição de parâmetros ótimos

Autores:
- Anderson Portella (UFF)
- Prof. Dr. Marcos dos Santos (Escola Naval)
- Prof. Dr. Carlos Francisco Simões Gomes (UFF)
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

import datetime
import os

# =============================================================================
# CITAÇÕES (ABNT / APA)
# =============================================================================

CITATIONS = {
    "ABNT": """
**Formato ABNT:**

**Software (Registro INPI nº BR512026001134-5):**
PORTELLA, Anderson Gonçalves (Titular); SANTOS, Marcos dos; GOMES, Carlos Francisco Simões (Autores).
**COBRA Machine Learning**. Registro de Programa de Computador nº BR512026001134-5.
Rio de Janeiro: Instituto Nacional da Propriedade Industrial (INPI), 20 fev. 2026.
Disponível em: https://cobramachinelearning.streamlit.app/. Acesso em: {date}.
""",
    "APA": """
**APA Format:**

**Software (INPI Registration No. BR512026001134-5):**
Portella, A. G. (Holder), Santos, M. dos, & Gomes, C. F. S. (Authors). (2026).
*COBRA Machine Learning* [Computer software]. Registration No. BR512026001134-5.
Instituto Nacional da Propriedade Industrial (INPI).
Retrieved {date}, from https://cobramachinelearning.streamlit.app/
"""
}

def get_citation(lang: str) -> str:
    """Retorna a citação no formato apropriado conforme o idioma"""
    citation_type = "ABNT" if lang == "Portuguese" else "APA"
    today = datetime.datetime.now().strftime("%d %b. %Y")
    return CITATIONS[citation_type].format(date=today)

# AI client — optional (OpenAI-compatible interface)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# optional .env
try:
    from dotenv import load_dotenv
    load_dotenv("code.env")
except Exception:
    pass

# =============================================================================
# DICIONÁRIO DE TRADUÇÕES
# =============================================================================
TRANSLATIONS = {
    "Portuguese": {
        "title": "🐍 COBRA Machine Learning — UFF/EN",
        "subtitle": "Comprehensive Distance Based Ranking (COBRA) integrado com Machine Learning, Balanceamento Gaussiano e PSI (Preference Selection Index)",
        "settings": "⚙️ Configurações",
        "data_source": "Fonte de Dados",
        "use_ml": "🤖 Usar Machine Learning para predição de parâmetros",
        "train_model": "🎓 Treinar modelo ML com dados sintéticos",
        "training_samples": "Número de amostras de treinamento",
        "developers": "👨‍💻 Desenvolvedores",
        "clear_results": "🧹 Limpar resultados",
        "upload_file": "📤 Upload de Arquivo",
        "upload_help": "Carregue um arquivo Excel ou CSV",
        "file_loaded": "✅ Arquivo carregado",
        "configure_criteria": "**Configure os tipos de critérios:**",
        "editable_data": "📊 Dados do Problema (Editáveis)",
        "alternatives": "**Alternativas:**",
        "decision_matrix": "**Matriz de Decisão:**",
        "custom_problem": "📝 Problema Customizado",
        "n_alternatives": "Número de Alternativas",
        "n_criteria": "Número de Critérios",
        "alternative_names": "**Nomes das Alternativas:**",
        "criteria_types": "**Critérios e Tipos:**",
        "criteria_info": "ℹ️ Informações dos Critérios",
        "verify_change": "**Verifique e altere se necessário:**",
        "maximize": "Maximizar",
        "minimize": "Minimizar",
        "manual_params": "🎛️ Parâmetros Manuais",
        "run_analysis": "🚀 Executar Análise COBRA-ML",
        "processing": "⚙️ Processando...",
        "analysis_complete": "✅ Análise concluída!",
        "results": "📈 Resultados",
        "final_ranking": "🏆 Ranking Final",
        "weights_analysis": "📊 Análise de Pesos",
        "matrices": "🔢 Matrizes",
        "ml_parameters": "⚙️ Parâmetros ML",
        "visualizations": "📉 Visualizações",
        "position": "Posição",
        "alternative": "Alternativa",
        "score": "Score COBRA-ML",
        "psi_weights": "**Pesos PSI (Objetivos)**",
        "final_weights": "**Pesos Finais (Balanceados)**",
        "normalized_matrix": "**Matriz Normalizada (R)**",
        "weighted_matrix": "**Matriz Ponderada (V)**",
        "model_parameters": "⚙️ Parâmetros do Modelo",
        "parameter": "Parâmetro",
        "value": "Valor",
        "description": "Descrição",
        "predicted_params": "🤖 Parâmetros preditos pelo modelo de Machine Learning",
        "manual_default_params": "⚙️ Parâmetros definidos manualmente ou valores padrão",
        "gaussian_viz": "**Visualização da Função Gaussiana de Balanceamento**",
        "additional_viz": "📉 Visualizações Adicionais",
        "heatmap": "**Heatmap da Matriz Normalizada**",
        "radar_chart": "**Perfil das Alternativas (Gráfico Radar)**",
        "select_alternative": "Selecione a alternativa para visualizar:",
        "about_method": "ℹ️ Sobre o Método COBRA-ML",
        "export_results": "📚 Exportar Resultados",
        "export_subtitle": "### Exportar dados para análise posterior",
        "download_csv": "📥 Download CSV",
        "download_excel": "📥 Download Excel",
        "download_json": "📥 Download JSON",
        "ai_analysis": "🤖 Análise com Inteligência Artificial",
        "ai_help": "Use IA para obter insights detalhados sobre os resultados da análise COBRA-ML",
        "ai_key_warning": "⚠️ Configure sua chave de API de IA para usar a análise com IA",
        "ai_generate": "🔍 Gerar Análise com IA",
        "ai_analyzing": "🤖 Analisando resultados...",
        "ai_generated": "✅ Análise gerada com sucesso!",
        "ai_error": "❌ Erro ao gerar análise",
        "ai_chat": "💬 Chat Interativo com IA",
        "ai_chat_placeholder": "Faça perguntas sobre os resultados...",
        "ai_thinking": "🤖 Pensando...",
        "language": "🌐 Idioma / Language",
        "generate_dataset": "🗄️ Gerar Dataset Sintético para Treinamento ML",
        "generate_dataset_help": "Cria um dataset de problemas MCDM sintéticos para treinar e validar o componente ML do COBRA-ML",
        "dataset_generated": "✅ Dataset gerado com sucesso!",
        "dataset_n_problems": "Número de problemas MCDM a gerar",
        "dataset_download": "📥 Download Dataset (CSV)",
        "dataset_preview": "Pré-visualização do Dataset",
        "dataset_info": "ℹ️ Como usar este dataset",
        "citation_title": "📚 Como citar"
    },
    "English": {
        "title": "🐍 COBRA Machine Learning — UFF/EN",
        "subtitle": "Comprehensive Distance Based Ranking (COBRA) integrated with Machine Learning, Gaussian Balancing and PSI (Preference Selection Index)",
        "settings": "⚙️ Settings",
        "data_source": "Data Source",
        "use_ml": "🤖 Use Machine Learning for parameter prediction",
        "train_model": "🎓 Train ML model with synthetic data",
        "training_samples": "Number of training samples",
        "developers": "👨‍💻 Developers",
        "clear_results": "🧹 Clear results",
        "upload_file": "📤 Upload File",
        "upload_help": "Upload an Excel or CSV file",
        "file_loaded": "✅ File loaded",
        "configure_criteria": "**Configure criteria types:**",
        "editable_data": "📊 Problem Data (Editable)",
        "alternatives": "**Alternatives:**",
        "decision_matrix": "**Decision Matrix:**",
        "custom_problem": "📝 Custom Problem",
        "n_alternatives": "Number of Alternatives",
        "n_criteria": "Number of Criteria",
        "alternative_names": "**Alternative Names:**",
        "criteria_types": "**Criteria and Types:**",
        "criteria_info": "ℹ️ Criteria Information",
        "verify_change": "**Verify and change if necessary:**",
        "maximize": "Maximize",
        "minimize": "Minimize",
        "manual_params": "🎛️ Manual Parameters",
        "run_analysis": "🚀 Run COBRA-ML Analysis",
        "processing": "⚙️ Processing...",
        "analysis_complete": "✅ Analysis complete!",
        "results": "📈 Results",
        "final_ranking": "🏆 Final Ranking",
        "weights_analysis": "📊 Weights Analysis",
        "matrices": "🔢 Matrices",
        "ml_parameters": "⚙️ ML Parameters",
        "visualizations": "📉 Visualizations",
        "position": "Position",
        "alternative": "Alternative",
        "score": "COBRA-ML Score",
        "psi_weights": "**PSI Weights (Objective)**",
        "final_weights": "**Final Weights (Balanced)**",
        "normalized_matrix": "**Normalized Matrix (R)**",
        "weighted_matrix": "**Weighted Matrix (V)**",
        "model_parameters": "⚙️ Model Parameters",
        "parameter": "Parameter",
        "value": "Value",
        "description": "Description",
        "predicted_params": "🤖 Parameters predicted by Machine Learning model",
        "manual_default_params": "⚙️ Manually defined parameters or default values",
        "gaussian_viz": "**Gaussian Balancing Function Visualization**",
        "additional_viz": "📉 Additional Visualizations",
        "heatmap": "**Normalized Matrix Heatmap**",
        "radar_chart": "**Alternatives Profile (Radar Chart)**",
        "select_alternative": "Select alternative to view:",
        "about_method": "ℹ️ About COBRA-ML Method",
        "export_results": "📚 Export Results",
        "export_subtitle": "### Export data for further analysis",
        "download_csv": "📥 Download CSV",
        "download_excel": "📥 Download Excel",
        "download_json": "📥 Download JSON",
        "ai_analysis": "🤖 AI Analysis",
        "ai_help": "Use AI to get detailed insights about COBRA-ML analysis results",
        "ai_key_warning": "⚠️ Configure your AI API key to use AI analysis",
        "ai_generate": "🔍 Generate AI Analysis",
        "ai_analyzing": "🤖 Analyzing results...",
        "ai_generated": "✅ Analysis generated successfully!",
        "ai_error": "❌ Error generating analysis",
        "ai_chat": "💬 Interactive AI Chat",
        "ai_chat_placeholder": "Ask questions about the results...",
        "ai_thinking": "🤖 Thinking...",
        "language": "🌐 Language / Idioma",
        "generate_dataset": "🗄️ Generate Synthetic Dataset for ML Training",
        "generate_dataset_help": "Creates a dataset of synthetic MCDM problems to train and validate COBRA-ML's ML component",
        "dataset_generated": "✅ Dataset generated successfully!",
        "dataset_n_problems": "Number of MCDM problems to generate",
        "dataset_download": "📥 Download Dataset (CSV)",
        "dataset_preview": "Dataset Preview",
        "dataset_info": "ℹ️ How to use this dataset",
        "citation_title": "📚 How to cite"
    }
}

def t(key, lang="Portuguese"):
    """Função de tradução"""
    return TRANSLATIONS.get(lang, TRANSLATIONS["Portuguese"]).get(key, key)

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="COBRA Machine Learning",
    layout="wide",
    page_icon="🐍"
)

# Inicializar idioma no session_state
if "language" not in st.session_state:
    st.session_state["language"] = "Portuguese"

LOGO_PATH = "UFF_EN_brasao.png"

INSTITUTION_LINE = (
    "Universidade Federal Fluminense – Programa de Pós-Graduação em Engenharia de Produção<br/>"
    "Escola Naval <br/>"
)

# =============================================================================
# OPENAI AND AI FUNCTIONS
# =============================================================================

def setup_openai_client():
    """Setup AI client with API key (OpenAI-compatible)"""
    if OpenAI is None:
        return None
    
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("ai_api_key", None)
    
    if not api_key:
        api_key = st.sidebar.text_input(
            "🔑 AI API Key",
            type="password",
            help="Insira sua chave de API (OpenAI, etc.) / Enter your AI API key (OpenAI, etc.)"
        )
        if api_key:
            st.session_state["ai_api_key"] = api_key
    
    if api_key:
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            st.sidebar.error(f"Erro ao configurar cliente AI: {e}")
            return None
    return None

def build_cobra_ai_prompt(lang, results_summary):
    """Build prompt for COBRA-ML analysis"""
    is_pt = (lang == "Portuguese")
    
    if is_pt:
        header = "Você é um especialista em Métodos Multicritério e Machine Learning.\n\n"
        header += "Analise os resultados do COBRA-ML e forneça insights:\n\n"
        instructions = "Forneça:\n1. Interpretação dos resultados\n2. Análise dos pesos\n"
        instructions += "3. Avaliação da alternativa vencedora\n4. Recomendações\n"
    else:
        header = "You are an MCDA and ML expert.\n\nAnalyze COBRA-ML results:\n\n"
        instructions = "Provide:\n1. Results interpretation\n2. Weights analysis\n"
        instructions += "3. Winning alternative evaluation\n4. Recommendations\n"
    
    prompt = header + json.dumps(results_summary, indent=2, default=str) + "\n\n" + instructions
    return prompt

def analyze_with_ai(client, lang, results_summary):
    """Generate AI analysis"""
    if client is None:
        return None
    
    prompt = build_cobra_ai_prompt(lang, results_summary)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an MCDA expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Erro: {str(e)}"

def chat_with_ai(client, lang, user_message):
    """Chat with AI"""
    if client is None:
        return {"error": "Not configured"}
    
    is_pt = (lang == "Portuguese")
    messages = [
        {"role": "system", "content": f"MCDA expert. Respond in {'Portuguese' if is_pt else 'English'}."}
    ]
    
    if st.session_state.get("ai_last_analysis"):
        messages.append({"role": "assistant", "content": st.session_state["ai_last_analysis"]})
    
    for msg in st.session_state.get("ai_chat_history", [])[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": user_message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=400
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# CLASSE COBRA_ML
# =============================================================================

class COBRA_ML:
    """
    Método COBRA-ML: Integração do COBRA com Machine Learning,
    balanceamento gaussiano e PSI (Preference Selection Index)
    """
    
    def __init__(self):
        self.weights_psi = None
        self.weights_final = None
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.ml_model      = None   # RF Regressor para λ, σ
        self.ml_classifier = None   # RF Classifier para μ
        self.scaler        = StandardScaler()

    def normalize_matrix(self, X, criteria_types):
        X = np.array(X, dtype=float)
        R = np.zeros_like(X, dtype=float)
        
        for j in range(X.shape[1]):
            col = X[:, j]
            
            if criteria_types[j] == 'benefit':
                denom = np.sqrt(np.sum(col**2))
                R[:, j] = col / denom if denom != 0 else col
            else:  # cost
                # Guard: valores zero em critério de custo geram 1/0 = inf e depois NaN,
                # corrompendo silenciosamente o ranking. Neutraliza o inf/NaN.
                # Para dados sem zeros este bloco é numericamente idêntico ao original.
                with np.errstate(divide='ignore', invalid='ignore'):
                    inv_col = 1.0 / col
                inv_col = np.nan_to_num(inv_col, nan=0.0, posinf=0.0, neginf=0.0)
                denom = np.sqrt(np.sum(inv_col**2))
                R[:, j] = inv_col / denom if denom != 0 else inv_col
        
        self.normalized_matrix = R
        return R


#    def normalize_matrix(self, X, criteria_types):
#        """Normaliza a matriz de decisão"""
#        X = np.array(X)
#        R = np.zeros_like(X, dtype=float)
        
#        for j in range(X.shape[1]):
#            if criteria_types[j] == 'benefit':
#                R[:, j] = X[:, j] / np.max(X[:, j])
#            else:  # cost
#                R[:, j] = np.min(X[:, j]) / X[:, j]
                
#        self.normalized_matrix = R
#        return R
    
    def calculate_psi_weights(self, R):
        """Calcula pesos usando Preference Selection Index (PSI)"""
        m, n = R.shape
        mean_values = np.mean(R, axis=0)
        PV = np.sum((R - mean_values)**2, axis=0)
        Phi = 1 - PV
        weights_psi = Phi / np.sum(Phi)
        self.weights_psi = weights_psi
        return weights_psi
    
    def gaussian_function(self, j, mu, sigma, n):
        """Função gaussiana para balanceamento"""
        return np.exp(-((j - mu)**2) / (2 * sigma**2))
    
    def calculate_gaussian_weights(self, n, mu, sigma, weights_psi, weights_sub, lambda_param):
        """Calcula pesos finais com balanceamento gaussiano"""
        alpha = np.array([self.gaussian_function(j, mu, sigma, n) for j in range(n)])
        alpha = alpha / np.sum(alpha)
        weights_bal = lambda_param * alpha * weights_psi + (1 - lambda_param) * alpha * weights_sub
        weights_final = weights_bal / np.sum(weights_bal)
        self.weights_final = weights_final
        return weights_final
    
    def calculate_cobra_scores(self, R, weights):
        """Calcula os scores COBRA"""
        V = R * weights
        self.weighted_matrix = V
        V_plus = np.max(V, axis=0)
        V_minus = np.min(V, axis=0)
        D_plus = np.sqrt(np.sum((V - V_plus)**2, axis=1))
        D_minus = np.sqrt(np.sum((V - V_minus)**2, axis=1))
        scores = D_minus / (D_plus + D_minus)
        return scores
    
    def calculate_entropy(self, weights):
        """Calcula entropia de Shannon dos pesos"""
        weights = weights + 1e-10
        return -np.sum(weights * np.log(weights))
    
    def extract_features(self, X, criteria_types):
        """Extrai features do problema para alimentar o modelo ML"""
        m, n = X.shape
        R = self.normalize_matrix(X, criteria_types)
        weights_psi = self.calculate_psi_weights(R)
        
        features = [
            m, n,
            np.var(weights_psi),
            np.mean(np.abs(np.diff(X, axis=0))),
            self.calculate_entropy(weights_psi),
            np.std(X.flatten()),
            np.mean(X),
            criteria_types.count('benefit') / n,
        ]
        return np.array(features)
    
    def train_ml_model(self, training_data):
        """Treina um único modelo ML (RF Regressor) para prever 
           simultaneamente λ, μ e σ baseados nos casos âncora contínuos.
        """
        if len(training_data) < 5:
            st.warning("⚠️ Poucos dados de treinamento. Usando parâmetros padrão.")
            return None

        X_train, y_target = [], []
        for problem in training_data:
            # Extrai as features estruturais da matriz
            features = self.extract_features(problem['X'], problem['criteria_types'])
            X_train.append(features)
            
            # Agora os 3 parâmetros [lambda, mu, sigma] vão para um alvo único de regressão
            params = problem['optimal_params']   
            y_target.append(params)

        X_train  = np.array(X_train)
        y_target = np.array(y_target)
        
        # Garante que o scaler está instanciado antes da transformação
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
            
        X_scaled = self.scaler.fit_transform(X_train)

        # Regressor único Múltiplas Saídas: prevê λ, μ e σ de uma só vez
        self.ml_model = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.ml_model.fit(X_scaled, y_target)

        # Como não usamos mais Classificador para o μ, definimos como None
        # para evitar erros caso outra função antiga o procure.
        self.ml_classifier = None

        return self.ml_model

    def predict_parameters(self, X, criteria_types):
        """Prediz λ, μ e σ via regressão múltipla baseada nos estudos de caso reais."""
        if self.ml_model is None:
            n = X.shape[1]
            return {'lambda': 0.5, 'mu': 0.5, 'sigma': n / 4}

        features = self.extract_features(X, criteria_types).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        # O regressor agora prevê os 3 parâmetros de uma só vez: [lambda, mu, sigma]
        predictions = self.ml_model.predict(features_scaled)[0]
        
        lambda_param = float(np.clip(predictions[0], 0.1, 0.99))
        mu           = float(predictions[1]) 
        # Força o Sigma a nunca ultrapassar o número de critérios (n), garantindo a curva de sino
        sigma        = float(np.clip(predictions[2], 0.5, X.shape[1]))

        return {'lambda': lambda_param, 'mu': mu, 'sigma': sigma}

    def run_cobra_ml(self, X, criteria_types, weights_sub=None, use_ml=True, ml_params=None):
        """Executa o método COBRA-ML completo"""
        X = np.array(X)
        m, n = X.shape
        
        if weights_sub is None:
            weights_sub = np.ones(n) / n
        
        R = self.normalize_matrix(X, criteria_types)
        weights_psi = self.calculate_psi_weights(R)
        
        if use_ml:
            params = self.predict_parameters(X, criteria_types)
        else:
            params = ml_params if ml_params else {
                'lambda': 0.5, 'mu': n / 2, 'sigma': n / 4
            }
        
        weights_final = self.calculate_gaussian_weights(
            n, params['mu'], params['sigma'], 
            weights_psi, weights_sub, params['lambda']
        )
        
        scores = self.calculate_cobra_scores(R, weights_final)
        ranking = np.argsort(-scores)
        
        return {
            'scores': scores,
            'ranking': ranking,
            'weights_psi': weights_psi,
            'weights_final': weights_final,
            'params': params,
            'normalized_matrix': R,
            'weighted_matrix': self.weighted_matrix
        }

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def generate_training_data(n_samples=50):
    """Gera dados de treinamento ancorados nos estudos de caso reais e dados sintéticos."""
    training_data = []
    
    # 1. Âncoras Reais (Estudos de Caso Otimizados dos CSVs)
    # Sigma redimensionado para o limite do número de critérios (n) para preservar a curva Gaussiana.
    anchor_cases = [
        {'m': 7,  'n': 6, 'lambda': 0.720024, 'mu': 0.805558, 'sigma': 6.0},
        {'m': 4,  'n': 4, 'lambda': 0.450307, 'mu': 0.256207, 'sigma': 4.0},
        {'m': 4,  'n': 7, 'lambda': 0.934701, 'mu': 0.209900, 'sigma': 7.0},
        {'m': 12, 'n': 8, 'lambda': 0.954417, 'mu': 0.283609, 'sigma': 8.0}
    ]
    
    for case in anchor_cases:
        X = np.random.uniform(10, 100, size=(case['m'], case['n']))
        criteria_types = ['benefit' if np.random.rand() > 0.3 else 'cost' for _ in range(case['n'])]
        
        training_data.append({
            'X': X,
            'criteria_types': criteria_types,
            'optimal_params': [case['lambda'], case['mu'], case['sigma']]
        })
    
    # 2. Preenchimento Sintético
    for i in range(n_samples - len(anchor_cases)):
        m = np.random.randint(3, 15)
        n = np.random.randint(3, 10)
        X = np.random.uniform(10, 100, size=(m, n))
        criteria_types = ['benefit' if np.random.rand() > 0.3 else 'cost' for _ in range(n)]
        
        # Parâmetros sintéticos baseados na distribuição dos casos reais
        lambda_opt = np.random.uniform(0.45, 0.96)
        mu_opt = np.random.uniform(0.20, 0.81)
        # Sigma variando de forma proporcional aos critérios (n/4 até n)
        sigma_opt = np.random.uniform(n / 4, n)
        
        training_data.append({
            'X': X,
            'criteria_types': criteria_types,
            'optimal_params': [lambda_opt, mu_opt, sigma_opt]
        })
    
    return training_data

def create_sample_data(problem_type='energy', lang='Portuguese'):
    """Cria dados de exemplo para diferentes tipos de problemas, respeitando o idioma."""

    is_en = (lang == 'English')

    if problem_type == 'energy':
        alternatives = (
            ['Solar PV', 'Onshore Wind', 'Offshore Wind', 'Hydroelectric', 'Biomass']
            if is_en else
            ['Solar PV', 'Eólica Onshore', 'Eólica Offshore', 'Hidrelétrica', 'Biomassa']
        )
        criteria = (
            ['LCOE ($/MWh)', 'Capacity Factor (%)', 'CO2 Emissions (g/kWh)',
             'Required Area (km²/MW)', 'Tech Maturity (1-10)']
            if is_en else
            ['LCOE ($/MWh)', 'Fator Capacidade (%)', 'Emissões CO2 (g/kWh)',
             'Área Necessária (km²/MW)', 'Maturidade Tecnológica (1-10)']
        )
        criteria_types = ['cost', 'benefit', 'cost', 'cost', 'benefit']
        data = [
            [45, 25, 50, 4.0, 9],
            [38, 35, 12, 0.5, 9],
            [65, 45, 10, 0.3, 7],
            [55, 50, 24, 15.0, 10],
            [60, 70, 230, 2.0, 8],
        ]

    elif problem_type == 'aircraft':
        alternatives = ['F-35A', 'F-15EX', 'Rafale', 'Eurofighter', 'Su-57']
        criteria = (
            ['Unit Cost (M$)', 'Max Speed (Mach)', 'Range (km)',
             'Payload (kg)', 'Stealth (1-10)', 'Maintainability (1-10)']
            if is_en else
            ['Custo Unitário (M$)', 'Velocidade Máx (Mach)', 'Alcance (km)',
             'Carga Útil (kg)', 'Stealth (1-10)', 'Manutenibilidade (1-10)']
        )
        criteria_types = ['cost', 'benefit', 'benefit', 'benefit', 'benefit', 'benefit']
        data = [
            [80, 1.6, 2200, 8000, 10, 6],
            [90, 2.5, 3900, 13000, 3, 8],
            [85, 1.8, 3700, 9500, 5, 7],
            [95, 2.0, 2900, 7500, 4, 6],
            [100, 2.0, 3500, 10000, 8, 4],
        ]

    else:  # suppliers
        alternatives = (
            ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D']
            if is_en else
            ['Fornecedor A', 'Fornecedor B', 'Fornecedor C', 'Fornecedor D']
        )
        criteria = (
            ['Price', 'Quality (1-10)', 'Delivery Time (days)',
             'Sustainability (1-10)', 'Reliability (1-10)']
            if is_en else
            ['Preço', 'Qualidade (1-10)', 'Prazo Entrega (dias)',
             'Sustentabilidade (1-10)', 'Confiabilidade (1-10)']
        )
        criteria_types = ['cost', 'benefit', 'cost', 'benefit', 'benefit']
        data = [
            [100, 8, 15, 7, 9],
            [120, 9, 10, 9, 8],
            [90, 7, 20, 6, 7],
            [110, 8.5, 12, 8, 8.5],
        ]

    return alternatives, criteria, criteria_types, data

def read_uploaded_file(uploaded_file):
    """Lê arquivo Excel ou CSV uploadado"""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)


def generate_synthetic_dataset(n_problems=100, seed=42):
    """
    Gera dataset sintético de problemas MCDM para treinamento e validação
    do componente de Machine Learning do COBRA-ML.

    Cada linha representa um problema MCDM com suas features estruturais (f1-f8)
    e os parâmetros ótimos correspondentes (lambda, mu, sigma).

    Os parâmetros ótimos são definidos por regras heurísticas baseadas nas
    propriedades estatísticas do problema, seguindo a lógica descrita no artigo.
    """
    np.random.seed(seed)
    cobra = COBRA_ML()
    records = []

    for k in range(n_problems):
        m = np.random.randint(3, 12)          # alternativas: 3–11
        n = np.random.randint(3, 10)           # critérios:    3–9

        # Distribuição variada: uniforme, normal, log-normal, mista
        dist = np.random.choice(['uniform', 'normal', 'lognormal', 'mixed'])
        if dist == 'uniform':
            X = np.random.uniform(1, 200, size=(m, n))
        elif dist == 'normal':
            X = np.abs(np.random.normal(50, 20, size=(m, n))) + 1
        elif dist == 'lognormal':
            X = np.random.lognormal(3, 0.8, size=(m, n))
        else:
            X = np.column_stack([
                np.random.uniform(1, 100, size=m) if np.random.rand() > 0.5
                else np.abs(np.random.normal(50, 15, size=m)) + 1
                for _ in range(n)
            ])

        # Tipos de critérios aleatórios
        benefit_ratio = np.random.uniform(0.3, 0.8)
        criteria_types = [
            'benefit' if np.random.rand() < benefit_ratio else 'cost'
            for _ in range(n)
        ]
        actual_benefit_ratio = criteria_types.count('benefit') / n

        # Extrair features estruturais
        try:
            R = cobra.normalize_matrix(X, criteria_types)
            weights_psi = cobra.calculate_psi_weights(R)
        except Exception:
            continue

        psi_var   = float(np.var(weights_psi))
        psi_entropy = float(cobra.calculate_entropy(weights_psi))
        data_std  = float(np.std(X))
        data_mean = float(np.mean(X))
        mean_adj  = float(np.mean(np.abs(np.diff(X, axis=0))))

        # ---------------------------------------------------------------
        # Parâmetros ótimos (regras heurísticas validadas na literatura)
        # ---------------------------------------------------------------
        # λ: combina entropia PSI normalizada, concentração de pesos (variância)
        #    e proporção de critérios de benefício.
        #    - Alta entropia (pesos uniformes) → maior objetividade → λ alto
        #    - Alta variância PSI (um critério domina) → mais correção subjetiva → λ baixo
        #    - Mais critérios de benefício → problema mais orientado a ganho → λ moderado-alto
        #    Fórmula cobre [0.1, 0.9] com distribuição rica entre os extremos.
        entropy_norm  = psi_entropy / np.log(n + 1e-10)   # normalizado [0,1]
        var_norm      = np.clip(psi_var * n, 0, 1)          # variância relativa
        # λ: função determinística de benefit_ratio, entropy_norm e var_norm
        # sem ruído — o sinal precisa ser aprendível pelo RF
        lambda_opt = float(np.clip(
            0.60 * actual_benefit_ratio
            + 0.25 * entropy_norm
            - 0.20 * var_norm
            + 0.10,
            0.10, 0.90
        ))

        # μ: índice inteiro do critério com maior peso PSI (decisão determinística)
        best_criterion_idx = int(np.argmax(weights_psi))
        mu_opt = best_criterion_idx

        # σ: dispersão proporcional ao número de critérios; maior quando pesos PSI
        #    são mais uniformes (todos critérios têm importância similar)
        sigma_opt = float(np.clip(
            (n / 4) * (1 + entropy_norm) + np.random.normal(0, 0.2),
            0.5, n
        ))

        records.append({
            'problem_id':          k,
            'm':                   m,
            'n':                   n,
            'benefit_ratio':       round(actual_benefit_ratio, 4),
            'psi_weight_variance': round(psi_var, 6),
            'psi_weight_entropy':  round(psi_entropy, 6),
            'data_std':            round(data_std, 4),
            'data_mean':           round(data_mean, 4),
            'mean_adj_variation':  round(mean_adj, 4),
            'lambda_opt':          round(lambda_opt, 4),
            'mu_opt':              round(mu_opt, 4),
            'sigma_opt':           round(sigma_opt, 4),
        })

    return pd.DataFrame(records)


# =============================================================================
# INTERFACE STREAMLIT PRINCIPAL
# =============================================================================

def main():
    # Obter idioma atual
    lang = st.session_state.get("language", "Portuguese")
    
    # MENU SUPERIOR PARA SELEÇÃO DE IDIOMA
    col_lang, col_space = st.columns([1, 5])
    with col_lang:
        new_lang = st.selectbox(
            t("language", lang),
            ["Portuguese", "English"],
            index=0 if lang == "Portuguese" else 1,
            key="language_selector"
        )
        if new_lang != lang:
            st.session_state["language"] = new_lang
            st.rerun()
    
    # Cabeçalho com logo
    col_logo, col_title = st.columns([1, 4])
    
    with col_logo:
        try:
            st.image(LOGO_PATH, width=150)
        except:
            st.write("🐍")
    
    with col_title:
        st.title(t("title", lang))
        st.markdown(INSTITUTION_LINE, unsafe_allow_html=True)
    
    st.markdown(f"""
    **{t("subtitle", lang)}**
    ---
    """)
    
    # =============================================================================
    # SIDEBAR
    # =============================================================================
    st.sidebar.header(t("settings", lang))
    
    # Seleção do tipo de entrada de dados
    data_source = st.sidebar.selectbox(
        t("data_source", lang),
        ['energy', 'aircraft', 'suppliers', 'custom', 'upload'],
        format_func=lambda x: {
            'energy': '⚡ Energia Renovável' if lang == "Portuguese" else '⚡ Renewable Energy',
            'aircraft': '✈️ Aviões de Combate' if lang == "Portuguese" else '✈️ Fighter Aircraft',
            'suppliers': '🏭 Fornecedores' if lang == "Portuguese" else '🏭 Suppliers',
            'custom': '📝 Problema Customizado' if lang == "Portuguese" else '📝 Custom Problem',
            'upload': '📤 Upload de Arquivo' if lang == "Portuguese" else '📤 Upload File'
        }[x]
    )
    
    st.sidebar.markdown("---")

    # Opções de ML — simplificado: uma única seleção de modo
    _ml_label  = "⚙️ Modo de Parâmetros" if lang == "Portuguese" else "⚙️ Parameter Mode"
    _ml_opts   = (
        ["🤖 ML automático", "🎛️ Manual"],
        ["🤖 Automatic ML", "🎛️ Manual"]
    )[0 if lang == "Portuguese" else 1]
    _ml_help = (
        "ML automático: Random Forest prevê λ, μ, σ a partir das features do problema.\n"
        "Manual: defina λ, μ, σ diretamente pelos sliders abaixo."
        if lang == "Portuguese" else
        "Automatic ML: Random Forest predicts λ, μ, σ from problem features.\n"
        "Manual: set λ, μ, σ directly with sliders below."
    )
    ml_mode = st.sidebar.radio(_ml_label, _ml_opts, help=_ml_help)
    use_ml = (_ml_opts.index(ml_mode) == 0)   # True se ML automático

    if use_ml:
        # O texto da âncora aparece aqui, visível assim que o ML automático é selecionado
        if lang == "Portuguese":
            st.sidebar.caption("🧠 **Âncora de Casos Reais:** O motor de predição está calibrado com os parâmetros otimizados (Tau de Kendall até 1.0) das matrizes da Arábia Saudita, Shenzhen, Sérvia e Egito.")
        else:
            st.sidebar.caption("🧠 **Real-Case Anchor:** The prediction engine is calibrated using the optimized parameters (Kendall's Tau up to 1.0) from the Saudi Arabia, Shenzhen, Serbia, and Egypt case studies.")

        if "trained_ml_model" in st.session_state:
            _model_status = (
                "✅ **Modelo treinado e ativo.** Os parâmetros serão preditos automaticamente."
                if lang == "Portuguese" else
                "✅ **Model trained and active.** Parameters will be predicted automatically."
            )
            st.sidebar.success(_model_status)
        else:
            _tip = (
                "⚠️ Nenhum modelo treinado ainda. Use a seção **Treinar Modelo ML** "
                "nos resultados da análise para treinar o modelo dentro do app."
                if lang == "Portuguese" else
                "⚠️ No trained model yet. Use the **Train ML Model** section "
                "in the analysis results to train the model inside the app."
            )
            st.sidebar.warning(_tip)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t('developers', lang)}")
    st.sidebar.markdown("""
    - [Anderson Portella](https://www.linkedin.com/in/andersonportella/) - UFF
    - [Prof. Dr. Marcos dos Santos](https://www.linkedin.com/in/profmarcosdossantos/) - Escola Naval
    - [Prof. Dr. Carlos Francisco Simões Gomes](https://www.linkedin.com/in/carlos-francisco-sim%2525C3%2525B5es-gomes-7284a3b/) - UFF
    """)
    
    st.sidebar.markdown("---")
    with st.sidebar.expander(t("citation_title", lang)):
        st.markdown(get_citation(lang))
    st.sidebar.markdown("---")

    # Manual de uso - download
    _manual_label = "📖 Download User Manual" if lang == "English" else "📖 Baixar Manual do Usuário"
    _manual_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "COBRA_ML_User_Manual.pdf")
    if os.path.exists(_manual_path):
        with open(_manual_path, "rb") as _f:
            st.sidebar.download_button(
                label=_manual_label,
                data=_f.read(),
                file_name="COBRA_ML_User_Manual.pdf",
                mime="application/pdf"
            )
    else:
        st.sidebar.caption(
            "ℹ️ Manual not found. Place COBRA_ML_User_Manual.pdf in the app folder." if lang == "English"
            else "ℹ️ Manual não encontrado. Coloque COBRA_ML_User_Manual.pdf na pasta do app."
        )
    st.sidebar.markdown("---")

    if st.sidebar.button(t("clear_results", lang)):
        keys_to_delete = [k for k in st.session_state.keys()
                          if k.startswith(('alt_', 'crit_', 'data_', 'type_',
                                           'info_type_', 'ctype_', 'cobra_',
                                           'ai_last', 'ai_chat', 'ai_enabled'))]
        for key in keys_to_delete:
            del st.session_state[key]
        st.session_state["_cleared"] = True
        st.rerun()

    if st.session_state.pop("_cleared", False):
        st.sidebar.success("✅ " + ("Resultados limpos!" if lang == "Portuguese" else "Results cleared!"))
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**For support/contact:** andersonportella@yahoo.com.br")
    
    # Fix 3 & 6: Also add "generate_dataset" translation keys
    for _lng in ["Portuguese", "English"]:
        if "generate_dataset" not in TRANSLATIONS[_lng]:
            TRANSLATIONS[_lng]["generate_dataset"] = (
                "🗄️ Gerar Dataset Sintético para Treinamento ML" if _lng == "Portuguese"
                else "🗄️ Generate Synthetic Dataset for ML Training"
            )
            TRANSLATIONS[_lng]["generate_dataset_help"] = (
                "Cria um dataset de problemas MCDM sintéticos para treinar e validar o componente ML do COBRA-ML"
                if _lng == "Portuguese"
                else "Creates a dataset of synthetic MCDM problems to train and validate COBRA-ML's ML component"
            )
            TRANSLATIONS[_lng]["dataset_generated"] = (
                "✅ Dataset gerado com sucesso!" if _lng == "Portuguese" else "✅ Dataset generated successfully!"
            )
            TRANSLATIONS[_lng]["dataset_n_problems"] = (
                "Número de problemas MCDM a gerar" if _lng == "Portuguese"
                else "Number of MCDM problems to generate"
            )
            TRANSLATIONS[_lng]["dataset_download"] = (
                "📥 Download Dataset (CSV)" if _lng == "Portuguese" else "📥 Download Dataset (CSV)"
            )
            TRANSLATIONS[_lng]["dataset_preview"] = (
                "Pré-visualização do Dataset" if _lng == "Portuguese" else "Dataset Preview"
            )
            TRANSLATIONS[_lng]["dataset_info"] = (
                "ℹ️ Como usar este dataset" if _lng == "Portuguese" else "ℹ️ How to use this dataset"
            )
    cobra_ml = COBRA_ML()
    # Reutilizar modelo treinado da sessão, se disponível
    if "trained_ml_model" in st.session_state:
        cobra_ml.ml_model      = st.session_state["trained_ml_model"]
        cobra_ml.ml_classifier = st.session_state.get("trained_ml_classifier")
        cobra_ml.scaler        = st.session_state["trained_ml_scaler"]
    
    # =============================================================================
    # ENTRADA DE DADOS
    # =============================================================================
    
    alternatives = []
    criteria = []
    criteria_types = []
    data = []
    
    if data_source == 'upload':
        st.subheader(t("upload_file", lang))

        # ── Exemplo para download ──────────────────────────────────────────
        with st.expander(
            "📎 Baixar arquivo de exemplo (CSV)" if lang == "Portuguese"
            else "📎 Download example file (CSV)"
        ):
            _ex_info = (
                "Use este arquivo como modelo para formatar seus dados corretamente antes do upload."
                if lang == "Portuguese" else
                "Use this file as a template to format your data correctly before uploading."
            )
            st.caption(_ex_info)

            # Gera CSV de exemplo (problema de energia no idioma atual)
            _ex_alts, _ex_crit, _ex_types, _ex_data = create_sample_data('energy', lang=lang)
            _ex_df = pd.DataFrame(_ex_data, columns=_ex_crit)
            _ex_df.insert(0, "Alternatives" if lang == "English" else "Alternativas", _ex_alts)

            st.dataframe(_ex_df, use_container_width=True)

            _note = (
                "ℹ️ A primeira linha do seu arquivo deve conter os cabeçalhos. "
                "A primeira coluna deve ter os nomes das alternativas. "
                "Os demais valores devem ser numéricos."
                if lang == "Portuguese" else
                "ℹ️ The first row of your file must contain headers. "
                "The first column must hold alternative names. "
                "All other values must be numeric."
            )
            st.caption(_note)

            _csv_example = _ex_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV" if lang == "English" else "📥 Baixar CSV",
                data=_csv_example,
                file_name="cobra_ml_example_template.csv",
                mime="text/csv",
                key="download_example_template"
            )

            # Excel de exemplo
            _buf = io.BytesIO()
            with pd.ExcelWriter(_buf, engine='xlsxwriter') as _w:
                _ex_df.to_excel(_w, sheet_name='Data', index=False)
                _info_df = pd.DataFrame({
                    'Column': _ex_df.columns.tolist(),
                    'Type': ['identifier'] + _ex_types
                })
                _info_df.to_excel(_w, sheet_name='Criteria Types', index=False)
            st.download_button(
                label="📥 Download Excel (.xlsx)" if lang == "English" else "📥 Baixar Excel (.xlsx)",
                data=_buf.getvalue(),
                file_name="cobra_ml_example_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_example_template_xlsx"
            )

        # ── Upload do arquivo do usuário ───────────────────────────────────
        uploaded_file = st.file_uploader(
            t("upload_help", lang),
            type=['xlsx', 'xls', 'csv'],
            help=t("upload_help", lang)
        )

        if uploaded_file is not None:
            df = read_uploaded_file(uploaded_file)
            st.success(f"{t('file_loaded', lang)}: {df.shape[0]} × {df.shape[1]}")
            st.dataframe(df.head())

            # Configurar alternativas e critérios
            alternatives = df.iloc[:, 0].tolist()
            criteria = df.columns[1:].tolist()
            data = df.iloc[:, 1:].values.tolist()

            # Usuário define tipos de critérios
            st.markdown(t("configure_criteria", lang))
            criteria_types = []
            cols = st.columns(min(len(criteria), 4))
            for idx, crit in enumerate(criteria):
                with cols[idx % len(cols)]:
                    ctype = st.selectbox(
                        f"{crit}",
                        ['benefit', 'cost'],
                        key=f"ctype_upload_{idx}"
                    )
                    criteria_types.append(ctype)
    
    elif data_source != 'custom':
        # Exemplos pré-definidos — somente leitura (Fix #1)
        alternatives, criteria, criteria_types, data = create_sample_data(data_source, lang=lang)

        st.subheader(
            "📊 Dados do Exemplo (Somente Leitura)" if lang == "Portuguese"
            else "📊 Example Data (Read-only)"
        )
        st.info(
            "ℹ️ Os exemplos são para demonstração e não podem ser editados. "
            "Para usar seus próprios dados, selecione **Problema Customizado** ou **Upload de Arquivo**."
            if lang == "Portuguese" else
            "ℹ️ Examples are for demonstration only and cannot be edited. "
            "To use your own data, select **Custom Problem** or **Upload File**."
        )

        st.markdown(f"**{'Alternativas' if lang == 'Portuguese' else 'Alternatives'}:** "
                    + " | ".join(alternatives))

        df_example = pd.DataFrame(data, columns=criteria, index=alternatives)
        st.dataframe(
            df_example.style
                .highlight_max(axis=0, color='lightgreen')
                .highlight_min(axis=0, color='lightcoral'),
            use_container_width=True
        )

        # Mostrar tipos dos critérios como badges informativos
        type_labels = []
        for crit, ctype in zip(criteria, criteria_types):
            icon = "⬆️" if ctype == "benefit" else "⬇️"
            label = "MAX" if ctype == "benefit" else "MIN"
            type_labels.append(f"{icon} **{crit}** ({label})")
        st.markdown("  ".join(type_labels))
        
    else:  # custom
        st.subheader(t("custom_problem", lang))
        
        col1, col2 = st.columns(2)
        with col1:
            n_alternatives = st.number_input(t("n_alternatives", lang), min_value=2, max_value=20, value=4)
        with col2:
            n_criteria = st.number_input(t("n_criteria", lang), min_value=2, max_value=15, value=5)
        
        # Entrada de nomes
        st.markdown(t("alternative_names", lang))
        alternatives = []
        cols = st.columns(min(n_alternatives, 4))
        for i in range(n_alternatives):
            with cols[i % len(cols)]:
                alt = st.text_input(f"Alt. {i+1}", value=f"Alt_{i+1}", key=f"alt_custom_{i}")
                alternatives.append(alt)
        
        # Entrada de critérios e tipos
        st.markdown(t("criteria_types", lang))
        criteria = []
        criteria_types = []
        for i in range(n_criteria):
            col1, col2 = st.columns([3, 1])
            with col1:
                crit = st.text_input(f"Critério {i+1}", value=f"C{i+1}", key=f"crit_{i}")
                criteria.append(crit)
            with col2:
                ctype = st.selectbox("Tipo", ['benefit', 'cost'], key=f"type_{i}")
                criteria_types.append(ctype)
        
        # Entrada de dados
        st.markdown(t("decision_matrix", lang))
        data = []
        for i in range(n_alternatives):
            row = []
            cols = st.columns(n_criteria)
            for j in range(n_criteria):
                with cols[j]:
                    val = st.number_input(
                        f"{alternatives[i][:10]}-{criteria[j][:10]}",
                        value=float(np.random.randint(50, 100)),
                        key=f"data_{i}_{j}",
                        label_visibility="collapsed"
                    )
                    row.append(val)
            data.append(row)
    
    # =============================================================================
    # INFORMAÇÕES DOS CRITÉRIOS (após dados carregados)
    # =============================================================================
    # Para exemplos pré-definidos: critérios são somente leitura
    # Para custom/upload: critérios são editáveis
    if len(criteria) > 0 and len(criteria_types) > 0:

        if data_source in ('custom', 'upload'):
            # Modo editável apenas para custom e upload
            with st.expander(t("criteria_info", lang)):
                st.markdown(t("verify_change", lang))
                for i, (crit, ctype) in enumerate(zip(criteria, criteria_types)):
                    col1, col2, col3 = st.columns([3, 2, 3])
                    with col1:
                        st.write(f"**{crit}**")
                    with col2:
                        new_type = st.selectbox(
                            "Tipo",
                            ['benefit', 'cost'],
                            index=0 if ctype == 'benefit' else 1,
                            key=f"info_type_{i}"
                        )
                        criteria_types[i] = new_type
                    with col3:
                        st.write(t("maximize", lang) if new_type == 'benefit' else t("minimize", lang))

        # Mostrar matriz com highlight (para custom/upload, a matriz já é exibida
        # acima nos exemplos; aqui exibimos apenas para custom/upload)
        if data_source in ('custom', 'upload') and len(data) > 0:
            st.subheader(t("decision_matrix", lang))
            df = pd.DataFrame(data, columns=criteria, index=alternatives)
            st.dataframe(
                df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral'),
                use_container_width=True
            )
    
    # =============================================================================
    # PARÂMETROS MANUAIS (se ML desabilitado)
    # =============================================================================
    manual_params = None
    if not use_ml and len(criteria) > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"### {t('manual_params', lang)}")
        lambda_param = st.sidebar.slider("λ (balanceamento)", 0.0, 1.0, 0.5, 0.05)
        mu_param = st.sidebar.slider("μ (centro gaussiano)", 0.0, float(len(criteria)-1), len(criteria)/2, 0.1)
        sigma_param = st.sidebar.slider("σ (dispersão)", 0.5, float(len(criteria)), len(criteria)/4, 0.1)
        
        manual_params = {
            'lambda': lambda_param,
            'mu': mu_param,
            'sigma': sigma_param
        }
    
    # =============================================================================
    # EXECUTAR ANÁLISE
    # =============================================================================
    if len(data) > 0 and len(criteria) > 0:
        if st.button(t("run_analysis", lang), type="primary"):
            
            with st.spinner(t("processing", lang)):
                results = cobra_ml.run_cobra_ml(
                    X=data,
                    criteria_types=criteria_types,
                    use_ml=use_ml,
                    ml_params=manual_params
                )
            
            st.success(t("analysis_complete", lang))
            
            # Salvar resultados no session_state
            st.session_state["cobra_results"] = results
            st.session_state["cobra_alternatives"] = alternatives
            st.session_state["cobra_criteria"] = criteria
    
    # =============================================================================
    # EXIBIR RESULTADOS (FORA DO BLOCO DO BOTÃO)
    # =============================================================================
    if "cobra_results" in st.session_state:
        # Recuperar resultados salvos
        results = st.session_state["cobra_results"]
        alternatives = st.session_state["cobra_alternatives"]
        criteria = st.session_state["cobra_criteria"]
        
        st.header(t("results", lang))
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            t("final_ranking", lang),
            t("weights_analysis", lang),
            t("matrices", lang),
            t("ml_parameters", lang),
            t("visualizations", lang)
        ])
        
        # TAB 1: RANKING FINAL
        with tab1:
            st.subheader(t("final_ranking", lang))
            
            ranking_df = pd.DataFrame({
                t("position", lang): range(1, len(alternatives) + 1),
                t("alternative", lang): [alternatives[i] for i in results['ranking']],
                t("score", lang): [f"{results['scores'][i]:.4f}" for i in results['ranking']]
            })
            
            st.dataframe(ranking_df, use_container_width=True)
            
            # Gráfico de barras
            fig_ranking = go.Figure(data=[
                go.Bar(
                    x=[alternatives[i] for i in results['ranking']],
                    y=[results['scores'][i] for i in results['ranking']],
                    marker_color='lightblue',
                    text=[f"{results['scores'][i]:.4f}" for i in results['ranking']],
                    textposition='auto',
                )
            ])
            fig_ranking.update_layout(
                title=t("score", lang),
                xaxis_title=t("alternatives", lang),
                yaxis_title=t("score", lang),
                showlegend=False
            )
            st.plotly_chart(fig_ranking, use_container_width=True)
        
        # TAB 2: ANÁLISE DE PESOS
        with tab2:
            st.subheader(t("weights_analysis", lang))
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(t("psi_weights", lang))
                psi_df = pd.DataFrame({
                    'Critério': criteria,
                    'Peso PSI': results['weights_psi']
                })
                st.dataframe(psi_df.style.format({'Peso PSI': '{:.4f}'}))
            
            with col2:
                st.write(t("final_weights", lang))
                final_df = pd.DataFrame({
                    'Critério': criteria,
                    'Peso Final': results['weights_final']
                })
                st.dataframe(final_df.style.format({'Peso Final': '{:.4f}'}))
            
            # Gráfico comparativo
            fig_weights = go.Figure()
            fig_weights.add_trace(go.Bar(
                name='PSI',
                x=criteria,
                y=results['weights_psi'],
                marker_color='indianred'
            ))
            fig_weights.add_trace(go.Bar(
                name='Final' if lang == "English" else 'Final (Balanceado)',
                x=criteria,
                y=results['weights_final'],
                marker_color='lightsalmon'
            ))
            fig_weights.update_layout(
                title="PSI vs Final",
                barmode='group',
                xaxis_title="Critérios",
                yaxis_title="Peso"
            )
            st.plotly_chart(fig_weights, use_container_width=True)
        
        # TAB 3: MATRIZES
        with tab3:
            st.subheader(t("matrices", lang))
            
            st.write(t("normalized_matrix", lang))
            st.info("ℹ️ " + ("Valores originais transformados para escala [0,1]" if lang == "Portuguese" else "Original values transformed to [0,1] scale"))
            norm_df = pd.DataFrame(
                results['normalized_matrix'],
                columns=criteria,
                index=alternatives
            )
            try:
                st.dataframe(
                    norm_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn'),
                    use_container_width=True
                )
            except ImportError:
                # Fallback sem matplotlib: mantém a tabela formatada, sem gradiente de cor
                st.dataframe(norm_df.style.format("{:.4f}"), use_container_width=True)
            
            st.write(t("weighted_matrix", lang))
            st.info("ℹ️ " + ("Matriz normalizada × pesos finais" if lang == "Portuguese" else "Normalized matrix × final weights"))
            weighted_df = pd.DataFrame(
                results['weighted_matrix'],
                columns=criteria,
                index=alternatives
            )
            try:
                st.dataframe(
                    weighted_df.style.format("{:.4f}").background_gradient(cmap='Blues'),
                    use_container_width=True
                )
            except ImportError:
                # Fallback sem matplotlib: mantém a tabela formatada, sem gradiente de cor
                st.dataframe(weighted_df.style.format("{:.4f}"), use_container_width=True)
        
        # TAB 4: PARÂMETROS
        with tab4:
            st.subheader(t("model_parameters", lang))
            
            params_df = pd.DataFrame({
                t("parameter", lang): ['λ (Lambda)', 'μ (Mu)', 'σ (Sigma)'],
                t("value", lang): [
                    f"{results['params']['lambda']:.4f}",
                    f"{results['params']['mu']:.4f}",
                    f"{results['params']['sigma']:.4f}"
                ],
                t("description", lang): [
                    'Balanceamento PSI vs Subjetivo' if lang == "Portuguese" else 'PSI vs Subjective Balance',
                    'Centro da distribuição gaussiana' if lang == "Portuguese" else 'Gaussian distribution center',
                    'Dispersão da gaussiana' if lang == "Portuguese" else 'Gaussian dispersion'
                ]
            })
            st.dataframe(params_df, use_container_width=True)
            
            if use_ml and cobra_ml.ml_model is not None:
                st.success("🤖 " + (
                    "Parâmetros preditos pelo modelo ML treinado (Random Forest Regressor)"
                    if lang == "Portuguese" else
                    "Parameters predicted by the trained ML model (Random Forest Regressor)"
                ))
            elif use_ml and cobra_ml.ml_model is None:
                st.warning("⚠️ " + (
                    "ML automático selecionado, mas nenhum modelo treinado encontrado. "
                    "Parâmetros padrão foram usados. Treine o modelo na seção abaixo."
                    if lang == "Portuguese" else
                    "Automatic ML selected, but no trained model found. "
                    "Default parameters were used. Train the model in the section below."
                ))
            else:
                st.info(t("manual_default_params", lang))
            
            # Visualizar função gaussiana
            st.write(t("gaussian_viz", lang))
            x = np.arange(len(criteria))
            y = [cobra_ml.gaussian_function(j, results['params']['mu'], 
                                           results['params']['sigma'], 
                                           len(criteria)) for j in x]
            
            fig_gaussian = go.Figure()
            fig_gaussian.add_trace(go.Scatter(
                x=criteria,
                y=y,
                mode='lines+markers',
                line=dict(color='purple', width=2),
                marker=dict(size=8)
            ))
            fig_gaussian.update_layout(
                title=t("gaussian_viz", lang),
                xaxis_title="Critérios",
                yaxis_title="Fator Gaussiano",
                showlegend=False
            )
            st.plotly_chart(fig_gaussian, use_container_width=True)
        
        # TAB 5: VISUALIZAÇÕES ADICIONAIS
        with tab5:
            st.subheader(t("additional_viz", lang))
            
            # Heatmap
            st.write(t("heatmap", lang))
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=results['normalized_matrix'],
                x=criteria,
                y=alternatives,
                colorscale='Reds',
                text=np.round(results['normalized_matrix'], 3),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            fig_heatmap.update_layout(
                title=t("heatmap", lang),
                xaxis_title="Critérios",
                yaxis_title=t("alternatives", lang)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Radar chart com selectbox
            st.write(t("radar_chart", lang))
            selected_alternative = st.selectbox(
                t("select_alternative", lang),
                alternatives,
                index=int(results['ranking'][0]),
                key="radar_selectbox"
            )
            
            if selected_alternative:
                fig_radar = go.Figure()
                alt_idx = alternatives.index(selected_alternative)
                fig_radar.add_trace(go.Scatterpolar(
                    r=results['normalized_matrix'][alt_idx],
                    theta=criteria,
                    fill='toself',
                    name=selected_alternative
                ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title=t("radar_chart", lang)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        
        # =============================================================================
        # SOBRE O MÉTODO
        # =============================================================================
        with st.expander(t("about_method", lang)):
            if lang == "Portuguese":
                st.markdown("""
                ### Método COBRA-ML
                
                O **COBRA-ML** é uma extensão do método COBRA (Comprehensive Distance Based Ranking) que integra:
                
                1. **PSI (Preference Selection Index)**: Cálculo objetivo de pesos baseado na variação dos critérios
                2. **Balanceamento Gaussiano**: Ajuste dos pesos usando uma função gaussiana
                3. **Machine Learning**: Predição automática dos parâmetros ótimos (λ, μ, σ)
                
                #### Fluxo do Método:
                1. Normalização da matriz de decisão
                2. Cálculo dos pesos PSI (objetivos)
                3. Predição de parâmetros via ML (ou definição manual)
                4. Balanceamento gaussiano dos pesos
                5. Cálculo das distâncias COBRA (PIS e NIS)
                6. Geração dos scores finais e ranking
                
                #### Aplicações:
                - ✅ Seleção de fontes de energia renovável
                - ✅ Avaliação de tecnologias
                - ✅ Seleção de fornecedores
                - ✅ Escolha de equipamentos militares
                - ✅ Qualquer problema de tomada de decisão multicritério
                
                **Desenvolvido para pesquisa em Engenharia de Produção e Pesquisa Operacional**
                """)
            else:
                st.markdown("""
                ### COBRA-ML Method
                
                **COBRA-ML** is an extension of COBRA (Comprehensive Distance Based Ranking) that integrates:
                
                1. **PSI (Preference Selection Index)**: Objective weight calculation based on criteria variation
                2. **Gaussian Balancing**: Weight adjustment using Gaussian function
                3. **Machine Learning**: Automatic prediction of optimal parameters (λ, μ, σ)
                
                #### Method Flow:
                1. Decision matrix normalization
                2. PSI weights calculation (objective)
                3. Parameter prediction via ML (or manual definition)
                4. Gaussian weight balancing
                5. COBRA distances calculation (PIS and NIS)
                6. Final scores and ranking generation
                
                #### Applications:
                - ✅ Renewable energy source selection
                - ✅ Technology assessment
                - ✅ Supplier selection
                - ✅ Military equipment choice
                - ✅ Any multi-criteria decision-making problem
                
                **Developed for research in Production Engineering and Operations Research**
                """)
        
        # =============================================================================
        # EXPORTAR RESULTADOS
        # =============================================================================
        with st.expander(t("export_results", lang)):
            st.markdown(t("export_subtitle", lang))
            
            export_data = {
                'Alternativa': alternatives,
                'Score': results['scores'].tolist(),
                'Ranking': [list(results['ranking']).index(i) + 1 for i in range(len(alternatives))]
            }
            
            for j, crit in enumerate(criteria):
                export_data[f'Peso_PSI_{crit}'] = [results['weights_psi'][j]] * len(alternatives)
                export_data[f'Peso_Final_{crit}'] = [results['weights_final'][j]] * len(alternatives)
            
            export_df = pd.DataFrame(export_data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label=t("download_csv", lang),
                    data=csv,
                    file_name="cobra_ml_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name='Resultados', index=False)
                    norm_df.to_excel(writer, sheet_name='Matriz_Normalizada')
                    weighted_df.to_excel(writer, sheet_name='Matriz_Ponderada')
                
                st.download_button(
                    label=t("download_excel", lang),
                    data=buffer.getvalue(),
                    file_name="cobra_ml_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            with col3:
                json_data = {
                    'results': export_data,
                    'parameters': results['params'],
                    'method': 'COBRA-ML'
                }
                json_str = json.dumps(json_data, indent=2, default=str)
                st.download_button(
                    label=t("download_json", lang),
                    data=json_str,
                    file_name="cobra_ml_results.json",
                    mime="application/json"
                )
        
        # =============================================================================
        # TREINAMENTO DO MODELO ML COM DATASET SINTÉTICO
        # =============================================================================
        st.markdown("---")
        _ds_title = (
            "🤖 Treinar Modelo ML com Dataset Sintético"
            if lang == "Portuguese" else
            "🤖 Train ML Model with Synthetic Dataset"
        )
        with st.expander(_ds_title):

            if lang == "Portuguese":
                st.markdown("""
O modelo de **Machine Learning** do COBRA-ML (Random Forest Regressor) prevê automaticamente
os parâmetros ótimos **λ, μ e σ** a partir das características estruturais do seu problema MCDM —
dispensando a definição manual quando o modelo está treinado.

**Como funciona o fluxo completo aqui no app:**
1. Você configura e gera um dataset de problemas MCDM sintéticos
2. O modelo é **treinado diretamente aqui**, sem precisar sair do app
3. Na próxima análise, selecione **ML automático** na barra lateral — o modelo treinado será usado
4. Opcionalmente baixe o dataset para uso externo ou para registrar o experimento
                """)
            else:
                st.markdown("""
The **Machine Learning** model in COBRA-ML (Random Forest Regressor) automatically predicts
the optimal parameters **λ, μ and σ** from the structural features of your MCDM problem —
no manual tuning needed once the model is trained.

**Full workflow — all inside this app:**
1. Configure and generate a dataset of synthetic MCDM problems
2. The model is **trained directly here**, no need to leave the app
3. On the next analysis, select **Automatic ML** in the sidebar — the trained model will be used
4. Optionally download the dataset for external use or to record the experiment
                """)

            st.markdown("---")

            # ── Step 1: configurar geração ────────────────────────────────
            col_ds1, col_ds2 = st.columns(2)
            with col_ds1:
                n_problems = st.slider(
                    ("Nº de problemas sintéticos" if lang == "Portuguese"
                     else "No. of synthetic problems"),
                    min_value=50, max_value=500, value=150, step=50,
                    help=("Mais problemas = modelo mais robusto, mas mais lento para treinar."
                          if lang == "Portuguese" else
                          "More problems = more robust model, but slower to train."),
                    key="ds_n_problems"
                )
            with col_ds2:
                ds_seed = st.number_input(
                    "Random Seed",
                    min_value=0, max_value=9999, value=42,
                    help=("Mesmo seed = mesmo dataset. Útil para reprodutibilidade."
                          if lang == "Portuguese" else
                          "Same seed = same dataset. Useful for reproducibility."),
                    key="ds_seed"
                )

            # ── Step 2: Gerar + Treinar ───────────────────────────────────
            btn_label = (
                "🚀 Gerar Dataset e Treinar Modelo"
                if lang == "Portuguese" else
                "🚀 Generate Dataset and Train Model"
            )
            if st.button(btn_label, key="btn_generate_and_train", type="primary"):
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
                from sklearn.ensemble import RandomForestClassifier

                # 2a. Gerar dataset
                with st.spinner("Gerando dataset..." if lang == "Portuguese" else "Generating dataset..."):
                    ds_df = generate_synthetic_dataset(n_problems, seed=int(ds_seed))
                    st.session_state["synthetic_dataset"] = ds_df

                # 2b. Treinar regressor (lambda, sigma) + classificador (mu)
                with st.spinner("Treinando modelos..." if lang == "Portuguese" else "Training models..."):
                    training_data = generate_training_data(n_samples=n_problems)
                    trained_cobra = COBRA_ML()
                    trained_cobra.train_ml_model(training_data)
                    st.session_state["trained_ml_model"]      = trained_cobra.ml_model
                    st.session_state["trained_ml_classifier"] = trained_cobra.ml_classifier
                    st.session_state["trained_ml_scaler"]     = trained_cobra.scaler

                # 2c. Validacao hold-out 20% com metricas corretas por parametro
                feat_cols = ['m','n','benefit_ratio','psi_weight_variance',
                             'psi_weight_entropy','data_std','data_mean','mean_adj_variation']

                X_all    = ds_df[feat_cols].values
                lam_all  = ds_df['lambda_opt'].values
                mu_all   = ds_df['mu_opt'].astype(int).values
                sig_all  = ds_df['sigma_opt'].values

                X_tr, X_te, idx_tr, idx_te = train_test_split(
                    X_all, np.arange(len(X_all)), test_size=0.2, random_state=int(ds_seed)
                )
                sc2 = StandardScaler().fit(X_tr)
                X_tr_s, X_te_s = sc2.transform(X_tr), sc2.transform(X_te)

                # Regressor: lambda e sigma
                rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10,
                                               random_state=42, n_jobs=-1)
                y_reg_tr = np.column_stack([lam_all[idx_tr], sig_all[idx_tr]])
                y_reg_te = np.column_stack([lam_all[idx_te], sig_all[idx_te]])
                rf_reg.fit(X_tr_s, y_reg_tr)
                y_reg_pred = rf_reg.predict(X_te_s)
                r2_lam  = r2_score(y_reg_te[:,0], y_reg_pred[:,0])
                mae_lam = mean_absolute_error(y_reg_te[:,0], y_reg_pred[:,0])
                r2_sig  = r2_score(y_reg_te[:,1], y_reg_pred[:,1])
                mae_sig = mean_absolute_error(y_reg_te[:,1], y_reg_pred[:,1])

                # Classificador: mu
                rf_cls = RandomForestClassifier(n_estimators=100, max_depth=10,
                                                random_state=42, n_jobs=-1)
                rf_cls.fit(X_tr_s, mu_all[idx_tr])
                mu_pred = rf_cls.predict(X_te_s)
                acc_mu  = accuracy_score(mu_all[idx_te], mu_pred)
                f1_mu   = f1_score(mu_all[idx_te], mu_pred, average='weighted', zero_division=0)

                # Feature importance media dos dois modelos
                fi_avg = (rf_reg.feature_importances_ + rf_cls.feature_importances_) / 2

                st.session_state["ml_validation"] = {
                    "r2_lam": r2_lam,   "mae_lam": mae_lam,
                    "r2_sig": r2_sig,   "mae_sig": mae_sig,
                    "acc_mu": acc_mu,   "f1_mu":   f1_mu,
                    "n_train": len(X_tr), "n_test": len(X_te),
                    "feature_importance": dict(zip(feat_cols, fi_avg.tolist())),
                }
                st.success("Modelos treinados e prontos para uso!" if lang == "Portuguese"
                           else "Models trained and ready to use!")

            # Step 3: Exibir metricas de validacao
            if "ml_validation" in st.session_state:
                val = st.session_state["ml_validation"]

                st.markdown("#### " + ("Resultados da Validacao (hold-out 20%)"
                             if lang == "Portuguese" else "Validation Results (hold-out 20%)"))

                # Lambda e Sigma: R2 + MAE
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R² (λ)", f"{val['r2_lam']:.3f}",
                          help="Coeficiente de determinacao para lambda" if lang == "Portuguese"
                               else "Coefficient of determination for lambda")
                c2.metric("MAE (λ)", f"{val['mae_lam']:.4f}",
                          help="Erro absoluto medio — lambda in [0.1, 0.9]" if lang == "Portuguese"
                               else "Mean absolute error — lambda in [0.1, 0.9]")
                c3.metric("R² (σ)", f"{val['r2_sig']:.3f}",
                          help="Coeficiente de determinacao para sigma")
                c4.metric("MAE (σ)", f"{val['mae_sig']:.4f}",
                          help="Erro absoluto medio — sigma")

                # Mu: Accuracy + F1
                c5, c6, c7 = st.columns(3)
                c5.metric("Accuracy (μ)", f"{val['acc_mu']:.3f}",
                          help="Acuracia da classificacao do criterio dominante" if lang == "Portuguese"
                               else "Classification accuracy of the dominant criterion index")
                c6.metric("F1-score (μ)", f"{val['f1_mu']:.3f}",
                          help="F1 ponderado — classifica o indice do criterio dominante" if lang == "Portuguese"
                               else "Weighted F1 — classifies dominant criterion index")
                c7.metric("Train / Test", f"{val['n_train']} / {val['n_test']}")

                # Feature importance
                if "feature_importance" in val:
                    fi = val["feature_importance"]
                    fi_sorted = sorted(fi.items(), key=lambda x: -x[1])
                    fi_label = "Importancia das Features (media RF)" if lang == "Portuguese" else "Feature Importance (RF average)"
                    with st.expander(fi_label):
                        for feat, imp in fi_sorted:
                            st.progress(float(imp), text=f"{feat}: {imp:.4f}")

                _model_ready_msg = (
                    "**Modelos treinados e ativos.** Selecione **ML automatico** na barra lateral "
                    "e rode uma nova analise — os parametros serao preditos automaticamente."
                    if lang == "Portuguese" else
                    "**Models trained and active.** Select **Automatic ML** in the sidebar "
                    "and run a new analysis — parameters will be predicted automatically."
                )
                st.info(_model_ready_msg)

            # ── Step 4: Download opcional do dataset ──────────────────────
            if "synthetic_dataset" in st.session_state:
                ds_df = st.session_state["synthetic_dataset"]
                with st.expander("📥 " + ("Baixar dataset gerado" if lang == "Portuguese"
                                           else "Download generated dataset")):
                    st.dataframe(ds_df.head(8), use_container_width=True)
                    st.caption(f"{'Total de linhas' if lang == 'Portuguese' else 'Total rows'}: {len(ds_df)}")
                    st.download_button(
                        label=("📥 Download CSV" if lang == "English" else "📥 Baixar CSV"),
                        data=ds_df.to_csv(index=False),
                        file_name="cobra_ml_synthetic_dataset.csv",
                        mime="text/csv",
                        key="download_dataset"
                    )

        st.markdown("---")
        st.header(t("ai_analysis", lang))
        st.write(t("ai_help", lang))
        
        client = setup_openai_client()
        
        if client is None:
            st.warning(t("ai_key_warning", lang))
        else:
            # Preparar resumo dos resultados para IA
            results_summary = {
                'ranking': {alternatives[i]: float(results['scores'][i]) for i in range(len(alternatives))},
                'top_alternative': alternatives[results['ranking'][0]],
                'weights_psi': {criteria[j]: float(results['weights_psi'][j]) for j in range(len(criteria))},
                'weights_final': {criteria[j]: float(results['weights_final'][j]) for j in range(len(criteria))},
                'parameters': results['params']
            }
            
            # Botão para gerar análise
            if st.button(t("ai_generate", lang), key="generate_ai"):
                with st.spinner(t("ai_analyzing", lang)):
                    ai_text = analyze_with_ai(client, lang, results_summary)
                    
                    if ai_text and not ai_text.startswith("❌"):
                        st.session_state["ai_last_analysis"] = ai_text
                        st.session_state["ai_enabled"] = True
                        st.success(t("ai_generated", lang))
                        st.info(ai_text)
                        st.session_state["ai_chat_history"] = []
                    else:
                        st.error(ai_text or t("ai_error", lang))
            
            # Chat interativo com IA
            if st.session_state.get("ai_enabled", False):
                st.markdown("---")
                st.header(t("ai_chat", lang))
                
                # Exibir histórico do chat
                for msg in st.session_state.get("ai_chat_history", []):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                
                # Input do chat
                prompt = st.chat_input(t("ai_chat_placeholder", lang))
                
                if prompt:
                    # Adicionar mensagem do usuário
                    st.session_state["ai_chat_history"].append({
                        "role": "user",
                        "content": prompt
                    })
                    
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Obter resposta da IA
                    with st.spinner(t("ai_thinking", lang)):
                        resp = chat_with_ai(client, lang, prompt)
                    
                    assistant_text = resp.get("answer", "") if not resp.get("error") else f"❌ Error: {resp.get('error')}"
                    
                    # Adicionar resposta da IA
                    st.session_state["ai_chat_history"].append({
                        "role": "assistant",
                        "content": assistant_text
                    })
                    
                    with st.chat_message("assistant"):
                        st.markdown(assistant_text)
                    
                    st.rerun()

if __name__ == "__main__":
    main()