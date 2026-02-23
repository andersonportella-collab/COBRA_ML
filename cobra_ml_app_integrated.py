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
import plotly.express as px
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

**Software:**
SANTOS, Marcos dos; GOMES, Carlos Francisco Simões. **COBRA Machine Learning Tool**.
[S.l.]: Anderson Gonçalves Portella, 2025. Programa de Computador.

**Artigo:**
SILVA, C. S.; SANTOS, M. R.
Análise do nível de maturidade em gestão de riscos: um estudo de caso em uma empresa do setor elétrico.
In: CONGRESSO NACIONAL DE EXCELÊNCIA EM GESTÃO, 19., 2025, Online.
**Anais...** Rio de Janeiro: CNEG, 2025.
Acesso em: {date}.
""",
    "APA": """
**APA Format:**

**Software:**
Portella, A. G., Santos, M. dos, & Gomes, C. F. S. (2025).
*COBRA Machine Learning Tool* [Computer software].

**Article:**
Silva, C. S., & Santos, M. R. (2025).
Risk management maturity level analysis: a case study in the electric sector.
In *Proceedings of the XIX National Congress of Management Excellence*.
"""
}

def get_citation(lang: str) -> str:
    """Retorna a citação no formato apropriado conforme o idioma"""
    citation_type = "ABNT" if lang == "Portuguese" else "APA"
    today = datetime.datetime.now().strftime("%d %b. %Y")
    return CITATIONS[citation_type].format(date=today)

# OpenAI client — optional
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
        "ai_key_warning": "⚠️ Configure sua chave da API OpenAI para usar a análise com IA",
        "ai_generate": "🔍 Gerar Análise com IA",
        "ai_analyzing": "🤖 Analisando resultados...",
        "ai_generated": "✅ Análise gerada com sucesso!",
        "ai_error": "❌ Erro ao gerar análise",
        "ai_chat": "💬 Chat Interativo com IA",
        "ai_chat_placeholder": "Faça perguntas sobre os resultados...",
        "ai_thinking": "🤖 Pensando...",
        "language": "🌐 Idioma / Language",
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
        "ai_key_warning": "⚠️ Configure your OpenAI API key to use AI analysis",
        "ai_generate": "🔍 Generate AI Analysis",
        "ai_analyzing": "🤖 Analyzing results...",
        "ai_generated": "✅ Analysis generated successfully!",
        "ai_error": "❌ Error generating analysis",
        "ai_chat": "💬 Interactive AI Chat",
        "ai_chat_placeholder": "Ask questions about the results...",
        "ai_thinking": "🤖 Thinking...",
        "language": "🌐 Language / Idioma",
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
    """Setup OpenAI client with API key"""
    if OpenAI is None:
        return None
    
    api_key = os.getenv("OPENAI_API_KEY") or st.session_state.get("openai_api_key", None)
    
    if not api_key:
        api_key = st.sidebar.text_input(
            "🔑 OpenAI API Key",
            type="password",
            help="Insira sua chave da API da OpenAI / Enter your OpenAI API key"
        )
        if api_key:
            st.session_state["openai_api_key"] = api_key
    
    if api_key:
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            st.sidebar.error(f"Erro ao configurar OpenAI: {e}")
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
        self.ml_model = None
        self.scaler = StandardScaler()

    def normalize_matrix(self, X, criteria_types):
        X = np.array(X, dtype=float)
        R = np.zeros_like(X, dtype=float)
        
        for j in range(X.shape[1]):
            col = X[:, j]
            
            if criteria_types[j] == 'benefit':
                denom = np.sqrt(np.sum(col**2))
                R[:, j] = col / denom if denom != 0 else col
            else:  # cost
                inv_col = 1.0 / col
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
        """Treina o modelo de Machine Learning"""
        if len(training_data) < 5:
            st.warning("⚠️ Poucos dados de treinamento. Usando parâmetros padrão.")
            return None
        
        X_train = []
        y_train = []
        
        for problem in training_data:
            features = self.extract_features(problem['X'], problem['criteria_types'])
            X_train.append(features)
            y_train.append(problem['optimal_params'])
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.ml_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.ml_model.fit(X_train_scaled, y_train)
        return self.ml_model
    
    def predict_parameters(self, X, criteria_types):
        """Prediz os parâmetros ótimos usando ML"""
        if self.ml_model is None:
            n = X.shape[1]
            return {'lambda': 0.5, 'mu': n / 2, 'sigma': n / 4}
        
        features = self.extract_features(X, criteria_types).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        prediction = self.ml_model.predict(features_scaled)[0]
        
        lambda_param = np.clip(prediction[0], 0, 1)
        mu = np.clip(prediction[1], 0, X.shape[1] - 1)
        sigma = np.clip(prediction[2], 0.5, X.shape[1])
        
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

def generate_training_data(n_samples=20):
    """Gera dados de treinamento sintéticos para o modelo ML"""
    training_data = []
    
    for i in range(n_samples):
        m = np.random.randint(3, 10)
        n = np.random.randint(3, 8)
        X = np.random.uniform(10, 100, size=(m, n))
        criteria_types = ['benefit' if np.random.rand() > 0.3 else 'cost' for _ in range(n)]
        
        lambda_opt = 0.3 + 0.4 * (n / 10)
        mu_opt = n / 2 + np.random.uniform(-1, 1)
        sigma_opt = n / 4 + np.random.uniform(-0.5, 0.5)
        
        training_data.append({
            'X': X,
            'criteria_types': criteria_types,
            'optimal_params': [lambda_opt, mu_opt, sigma_opt]
        })
    
    return training_data


def create_sample_data(problem_type='energy'):
    """Cria dados de exemplo para diferentes tipos de problemas"""
    
    if problem_type == 'energy':
        alternatives = ['Solar PV', 'Eólica Onshore', 'Eólica Offshore', 'Hidrelétrica', 'Biomassa']
        criteria = ['LCOE ($/MWh)', 'Fator Capacidade (%)', 'Emissões CO2 (g/kWh)', 
                   'Área Necessária (km²/MW)', 'Maturidade Tecnológica (1-10)']
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
        criteria = ['Custo Unitário (M$)', 'Velocidade Máx (Mach)', 'Alcance (km)', 
                   'Carga Útil (kg)', 'Stealth (1-10)', 'Manutenibilidade (1-10)']
        criteria_types = ['cost', 'benefit', 'benefit', 'benefit', 'benefit', 'benefit']
        data = [
            [80, 1.6, 2200, 8000, 10, 6],
            [90, 2.5, 3900, 13000, 3, 8],
            [85, 1.8, 3700, 9500, 5, 7],
            [95, 2.0, 2900, 7500, 4, 6],
            [100, 2.0, 3500, 10000, 8, 4],
        ]
        
    else:  # suppliers
        alternatives = ['Fornecedor A', 'Fornecedor B', 'Fornecedor C', 'Fornecedor D']
        criteria = ['Preço', 'Qualidade (1-10)', 'Prazo Entrega (dias)', 
                   'Sustentabilidade (1-10)', 'Confiabilidade (1-10)']
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
    
    # Opções de ML
    use_ml = st.sidebar.checkbox(t("use_ml", lang), value=True)
    
    if use_ml:
        train_model = st.sidebar.checkbox(t("train_model", lang), value=False)
        if train_model:
            n_training_samples = st.sidebar.slider(t("training_samples", lang), 10, 100, 20)
    
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
        # Limpar todas as variáveis de sessão relacionadas aos resultados
        for key in list(st.session_state.keys()):
            if key.startswith('alt_') or key.startswith('crit_') or key.startswith('data_') or \
               key.startswith('type_') or key.startswith('info_type_') or key.startswith('ctype_') or \
               key.startswith('cobra_'):
                del st.session_state[key]
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**For support/contact:** andersonportella@yahoo.com.br")
    
    # Criar instância do COBRA-ML
    cobra_ml = COBRA_ML()
    
    # Treinar modelo se solicitado
    if use_ml and train_model:
        with st.spinner(t("processing", lang)):
            training_data = generate_training_data(n_training_samples)
            cobra_ml.train_ml_model(training_data)
            st.sidebar.success(f"✅ {t('analysis_complete', lang)}")
    
    # =============================================================================
    # ENTRADA DE DADOS
    # =============================================================================
    
    alternatives = []
    criteria = []
    criteria_types = []
    data = []
    
    if data_source == 'upload':
        st.subheader(t("upload_file", lang))
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
        # Exemplos pré-definidos (editáveis)
        alternatives, criteria, criteria_types, data = create_sample_data(data_source)
        
        st.subheader(t("editable_data", lang))
        
        # Permitir edição de alternativas
        st.markdown(t("alternatives", lang))
        alternatives_edit = []
        for i, alt in enumerate(alternatives):
            alt_new = st.text_input(f"{t('alternative', lang)} {i+1}", value=alt, key=f"alt_{i}")
            alternatives_edit.append(alt_new)
        alternatives = alternatives_edit
        
        # Permitir edição da matriz de decisão
        st.markdown(t("decision_matrix", lang))
        df_edit = pd.DataFrame(data, columns=criteria, index=alternatives)
        edited_df = st.data_editor(df_edit, use_container_width=True)
        data = edited_df.values.tolist()
        
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
                ctype = st.selectbox(f"Tipo", ['benefit', 'cost'], key=f"type_{i}")
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
    if len(criteria) > 0 and len(criteria_types) > 0:
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
        
        # Mostrar matriz com highlight
        if len(data) > 0:
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
            st.dataframe(norm_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn'))
            
            st.write(t("weighted_matrix", lang))
            st.info("ℹ️ " + ("Matriz normalizada × pesos finais" if lang == "Portuguese" else "Normalized matrix × final weights"))
            weighted_df = pd.DataFrame(
                results['weighted_matrix'],
                columns=criteria,
                index=alternatives
            )
            st.dataframe(weighted_df.style.format("{:.4f}").background_gradient(cmap='Blues'))
        
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
                st.success(t("predicted_params", lang))
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
        
        # =================================================================
        # SEÇÃO DE ANÁLISE COM IA
        # =================================================================
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