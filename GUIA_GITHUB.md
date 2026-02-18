# 📦 GUIA DE UPLOAD PARA GITHUB - COBRA-ML

## ✅ Arquivos Prontos para Upload

Você recebeu os seguintes arquivos para subir no GitHub:

### 1️⃣ **Arquivo Principal**
- `cobra_ml_app_integrated_FIXED4.py` - Aplicação completa (1256 linhas)

### 2️⃣ **Documentação**
- `README.md` - Documentação completa do projeto
- `FORMULACAO_MATEMATICA_COBRA_ML.py` - Equações completas em LaTeX para o artigo

### 3️⃣ **Configuração**
- `requirements.txt` - Dependências Python
- `.gitignore` - Arquivos a serem ignorados pelo Git

### 4️⃣ **Extras (Opcionais)**
- `exemplo_numerico_cobra_ml.py` - Tutorial passo-a-passo
- `guia_ml_cobra.py` - Guia didático de ML
- `CHANGELOG.md` - Histórico de mudanças

---

## 🚀 Passo a Passo para Subir no GitHub

### Opção A: Criar Repositório Novo

#### 1. No GitHub (navegador)
```
1. Acesse github.com
2. Clique em "New repository"
3. Nome: cobra-ml (ou outro de sua escolha)
4. Descrição: "COBRA-ML: Método Multicritério com Machine Learning"
5. Escolha: Public ou Private
6. NÃO marque "Initialize with README" (já temos um)
7. Clique em "Create repository"
```

#### 2. No seu computador (terminal)
```bash
# Entrar na pasta do projeto
cd caminho/para/sua/pasta/cobra-ml

# Inicializar Git
git init

# Adicionar arquivos
git add .

# Fazer primeiro commit
git commit -m "Initial commit: COBRA-ML v1.0"

# Conectar com GitHub (substitua SEU-USUARIO pelo seu username)
git remote add origin https://github.com/SEU-USUARIO/cobra-ml.git

# Enviar para GitHub
git branch -M main
git push -u origin main
```

### Opção B: Atualizar Repositório Existente

```bash
# Entrar na pasta do projeto
cd caminho/para/sua/pasta/cobra-ml

# Verificar status
git status

# Adicionar novos arquivos
git add .

# Fazer commit
git commit -m "Update: FIXED4 version with complete formulation"

# Enviar para GitHub
git push
```

---

## 📋 Estrutura Recomendada do Repositório

Organize os arquivos assim no GitHub:

```
cobra-ml/
│
├── README.md                                    # Documentação principal
├── requirements.txt                             # Dependências
├── .gitignore                                   # Arquivos ignorados
│
├── cobra_ml_app_integrated_FIXED4.py            # Aplicação principal ⭐
│
├── docs/                                        # Pasta de documentação
│   ├── FORMULACAO_MATEMATICA_COBRA_ML.py        # Equações LaTeX
│   ├── exemplo_numerico_cobra_ml.py             # Tutorial
│   └── guia_ml_cobra.py                         # Guia ML
│
├── assets/                                      # Recursos (opcional)
│   └── UFF_EN_brasao.png                        # Logo institucional
│
└── CHANGELOG.md                                 # Histórico de versões
```

**Para criar a pasta docs:**
```bash
mkdir docs
mv FORMULACAO_MATEMATICA_COBRA_ML.py docs/
mv exemplo_numerico_cobra_ml.py docs/
mv guia_ml_cobra.py docs/
```

---

## ⚠️ IMPORTANTE: Não Commitar

Certifique-se de que os seguintes arquivos/pastas **NÃO** estão no repositório:

### 🔴 Nunca Commitar:
- `code.env` (chave OpenAI)
- `.env` (variáveis de ambiente)
- `__pycache__/` (cache Python)
- `.vscode/` ou `.idea/` (configurações de IDE)
- Chaves de API ou senhas

O `.gitignore` já está configurado para ignorar estes arquivos.

### Verificar Antes de Commitar:
```bash
# Ver o que será enviado
git status

# Se vir algo que não deve ser enviado:
git reset HEAD nome_do_arquivo
```

---

## 📝 Mensagens de Commit Sugeridas

Use mensagens claras e descritivas:

**Exemplos:**
```bash
git commit -m "feat: Add vectorial normalization to COBRA method"
git commit -m "docs: Update mathematical formulation with LaTeX"
git commit -m "fix: Correct PSI weight calculation"
git commit -m "refactor: Improve ML parameter prediction"
git commit -m "docs: Add bilingual README (PT/EN)"
```

**Prefixos comuns:**
- `feat:` Nova funcionalidade
- `fix:` Correção de bug
- `docs:` Mudanças na documentação
- `refactor:` Refatoração de código
- `test:` Adição/modificação de testes
- `chore:` Tarefas gerais (dependências, configuração)

---

## 🔗 Depois de Subir no GitHub

### 1. Adicionar Badges ao README
Edite o README.md no GitHub e adicione badges no topo:

```markdown
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Academic-green)]()
[![Stars](https://img.shields.io/github/stars/SEU-USUARIO/cobra-ml?style=social)](https://github.com/SEU-USUARIO/cobra-ml)
```

### 2. Criar Releases (Versões)

No GitHub:
```
1. Vá em "Releases" → "Create a new release"
2. Tag: v1.0.0
3. Title: "COBRA-ML v1.0.0 - Initial Release"
4. Descrição: Listar principais funcionalidades
5. Anexar arquivo ZIP (opcional)
6. Publicar
```

### 3. Adicionar Topics (Tags)

No repositório GitHub:
```
Settings → Topics → Adicionar:
- multi-criteria-decision-making
- machine-learning
- operations-research
- mcdm
- python
- streamlit
- decision-support-system
```

### 4. Criar GitHub Pages (Opcional)

Para documentação online:
```
Settings → Pages → Source: main branch / docs folder
```

---

## 🎓 Para o Artigo SBPO

### Citando o Repositório no Artigo:

**No texto:**
> "O método COBRA-ML foi implementado em Python e está disponível publicamente 
> em https://github.com/seu-usuario/cobra-ml"

**Nas Referências:**
```
PORTELLA, Anderson; SANTOS, Marcos dos; GOMES, Carlos Francisco Simões. 
COBRA-ML: GitHub Repository. Disponível em: 
https://github.com/seu-usuario/cobra-ml. Acesso em: 18 fev. 2024.
```

### QR Code para Apresentação:

Gere um QR Code apontando para o repositório:
- Use: https://www.qr-code-generator.com/
- Cole a URL do seu repositório GitHub
- Adicione nos slides da apresentação SBPO

---

## 📊 Estatísticas do Projeto

Após upload, você pode ver no GitHub:

- **Total de linhas de código**: ~3.500+
- **Linguagem principal**: Python
- **Arquivos**: 8+ principais
- **Documentação**: Completa (README + Formulação + Tutoriais)

---

## ✅ Checklist Final

Antes de considerar completo:

- [ ] Todos os arquivos foram commitados
- [ ] `.gitignore` está funcionando (sem arquivos sensíveis)
- [ ] README.md está correto e bem formatado
- [ ] requirements.txt está completo
- [ ] Código roda localmente (`streamlit run cobra_ml_app_integrated_FIXED4.py`)
- [ ] Links no README apontam para lugares corretos
- [ ] Informações dos autores estão corretas
- [ ] License está clara
- [ ] Changelog está atualizado

---

## 🆘 Problemas Comuns

### "Permission denied"
```bash
# Solução: Configurar SSH ou usar HTTPS com token
git remote set-url origin https://github.com/SEU-USUARIO/cobra-ml.git
```

### "Push rejected"
```bash
# Solução: Fazer pull primeiro
git pull origin main --rebase
git push
```

### "Large file warning"
```bash
# Solução: Arquivos > 50MB não devem estar no Git
# Adicione-os ao .gitignore
```

---

## 📞 Contato para Dúvidas

Se tiver problemas com GitHub:
- Documentação oficial: https://docs.github.com/
- GitHub Desktop (interface gráfica): https://desktop.github.com/

---

## 🎉 Parabéns!

Seu projeto COBRA-ML está pronto para o GitHub e para o artigo SBPO!

**Próximos passos sugeridos:**
1. ✅ Upload no GitHub
2. ✅ Compartilhar link no artigo
3. ✅ Apresentar no SBPO
4. ✅ Publicar artigo
5. ✅ Divulgar na comunidade MCDM

**Boa sorte na apresentação! 🚀**
