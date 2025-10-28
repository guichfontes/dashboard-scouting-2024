# ⚽ Dashboard de Scouting — Jogadores "Bons e Baratos" (Brasileirão 2024)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-%E2%9A%A1-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Online-success)

Este dashboard interativo foi desenvolvido para **analisar dados de jogadores do Campeonato Brasileiro 2024**, com o objetivo de **identificar atletas de bom desempenho (outliers positivos) e baixo custo de mercado** — os chamados *bons e baratos*.

Utilizando [ScraperFC](https://scraperfc.readthedocs.io/en/latest/), Os dados foram coletados do [FBref](https://fbref.com/) e do [Transfermarkt](transfermarkt.com.br), processados e analisados com técnicas de *machine learning não supervisionado* (DBSCAN e PCA), visando apoiar o **scouting e a recomendação de reforços** para clubes brasileiros.

---

## 🔗 Acesso ao Dashboard

O dashboard está disponível em:  
👉 **[https://dashboard-scouting-de-bons-e-baratos.onrender.com/](https://dashboard-scouting-de-bons-e-baratos.onrender.com/)**  

> Para obter **usuário e senha de acesso**, entre em contato com:  
> 📧 **guilhermefontes@ufba.br**

---

## 🧠 Principais Funcionalidades

- **Login Seguro** para acesso restrito.
- **Seleção de posição** (Atacantes, Meio-Campistas, Defensores e Goleiros).  
- **Remoção de variáveis** para ajustar a base de análise.
- **Visualização da matriz de correlação** (Plotly).
- **Busca automática dos melhores parâmetros do DBSCAN**, com avaliação por *Silhouette Score* e proporção de ruído.
- **Projeção PCA 3D interativa** dos clusters.
- **Ajuste de pesos personalizados** para cálculo de *performance* por posição.
- **Cálculo de custo-benefício individual e por cluster**.
- **Identificação de bons outliers** (jogadores com alta performance e baixo valor de mercado).
- **Listagem dos Top jogadores** em três categorias:
  1. Mais próximos do cluster de bons e baratos  
  2. Maior custo-benefício individual  
  3. Maior *score híbrido* (performance ajustada pela distância do cluster ideal)
- **Consulta detalhada** dos atributos de qualquer jogador.

---

## 🧩 Tecnologias Utilizadas

| Categoria | Ferramenta |
|------------|-------------|
| Interface | [Streamlit](https://streamlit.io) |
| Visualização | [Plotly Express](https://plotly.com/python/plotly-express/) |
| Manipulação de Dados | [Pandas](https://pandas.pydata.org/) |
| Machine Learning | [Scikit-Learn](https://scikit-learn.org/) |
| Estatística e Cálculos | NumPy, SciPy |
| Deploy | [Render](https://render.com/) |

---

## ⚙️ Como Executar Localmente

### 1. Clone o repositório:
```bash
git clone https://github.com/SEU_USUARIO/dashboard-bons-e-baratos.git
cd dashboard-bons-e-baratos
streamlit run main.py

---

## 👤 Autor

**Guilherme Fontes**  
🎓 Universidade Federal da Bahia (UFBA)
📧 guilhermefontes@ufba.br
