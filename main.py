import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from itertools import product
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from scipy.spatial import distance

# Usuários e senhas
USUARIOS = {"guilherme": "senha2024"}

# Inicializa estado
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

def login():
    st.title("Login - Dashboard de Scouting")
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if usuario in USUARIOS and USUARIOS[usuario] == senha:
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")

@st.cache_data
def carregar_dados(sheet_name):
    df = pd.read_excel("dados/dados_jogadores_br_24.xlsx", sheet_name=sheet_name)
    return df.copy()

@st.cache_data
def calcular_matriz_correlacao(df):
    return df.corr()

def avaliar_clustering(X, labels):
    mask = labels != -1
    n_ruido = np.sum(~mask)
    proporcao_ruido = n_ruido / len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return -1, proporcao_ruido
    sil_score = silhouette_score(X[mask], labels[mask])
    return sil_score, proporcao_ruido

def app():
    st.set_page_config(page_title="Scouting de Jogadores Bons e Baratos", layout="wide")
    st.title("Scouting de Jogadores Bons e Baratos - Campeonato Brasileiro 2024")

    # Botão de logout na barra lateral
    logout = st.sidebar.button("🔒 Sair")
    if logout:
        st.session_state["autenticado"] = False
        st.rerun()

    # Selecionar posição
    posicao = st.sidebar.selectbox("Selecione a posição:", ["Atacantes", "Meio-campistas", "Defensores", "Goleiros"], index=0)
    if posicao == "Goleiros":
        st.title("Dashboard de Goleiros")
        st.markdown("---")
        df = carregar_dados(sheet_name='GK')
    elif posicao == "Defensores":
        st.title("Dashboard de Defensores")
        st.markdown("---")
        df = carregar_dados(sheet_name='DF')
    elif posicao == "Meio-campistas":
        st.title("Dashboard de Meio-Campistas")
        st.markdown("---")
        df = carregar_dados(sheet_name='MF')
    else:
        st.title("Dashboard de Atacantes")
        st.markdown("---")
        df = carregar_dados(sheet_name='FW')

    # Remover colunas fixas que não entram no treino
    colunas_nao_usadas = [
        "Player ID", "Player", "Nation", "Position", "Age at season start", 
        "Year of birth", "Squad", "Minutes Played", "Minutes Played/90", "Completed Matches"
    ]
    X = df.copy()
    X.drop(columns=colunas_nao_usadas, inplace=True, errors='ignore')

    # Primeira remoção pelo usuário
    colunas_removiveis = [col for col in X.columns]
    colunas_excluir = st.multiselect("Selecione colunas que deseja excluir da análise:", colunas_removiveis)
    X.drop(columns=colunas_excluir, inplace=True, errors='ignore')

    # Matriz de correlação
    st.subheader("Matriz de Correlação")
    matriz = calcular_matriz_correlacao(X)
    fig_corr = px.imshow(matriz, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

    # Listar pares correlacionados (|corr| >= 0.7)
    limite_corr = 0.7
    upper = matriz.abs().where(np.triu(np.ones(matriz.shape), k=1).astype(bool))
    pares_correlacionados = [
        (col, row, upper.loc[row, col]) for col in upper.columns for row in upper.index if upper.loc[row, col] >= limite_corr
    ]
    if pares_correlacionados:
        st.markdown(f"**Variáveis altamente correlacionadas (|corr| >= {limite_corr}):**")
        for var1, var2, corr in sorted(pares_correlacionados, key=lambda x: -x[2]):
            st.write(f"- {var1} ↔ {var2} = {corr:.2f}")
    else:
        st.write("Nenhuma variável altamente correlacionada encontrada.")

    # Permitir nova exclusão com base na correlação
    novas_colunas_excluir = st.multiselect(
        "Selecione colunas adicionais para remover com base na correlação alta:",
        options=X.columns,
        default=[]
    )
    X.drop(columns=novas_colunas_excluir, inplace=True, errors='ignore')

    # Escalar e preparar dados
    scaler = MinMaxScaler()
    X['Market Value (in millions of euros)'] = scaler.fit_transform(X[['Market Value (in millions of euros)']])

    # Encontrar melhor DBSCAN pelo score final
    resultados = []
    for eps, min_sample in product(np.arange(0.5, 3.6, 0.1), np.arange(5, 11)):
        clusterer = DBSCAN(eps=eps, min_samples=min_sample, n_jobs=-1).fit(X)
        labels = clusterer.labels_
        sil_score, ruido = avaliar_clustering(X, labels)
        if sil_score != -1:
            resultados.append({'eps': eps, 'min_samples': min_sample, 'silhouette': sil_score, 'ruido': ruido})

    if not resultados:
        st.warning("Nenhum modelo DBSCAN válido encontrado com os parâmetros testados.")
        return

    df_resultados = pd.DataFrame(resultados)

    # Normalizar scores para combinar
    df_resultados[['silhouette_norm', 'ruido_norm']] = scaler.fit_transform(df_resultados[['silhouette', 'ruido']])
    df_resultados['score_final'] = 0.7 * df_resultados['silhouette_norm'] - 0.3 * df_resultados['ruido_norm']

    melhor = df_resultados.loc[df_resultados['score_final'].idxmax()]
    st.markdown(f"**Melhor modelo DBSCAN:** eps={melhor['eps']:.2f}, min_samples={int(melhor['min_samples'])}, score_final={melhor['score_final']:.4f}")

    # Refit melhor modelo
    clusterer = DBSCAN(eps=melhor['eps'], min_samples=int(melhor['min_samples']), n_jobs=-1).fit(X)
    labels = clusterer.labels_

    df['Cluster'] = labels

    # PCA 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_plot['Cluster'] = pd.Series(labels.astype(str)).replace({'-1': 'Outlier'})

    fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', color='Cluster', title="PCA 3D dos Atacantes")
    st.plotly_chart(fig, use_container_width=True)

    # Performance customizada
    st.subheader("Customização da Performance")
    if posicao == 'Atacantes':
        pesos = {
            'Goals/90': st.number_input("Peso: Goals/90", value=5.0),
            'Assists/90': st.number_input("Peso: Assists/90", value=4.0),
            'Shots on Target/90': st.number_input("Peso: Shots on Target/90", value=2.0),
            'Goal-Creating Actions/90': st.number_input("Peso: Goal-Creating Actions/90", value=3.0),
            'Penalty Goals': st.number_input("Peso: Penalty Goals", value=2.5),
            'Penalties Missed': st.number_input("Peso: Penalties Missed", value=-2.5),
            'Penalties Won/90': st.number_input("Peso: Penalties Won/90", value=2.5),
            'Expected Goals/90': st.number_input("Peso: Expected Goals/90", value=2.5),
            'Expected Assists/90': st.number_input("Peso: Expected Assists/90", value=2.0),
            'Shots off Target/90': st.number_input("Peso: Shots off Target/90", value=1.0),
            'Missed Passes/90': st.number_input("Peso: Missed Passes/90", value=-1.0),
            'Progressive Carries/90': st.number_input("Peso: Progressive Carries/90", value=1.0),
            'Progressive Passes/90': st.number_input("Peso: Progressive Passes/90", value=1.0),
            'Progressive Passes Received/90': st.number_input("Peso: Progressive Passes Received/90", value=1.0),
            'Fouls Drawn/90': st.number_input("Peso: Fouls Drawn/90", value=1.0),
            'Aerials Won/90': st.number_input("Peso: Aerials Won/90", value=1.5),
            'Aerials Lost/90': st.number_input("Peso: Aerials Lost/90", value=-1.5),
            'Succesfull Take-Ons/90': st.number_input("Peso: Succesfull Take-Ons/90", value=1.5),
            'Unsuccesfull Take-Ons/90': st.number_input("Peso: Unsuccesfull Take-Ons/90", value=-1.5),
            'Offsides/90': st.number_input("Peso: Offsides/90", value=-1.0),
            'Through Balls/90': st.number_input("Peso: Through Balls/90", value=1.5),
            'Yellow Cards/90': st.number_input("Peso: Yellow Cards/90", value=-1.0),
            'Red Cards/90': st.number_input("Peso: Red Cards/90", value=-3.0),
            'Fouls Commited/90': st.number_input("Peso: Fouls Commited/90", value=-1.0),
            'Miscontrols/90': st.number_input("Peso: Miscontrols/90", value=-1.0),
            'Dispossessed/90': st.number_input("Peso: Dispossessed/90", value=-1.0),
            'Tackles/90': st.number_input("Peso: Tackles/90", value=0.0),
            'Challenges Lost/90': st.number_input("Peso: Challenges Lost/90", value=0.0),
            'Blocks/90': st.number_input("Peso: Blocks/90", value=0.0),
            'Interceptions/90': st.number_input("Peso: Interceptions/90", value=0.0),
            'Clearances/90': st.number_input("Peso: Clearances/90", value=0.0),
            'Errors/90': st.number_input("Peso: Errors/90", value=0.0),
            'Own Goals/90': st.number_input("Peso: Own Goals/90", value=0.0),
            'Ball Recoveries/90': st.number_input("Peso: Ball Recoveries/90", value=0.0),
            'Penalties Conceded/90': st.number_input("Peso: Penalties Conceded/90", value=0.0),
        }
    elif posicao == 'Meio-campistas':
        pesos = {
            'Goals/90': st.number_input("Peso: Goals/90", value=3.0),
            'Assists/90': st.number_input("Peso: Assists/90", value=3.0),
            'Shots on Target/90': st.number_input("Peso: Shots on Target/90", value=1.0),
            'Shots off Target/90': st.number_input("Peso: Shots off Target/90", value=-1.0),
            'Goal-Creating Actions/90': st.number_input("Peso: Goal-Creating Actions/90", value=2.0),
            'Penalty Goals': st.number_input("Peso: Penalty Goals", value=1.5),
            'Penalties Missed': st.number_input("Peso: Penalties Missed", value=-2.5),
            'Penalties Won/90': st.number_input("Peso: Penalties Won/90", value=2.0),
            'Expected Goals/90': st.number_input("Peso: Expected Goals/90", value=1.5),
            'Expected Assists/90': st.number_input("Peso: Expected Assists/90", value=2.0),
            'Missed Passes/90': st.number_input("Peso: Missed Passes/90", value=-1.0),
            'Progressive Carries/90': st.number_input("Peso: Progressive Carries/90", value=1.5),
            'Progressive Passes/90': st.number_input("Peso: Progressive Passes/90", value=1.5),
            'Progressive Passes Received/90': st.number_input("Peso: Progressive Passes Received/90", value=1.0),
            'Fouls Drawn/90': st.number_input("Peso: Fouls Drawn/90", value=1.0),
            'Succesfull Take-Ons/90': st.number_input("Peso: Succesfull Take-Ons/90", value=1.5),
            'Unsuccesfull Take-Ons/90': st.number_input("Peso: Unsuccesfull Take-Ons/90", value=-1.5),
            'Through Balls/90': st.number_input("Peso: Through Balls/90", value=1.5),
            'Yellow Cards/90': st.number_input("Peso: Yellow Cards/90", value=-1.0),
            'Red Cards/90': st.number_input("Peso: Red Cards/90", value=-3.0),
            'Fouls Commited/90': st.number_input("Peso: Fouls Commited/90", value=-1.0),
            'Tackles/90': st.number_input("Peso: Tackles/90", value=1.5),
            'Challenges Lost/90': st.number_input("Peso: Challenges Lost/90", value=-1.5),
            'Blocks/90': st.number_input("Peso: Blocks/90", value=1.5),
            'Interceptions/90': st.number_input("Peso: Interceptions/90", value=1.5),
            'Clearances/90': st.number_input("Peso: Clearances/90", value=0.5),
            'Miscontrols/90': st.number_input("Peso: Miscontrols/90", value=-1.0),
            'Dispossessed/90': st.number_input("Peso: Dispossessed/90", value=-1.5),
            'Errors/90': st.number_input("Peso: Errors/90", value=-2.0),
            'Own Goals/90': st.number_input("Peso: Own Goals/90", value=-3.0),
            'Ball Recoveries/90': st.number_input("Peso: Ball Recoveries/90", value=1.5),
            'Penalties Conceded/90': st.number_input("Peso: Penalties Conceded/90", value=-2.5),
            'Offsides/90': st.number_input("Peso: Offsides/90", value=0.0),
            'Aerials Won/90': st.number_input("Peso: Aerials Won/90", value=0.0),
            'Aerials Lost/90': st.number_input("Peso: Aerials Lost/90", value=0.0),
        }
    elif posicao == 'Defensores':
        pesos = {
            'Goals/90': st.number_input("Peso: Goals/90", value=1.5),
            'Assists/90': st.number_input("Peso: Assists/90", value=2.0),
            'Shots on Target/90': st.number_input("Peso: Shots on Target/90", value=0.5),
            'Goal-Creating Actions/90': st.number_input("Peso: Goal-Creating Actions/90", value=1.0),
            'Penalties Won/90': st.number_input("Peso: Penalties Won/90", value=0.5),
            'Penalty Goals': st.number_input("Peso: Penalty Goals", value=0.0),
            'Penalties Missed': st.number_input("Peso: Penalties Missed", value=0.0),
            'Expected Goals/90': st.number_input("Peso: Expected Goals/90", value=0.5),
            'Expected Assists/90': st.number_input("Peso: Expected Assists/90", value=0.75),
            'Shots off Target/90': st.number_input("Peso: Shots off Target/90", value=0.25),
            'Missed Passes/90': st.number_input("Peso: Missed Passes/90", value=-1.0),
            'Progressive Carries/90': st.number_input("Peso: Progressive Carries/90", value=1.5),
            'Progressive Passes/90': st.number_input("Peso: Progressive Passes/90", value=1.5),
            'Progressive Passes Received/90': st.number_input("Peso: Progressive Passes Received/90", value=1.0),
            'Fouls Drawn/90': st.number_input("Peso: Fouls Drawn/90", value=1.0),
            'Aerials Won/90': st.number_input("Peso: Aerials Won/90", value=2.0),
            'Aerials Lost/90': st.number_input("Peso: Aerials Lost/90", value=-2.0),
            'Succesfull Take-Ons/90': st.number_input("Peso: Succesfull Take-Ons/90", value=1.0),
            'Unsuccesfull Take-Ons/90': st.number_input("Peso: Unsuccesfull Take-Ons/90", value=-1.0),
            'Offsides/90': st.number_input("Peso: Offsides/90", value=-0.25),
            'Through Balls/90': st.number_input("Peso: Through Balls/90", value=1.0),
            'Yellow Cards/90': st.number_input("Peso: Yellow Cards/90", value=-1.0),
            'Red Cards/90': st.number_input("Peso: Red Cards/90", value=-3.0),
            'Fouls Commited/90': st.number_input("Peso: Fouls Commited/90", value=-1.0),
            'Tackles/90': st.number_input("Peso: Tackles/90", value=2.0),
            'Challenges Lost/90': st.number_input("Peso: Challenges Lost/90", value=-2.0),
            'Blocks/90': st.number_input("Peso: Blocks/90", value=2.0),
            'Interceptions/90': st.number_input("Peso: Interceptions/90", value=2.0),
            'Clearances/90': st.number_input("Peso: Clearances/90", value=2.0),
            'Miscontrols/90': st.number_input("Peso: Miscontrols/90", value=-1.0),
            'Dispossessed/90': st.number_input("Peso: Dispossessed/90", value=-1.0),
            'Errors/90': st.number_input("Peso: Errors/90", value=-2.0),
            'Own Goals/90': st.number_input("Peso: Own Goals/90", value=-3.0),
            'Ball Recoveries/90': st.number_input("Peso: Ball Recoveries/90", value=2.0),
            'Penalties Conceded/90': st.number_input("Peso: Penalties Conceded/90", value=-3.0),
        }
    else:
        pesos = {
            'Penalty Goals': st.number_input("Peso: Penalty Goals", value=0.0),
            'Penalties Missed': st.number_input("Peso: Penalties Missed", value=0.0),
            'Progressive Carries/90': st.number_input("Peso: Progressive Carries/90", value=0.0),
            'Progressive Passes/90': st.number_input("Peso: Progressive Passes/90", value=0.0),
            'Goals/90': st.number_input("Peso: Goals/90", value=0.0),
            'Assists/90': st.number_input("Peso: Assists/90", value=0.0),
            'Expected Goals/90': st.number_input("Peso: Expected Goals/90", value=0.0),
            'Progressive Passes Received/90': st.number_input("Peso: Progressive Passes Received/90", value=0.0),
            'Shots on Target/90': st.number_input("Peso: Shots on Target/90", value=0.0),
            'Expected Assists/90': st.number_input("Peso: Expected Assists/90", value=0.0),
            'Through Balls/90': st.number_input("Peso: Through Balls/90", value=0.0),
            'Goal-Creating Actions/90': st.number_input("Peso: Goal-Creating Actions/90", value=0.0),
            'Tackles/90': st.number_input("Peso: Tackles/90", value=0.0),
            'Challenges Lost/90': st.number_input("Peso: Challenges Lost/90", value=0.0),
            'Blocks/90': st.number_input("Peso: Blocks/90", value=0.0),
            'Interceptions/90': st.number_input("Peso: Interceptions/90", value=0.0),
            'Clearances/90': st.number_input("Peso: Clearances/90", value=0.0),
            'Succesfull Take-Ons/90': st.number_input("Peso: Succesfull Take-Ons/90", value=0.0),
            'Unsuccesfull Take-Ons/90': st.number_input("Peso: Unsuccesfull Take-Ons/90", value=0.0),
            'Miscontrols/90': st.number_input("Peso: Miscontrols/90", value=0.0),
            'Dispossessed/90': st.number_input("Peso: Dispossessed/90", value=0.0),
            'Fouls Commited/90': st.number_input("Peso: Fouls Commited/90", value=0.0),
            'Fouls Drawn/90': st.number_input("Peso: Fouls Drawn/90", value=0.0),
            'Offsides/90': st.number_input("Peso: Offsides/90", value=0.0),
            'Penalties Won/90': st.number_input("Peso: Penalties Won/90", value=0.0),
            'Ball Recoveries/90': st.number_input("Peso: Ball Recoveries/90", value=0.0),
            'Aerials Won/90': st.number_input("Peso: Aerials Won/90", value=0.0),
            'Aerials Lost/90': st.number_input("Peso: Aerials Lost/90", value=0.0),
            'Shots off Target/90': st.number_input("Peso: Shots off Target/90", value=0.0),
            'Yellow Cards/90': st.number_input("Peso: Yellow Cards/90", value=-0.5),
            'Red Cards/90': st.number_input("Peso: Red Cards/90", value=-2.0),
            'Missed Passes/90': st.number_input("Peso: Missed Passes/90", value=-0.5),
            'Errors/90': st.number_input("Peso: Errors/90", value=-1.0),
            'Penalties Conceded/90': st.number_input("Peso: Penalties Conceded/90", value=-2.0),
            'Own Goals/90': st.number_input("Peso: Own Goals/90", value=-2.0),
            'Goals Against/90': st.number_input("Peso: Goals Against/90", value=-3.0),
            'Saves/90': st.number_input("Peso: Saves/90", value=5.0),
            'Clean Sheets/Complete Matches Played': st.number_input("Peso: Clean Sheets/Complete Matches Played", value=4.0),
            'Non-Clean Sheets/Complete Matches Played': st.number_input("Peso: Non-Clean Sheets/Complete Matches Played", value=-1.5),
            'Penalties Allowed': st.number_input("Peso: Penalties Allowed", value=-1.5),
            'Penalties Saved': st.number_input("Peso: Penalties Saved", value=5.0),
            'Post-Shot Expected Goals/90': st.number_input("Peso: Post-Shot Expected Goals/90", value=-1.0),
            'Not Stopped Crosses/90': st.number_input("Peso: Not Stopped Crosses/90", value=-0.5),
            'Crosses Stopped/90': st.number_input("Peso: Crosses Stopped/90", value=0.5),
            'Deffensive Actions Outside Penalty Area/90': st.number_input("Peso: Deffensive Actions Outside Penalty Area/90", value=0.5),
        }

    df['performance'] = sum(df[k] * v for k, v in pesos.items() if k in df.columns)

    # Escalar performance e market value
    df['market_value_scaled'] = scaler.fit_transform(df[['Market Value (in millions of euros)']])
    df['performance_scaled'] = scaler.fit_transform(df[['performance']])
    df['custo_beneficio'] = df['performance_scaled'] / (df['market_value_scaled'] + 1)

    cluster_labels = pd.Series(labels, name="cluster")

    # Ignorar outliers (-1)
    df_valid = df[df["Cluster"] != -1]

    # Resumo por cluster
    resumo = df_valid.groupby("Cluster").agg(
        performance_média=("performance_scaled", "mean"),
        market_value_médio=("market_value_scaled", "mean"),
        performance_variância=("performance_scaled", "std"),
        market_value_variância=("market_value_scaled", "std"),
        n_jogadores=('Player', 'count')
    ).fillna(0)

    # Cálculo do novo custo-benefício
    resumo['custo_beneficio'] = (
        resumo['performance_média'] /
        (
            resumo['market_value_médio'] *
            (1 + resumo['performance_variância']) *
            (1 + resumo['market_value_variância'])
        )
    )

    # Selecionar cluster com MAX custo-benefício
    cluster_bom = resumo['custo_beneficio'].idxmax()

    st.subheader("Resumo dos Clusters")
    # Reorganizar as colunas
    resumo = resumo[[
        'performance_média',
        'performance_variância',
        'market_value_médio',
        'n_jogadores',
        'custo_beneficio'
    ]]

    resumo = resumo.sort_values(by='custo_beneficio', ascending=False)
    st.write(resumo)

    # Tabelas top jogadores
    st.subheader("Top Jogadores")
    top_n = st.slider("Quantidade a exibir:", 5, 50, 20)

    st.markdown("#### 1. Mais próximos do cluster de bons e baratos")

    centroide = X[cluster_labels == cluster_bom].mean(axis=0)
    distancias = X.apply(lambda row: distance.euclidean(centroide, row), axis=1)
    idx_outliers = np.where(cluster_labels == -1)[0]
    df_outliers = df.iloc[idx_outliers].copy()
    df_outliers['distancia'] = distancias
    df_outliers['distancia_scaled'] = scaler.fit_transform(df_outliers[['distancia']])
    df_outliers['score_hibrido'] = df_outliers['custo_beneficio'] / (df_outliers['distancia_scaled'] + 1)

    top_cluster = df_outliers.nsmallest(top_n, 'distancia')
    st.table(top_cluster[['Player', 'performance', 'Market Value (in millions of euros)', 'custo_beneficio', 'distancia']].reset_index(drop=True))

    st.markdown("#### 2. Maior custo-benefício individual")
    top_cb = df.sort_values(by='custo_beneficio', ascending=False).head(top_n)
    st.table(top_cb[['Player', 'custo_beneficio', 'Cluster']].reset_index(drop=True))

    st.markdown("#### 3. Maior score híbrido")
    top_hibrido = df_outliers.sort_values(by='score_hibrido', ascending=False).head(top_n)
    st.table(top_hibrido[['Player', 'performance', 'Market Value (in millions of euros)', 'custo_beneficio', 'distancia', 'score_hibrido']].reset_index(drop=True))

    st.subheader("Ver detalhes de um jogador")
    jogador = st.selectbox("Selecione o jogador:", df['Player'].unique())
    st.write(df[df['Player'] == jogador].T)

if st.session_state["autenticado"]:
    app()
else:
    login()