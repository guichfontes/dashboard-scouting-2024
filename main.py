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

# Usernames and passwords
USERS = {"test": "password"}

# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    st.title("Login - Scouting Dashboard")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Enter"):
        if username in USERS and USERS[username] == password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect username or password.")

@st.cache_data
def load_data(sheet_name):
    df = pd.read_excel("data/data_players_br_2024.xlsx", sheet_name=sheet_name)
    return df.copy()

@st.cache_data
def calculate_correlation_matrix(df):
    return df.corr()

def evaluate_clustering(X, labels):
    mask = labels != -1
    n_noise = np.sum(~mask)
    noise_proportion = n_noise / len(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return -1, noise_proportion
    sil_score = silhouette_score(X[mask], labels[mask])
    return sil_score, noise_proportion

def app():
    st.set_page_config(page_title="Low-cost Scouting", layout="wide")
    st.title("Low-cost Scouting - Brazilian Championship 2024")

    # Logout button
    logout = st.sidebar.button("🔒 Logout")
    if logout:
        st.session_state["authenticated"] = False
        st.rerun()

    # Select position
    position = st.sidebar.selectbox("Select position:", ["Forwards", "Midfielders", "Defenders", "Goalkeepers"], index=0)
    if position == "Goalkeepers":
        st.title("Goalkeepers Dashboard")
        st.markdown("---")
        df = load_data(sheet_name='GK')
    elif position == "Defenders":
        st.title("Defenders Dashboard")
        st.markdown("---")
        df = load_data(sheet_name='DF')
    elif position == "Midfielders":
        st.title("Midfielders Dashboard")
        st.markdown("---")
        df = load_data(sheet_name='MF')
    else:
        st.title("Forwards Dashboard")
        st.markdown("---")
        df = load_data(sheet_name='FW')

    # Remove fixed columns that are not used in training
    unused_columns = [
        "Player ID", "Player", "Nation", "Position", "Age at season start", 
        "Year of birth", "Squad", "Minutes Played", "Minutes Played/90", "Completed Matches"
    ]
    X = df.copy()
    X.drop(columns=unused_columns, inplace=True, errors='ignore')

    # First removal by the user
    removable_columns = [col for col in X.columns]
    columns_to_exclude = st.multiselect("Select columns you want to exclude from the analysis:", removable_columns)
    X.drop(columns=columns_to_exclude, inplace=True, errors='ignore')

    # Correlation matrix
    st.subheader("Correlation Matrix")
    matrix = calculate_correlation_matrix(X)
    fig_corr = px.imshow(matrix, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

    # List correlated pairs (|corr| >= 0.7)
    corr_limit = 0.7
    upper = matrix.abs().where(np.triu(np.ones(matrix.shape), k=1).astype(bool))
    correlated_pairs = [
        (col, row, upper.loc[row, col]) for col in upper.columns for row in upper.index if upper.loc[row, col] >= corr_limit
    ]
    if correlated_pairs:
        st.markdown(f"**Highly correlated variables (|corr| >= {corr_limit}):**")
        for var1, var2, corr in sorted(correlated_pairs, key=lambda x: -x[2]):
            st.write(f"- {var1} ↔ {var2} = {corr:.2f}")
    else:
        st.write("No highly correlated variables found.")

    # Allow new exclusion based on correlation
    new_columns_to_exclude = st.multiselect(
        "Select additional columns to remove based on high correlation:",
        options=X.columns,
        default=[]
    )
    X.drop(columns=new_columns_to_exclude, inplace=True, errors='ignore')

    # Scale and prepare data
    scaler = MinMaxScaler()
    X['Market Value (in millions of euros)'] = scaler.fit_transform(X[['Market Value (in millions of euros)']])

    # Find best DBSCAN by final score
    results = []
    for eps, min_sample in product(np.arange(0.5, 3.6, 0.1), np.arange(5, 11)):
        clusterer = DBSCAN(eps=eps, min_samples=min_sample, n_jobs=-1).fit(X)
        labels = clusterer.labels_
        sil_score, noise = evaluate_clustering(X, labels)
        if sil_score != -1:
            results.append({'eps': eps, 'min_samples': min_sample, 'silhouette': sil_score, 'noise': noise})

    if not results:
        st.warning("No valid DBSCAN model found with the tested parameters.")
        return

    df_results = pd.DataFrame(results)

    # Normalize scores to combine
    df_results[['silhouette_norm', 'noise_norm']] = scaler.fit_transform(df_results[['silhouette', 'noise']])
    df_results['final_score'] = 0.7 * df_results['silhouette_norm'] - 0.3 * df_results['noise_norm']

    best = df_results.loc[df_results['final_score'].idxmax()]
    st.markdown(f"**Best DBSCAN model:** eps={best['eps']:.2f}, min_samples={int(best['min_samples'])}, final_score={best['final_score']:.4f}")

    # Refit best model
    clusterer = DBSCAN(eps=best['eps'], min_samples=int(best['min_samples']), n_jobs=-1).fit(X)
    labels = clusterer.labels_

    df['Cluster'] = labels

    # PCA 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    df_plot = pd.DataFrame(X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_plot['Cluster'] = pd.Series(labels.astype(str)).replace({'-1': 'Outlier'})

    fig = px.scatter_3d(df_plot, x='PC1', y='PC2', z='PC3', color='Cluster', title="PCA 3D of Forwards")
    st.plotly_chart(fig, use_container_width=True)

    # Custom performance
    st.subheader("Performance Customization")
    if position == 'Forwards':
        pesos = {
            'Goals/90': st.number_input("Weight: Goals/90", value=5.0),
            'Assists/90': st.number_input("Weight: Assists/90", value=4.0),
            'Shots on Target/90': st.number_input("Weight: Shots on Target/90", value=2.0),
            'Goal-Creating Actions/90': st.number_input("Weight: Goal-Creating Actions/90", value=3.0),
            'Penalty Goals': st.number_input("Weight: Penalty Goals", value=2.5),
            'Penalties Missed': st.number_input("Weight: Penalties Missed", value=-2.5),
            'Penalties Won/90': st.number_input("Weight: Penalties Won/90", value=2.5),
            'Expected Goals/90': st.number_input("Weight: Expected Goals/90", value=2.5),
            'Expected Assists/90': st.number_input("Weight: Expected Assists/90", value=2.0),
            'Shots off Target/90': st.number_input("Weight: Shots off Target/90", value=1.0),
            'Missed Passes/90': st.number_input("Weight: Missed Passes/90", value=-1.0),
            'Progressive Carries/90': st.number_input("Weight: Progressive Carries/90", value=1.0),
            'Progressive Passes/90': st.number_input("Weight: Progressive Passes/90", value=1.0),
            'Progressive Passes Received/90': st.number_input("Weight: Progressive Passes Received/90", value=1.0),
            'Fouls Drawn/90': st.number_input("Weight: Fouls Drawn/90", value=1.0),
            'Aerials Won/90': st.number_input("Weight: Aerials Won/90", value=1.5),
            'Aerials Lost/90': st.number_input("Weight: Aerials Lost/90", value=-1.5),
            'Succesfull Take-Ons/90': st.number_input("Weight: Succesfull Take-Ons/90", value=1.5),
            'Unsuccesfull Take-Ons/90': st.number_input("Weight: Unsuccesfull Take-Ons/90", value=-1.5),
            'Offsides/90': st.number_input("Weight: Offsides/90", value=-1.0),
            'Through Balls/90': st.number_input("Weight: Through Balls/90", value=1.5),
            'Yellow Cards/90': st.number_input("Weight: Yellow Cards/90", value=-1.0),
            'Red Cards/90': st.number_input("Weight: Red Cards/90", value=-3.0),
            'Fouls Commited/90': st.number_input("Weight: Fouls Commited/90", value=-1.0),
            'Miscontrols/90': st.number_input("Weight: Miscontrols/90", value=-1.0),
            'Dispossessed/90': st.number_input("Weight: Dispossessed/90", value=-1.0),
            'Tackles/90': st.number_input("Weight: Tackles/90", value=0.0),
            'Challenges Lost/90': st.number_input("Weight: Challenges Lost/90", value=0.0),
            'Blocks/90': st.number_input("Weight: Blocks/90", value=0.0),
            'Interceptions/90': st.number_input("Weight: Interceptions/90", value=0.0),
            'Clearances/90': st.number_input("Weight: Clearances/90", value=0.0),
            'Errors/90': st.number_input("Weight: Errors/90", value=0.0),
            'Own Goals/90': st.number_input("Weight: Own Goals/90", value=0.0),
            'Ball Recoveries/90': st.number_input("Weight: Ball Recoveries/90", value=0.0),
            'Penalties Conceded/90': st.number_input("Weight: Penalties Conceded/90", value=0.0),
        }
    elif position == 'Meio-campistas':
        pesos = {
            'Goals/90': st.number_input("Weight: Goals/90", value=3.0),
            'Assists/90': st.number_input("Weight: Assists/90", value=3.0),
            'Shots on Target/90': st.number_input("Weight: Shots on Target/90", value=1.0),
            'Shots off Target/90': st.number_input("Weight: Shots off Target/90", value=-1.0),
            'Goal-Creating Actions/90': st.number_input("Weight: Goal-Creating Actions/90", value=2.0),
            'Penalty Goals': st.number_input("Weight: Penalty Goals", value=1.5),
            'Penalties Missed': st.number_input("Weight: Penalties Missed", value=-2.5),
            'Penalties Won/90': st.number_input("Weight: Penalties Won/90", value=2.0),
            'Expected Goals/90': st.number_input("Weight: Expected Goals/90", value=1.5),
            'Expected Assists/90': st.number_input("Weight: Expected Assists/90", value=2.0),
            'Missed Passes/90': st.number_input("Weight: Missed Passes/90", value=-1.0),
            'Progressive Carries/90': st.number_input("Weight: Progressive Carries/90", value=1.5),
            'Progressive Passes/90': st.number_input("Weight: Progressive Passes/90", value=1.5),
            'Progressive Passes Received/90': st.number_input("Weight: Progressive Passes Received/90", value=1.0),
            'Fouls Drawn/90': st.number_input("Weight: Fouls Drawn/90", value=1.0),
            'Succesfull Take-Ons/90': st.number_input("Weight: Succesfull Take-Ons/90", value=1.5),
            'Unsuccesfull Take-Ons/90': st.number_input("Weight: Unsuccesfull Take-Ons/90", value=-1.5),
            'Through Balls/90': st.number_input("Weight: Through Balls/90", value=1.5),
            'Yellow Cards/90': st.number_input("Weight: Yellow Cards/90", value=-1.0),
            'Red Cards/90': st.number_input("Weight: Red Cards/90", value=-3.0),
            'Fouls Commited/90': st.number_input("Weight: Fouls Commited/90", value=-1.0),
            'Tackles/90': st.number_input("Weight: Tackles/90", value=1.5),
            'Challenges Lost/90': st.number_input("Weight: Challenges Lost/90", value=-1.5),
            'Blocks/90': st.number_input("Weight: Blocks/90", value=1.5),
            'Interceptions/90': st.number_input("Weight: Interceptions/90", value=1.5),
            'Clearances/90': st.number_input("Weight: Clearances/90", value=0.5),
            'Miscontrols/90': st.number_input("Weight: Miscontrols/90", value=-1.0),
            'Dispossessed/90': st.number_input("Weight: Dispossessed/90", value=-1.5),
            'Errors/90': st.number_input("Weight: Errors/90", value=-2.0),
            'Own Goals/90': st.number_input("Weight: Own Goals/90", value=-3.0),
            'Ball Recoveries/90': st.number_input("Weight: Ball Recoveries/90", value=1.5),
            'Penalties Conceded/90': st.number_input("Weight: Penalties Conceded/90", value=-2.5),
            'Offsides/90': st.number_input("Weight: Offsides/90", value=0.0),
            'Aerials Won/90': st.number_input("Weight: Aerials Won/90", value=0.0),
            'Aerials Lost/90': st.number_input("Weight: Aerials Lost/90", value=0.0),
        }
    elif position == 'Defensores':
        pesos = {
            'Goals/90': st.number_input("Weight: Goals/90", value=1.5),
            'Assists/90': st.number_input("Weight: Assists/90", value=2.0),
            'Shots on Target/90': st.number_input("Weight: Shots on Target/90", value=0.5),
            'Goal-Creating Actions/90': st.number_input("Weight: Goal-Creating Actions/90", value=1.0),
            'Penalties Won/90': st.number_input("Weight: Penalties Won/90", value=0.5),
            'Penalty Goals': st.number_input("Weight: Penalty Goals", value=0.0),
            'Penalties Missed': st.number_input("Weight: Penalties Missed", value=0.0),
            'Expected Goals/90': st.number_input("Weight: Expected Goals/90", value=0.5),
            'Expected Assists/90': st.number_input("Weight: Expected Assists/90", value=0.75),
            'Shots off Target/90': st.number_input("Weight: Shots off Target/90", value=0.25),
            'Missed Passes/90': st.number_input("Weight: Missed Passes/90", value=-1.0),
            'Progressive Carries/90': st.number_input("Weight: Progressive Carries/90", value=1.5),
            'Progressive Passes/90': st.number_input("Weight: Progressive Passes/90", value=1.5),
            'Progressive Passes Received/90': st.number_input("Weight: Progressive Passes Received/90", value=1.0),
            'Fouls Drawn/90': st.number_input("Weight: Fouls Drawn/90", value=1.0),
            'Aerials Won/90': st.number_input("Weight: Aerials Won/90", value=2.0),
            'Aerials Lost/90': st.number_input("Weight: Aerials Lost/90", value=-2.0),
            'Succesfull Take-Ons/90': st.number_input("Weight: Succesfull Take-Ons/90", value=1.0),
            'Unsuccesfull Take-Ons/90': st.number_input("Weight: Unsuccesfull Take-Ons/90", value=-1.0),
            'Offsides/90': st.number_input("Weight: Offsides/90", value=-0.25),
            'Through Balls/90': st.number_input("Weight: Through Balls/90", value=1.0),
            'Yellow Cards/90': st.number_input("Weight: Yellow Cards/90", value=-1.0),
            'Red Cards/90': st.number_input("Weight: Red Cards/90", value=-3.0),
            'Fouls Commited/90': st.number_input("Weight: Fouls Commited/90", value=-1.0),
            'Tackles/90': st.number_input("Weight: Tackles/90", value=2.0),
            'Challenges Lost/90': st.number_input("Weight: Challenges Lost/90", value=-2.0),
            'Blocks/90': st.number_input("Weight: Blocks/90", value=2.0),
            'Interceptions/90': st.number_input("Weight: Interceptions/90", value=2.0),
            'Clearances/90': st.number_input("Weight: Clearances/90", value=2.0),
            'Miscontrols/90': st.number_input("Weight: Miscontrols/90", value=-1.0),
            'Dispossessed/90': st.number_input("Weight: Dispossessed/90", value=-1.0),
            'Errors/90': st.number_input("Weight: Errors/90", value=-2.0),
            'Own Goals/90': st.number_input("Weight: Own Goals/90", value=-3.0),
            'Ball Recoveries/90': st.number_input("Weight: Ball Recoveries/90", value=2.0),
            'Penalties Conceded/90': st.number_input("Weight: Penalties Conceded/90", value=-3.0),
        }
    else:
        pesos = {
            'Penalty Goals': st.number_input("Weight: Penalty Goals", value=0.0),
            'Penalties Missed': st.number_input("Weight: Penalties Missed", value=0.0),
            'Progressive Carries/90': st.number_input("Weight: Progressive Carries/90", value=0.0),
            'Progressive Passes/90': st.number_input("Weight: Progressive Passes/90", value=0.0),
            'Goals/90': st.number_input("Weight: Goals/90", value=0.0),
            'Assists/90': st.number_input("Weight: Assists/90", value=0.0),
            'Expected Goals/90': st.number_input("Weight: Expected Goals/90", value=0.0),
            'Progressive Passes Received/90': st.number_input("Weight: Progressive Passes Received/90", value=0.0),
            'Shots on Target/90': st.number_input("Weight: Shots on Target/90", value=0.0),
            'Expected Assists/90': st.number_input("Weight: Expected Assists/90", value=0.0),
            'Through Balls/90': st.number_input("Weight: Through Balls/90", value=0.0),
            'Goal-Creating Actions/90': st.number_input("Weight: Goal-Creating Actions/90", value=0.0),
            'Tackles/90': st.number_input("Weight: Tackles/90", value=0.0),
            'Challenges Lost/90': st.number_input("Weight: Challenges Lost/90", value=0.0),
            'Blocks/90': st.number_input("Weight: Blocks/90", value=0.0),
            'Interceptions/90': st.number_input("Weight: Interceptions/90", value=0.0),
            'Clearances/90': st.number_input("Weight: Clearances/90", value=0.0),
            'Succesfull Take-Ons/90': st.number_input("Weight: Succesfull Take-Ons/90", value=0.0),
            'Unsuccesfull Take-Ons/90': st.number_input("Weight: Unsuccesfull Take-Ons/90", value=0.0),
            'Miscontrols/90': st.number_input("Weight: Miscontrols/90", value=0.0),
            'Dispossessed/90': st.number_input("Weight: Dispossessed/90", value=0.0),
            'Fouls Commited/90': st.number_input("Weight: Fouls Commited/90", value=0.0),
            'Fouls Drawn/90': st.number_input("Weight: Fouls Drawn/90", value=0.0),
            'Offsides/90': st.number_input("Weight: Offsides/90", value=0.0),
            'Penalties Won/90': st.number_input("Weight: Penalties Won/90", value=0.0),
            'Ball Recoveries/90': st.number_input("Weight: Ball Recoveries/90", value=0.0),
            'Aerials Won/90': st.number_input("Weight: Aerials Won/90", value=0.0),
            'Aerials Lost/90': st.number_input("Weight: Aerials Lost/90", value=0.0),
            'Shots off Target/90': st.number_input("Weight: Shots off Target/90", value=0.0),
            'Yellow Cards/90': st.number_input("Weight: Yellow Cards/90", value=-0.5),
            'Red Cards/90': st.number_input("Weight: Red Cards/90", value=-2.0),
            'Missed Passes/90': st.number_input("Weight: Missed Passes/90", value=-0.5),
            'Errors/90': st.number_input("Weight: Errors/90", value=-1.0),
            'Penalties Conceded/90': st.number_input("Weight: Penalties Conceded/90", value=-2.0),
            'Own Goals/90': st.number_input("Weight: Own Goals/90", value=-2.0),
            'Goals Against/90': st.number_input("Weight: Goals Against/90", value=-3.0),
            'Saves/90': st.number_input("Weight: Saves/90", value=5.0),
            'Clean Sheets/Complete Matches Played': st.number_input("Weight: Clean Sheets/Complete Matches Played", value=4.0),
            'Non-Clean Sheets/Complete Matches Played': st.number_input("Weight: Non-Clean Sheets/Complete Matches Played", value=-1.5),
            'Penalties Allowed': st.number_input("Weight: Penalties Allowed", value=-1.5),
            'Penalties Saved': st.number_input("Weight: Penalties Saved", value=5.0),
            'Post-Shot Expected Goals/90': st.number_input("Weight: Post-Shot Expected Goals/90", value=-1.0),
            'Not Stopped Crosses/90': st.number_input("Weight: Not Stopped Crosses/90", value=-0.5),
            'Crosses Stopped/90': st.number_input("Weight: Crosses Stopped/90", value=0.5),
            'Deffensive Actions Outside Penalty Area/90': st.number_input("Weight: Deffensive Actions Outside Penalty Area/90", value=0.5),
        }

    df['performance'] = sum(df[k] * v for k, v in pesos.items() if k in df.columns)

    # Scale performance and market value
    df['market_value_scaled'] = scaler.fit_transform(df[['Market Value (in millions of euros)']])
    df['performance_scaled'] = scaler.fit_transform(df[['performance']])
    df['cost_benefit'] = df['performance_scaled'] / (df['market_value_scaled'] + 1)

    cluster_labels = pd.Series(labels, name="cluster")

    # Ignore outliers (-1)
    df_valid = df[df["Cluster"] != -1]

    # Cluster summary
    summary = df_valid.groupby("Cluster").agg(
        performance_mean=("performance_scaled", "mean"),
        market_value_mean=("market_value_scaled", "mean"),
        performance_variance=("performance_scaled", "std"),
        market_value_variance=("market_value_scaled", "std"),
        n_players=('Player', 'count')
    ).fillna(0)

    # New cost-benefit metric
    summary['cost_benefit'] = (
        summary['performance_mean'] /
        (
            summary['market_value_mean'] *
            (1 + summary['performance_variance']) *
            (1 + summary['market_value_variance'])
        )
    )

    # Select cluster by best cost-benefit
    good_cluster = summary['cost_benefit'].idxmax()

    st.subheader("Cluster Summary")
    # Reorganize columns
    summary = summary[[
        'performance_mean',
        'performance_variance',
        'market_value_mean',
        'n_players',
        'cost_benefit'
    ]]

    summary = summary.sort_values(by='cost_benefit', ascending=False)
    st.write(summary)

    # Top players tables
    st.subheader("Top Players")
    top_n = st.slider("Number to display:", 5, 50, 20)

    st.markdown("#### 1. Closest to the good and cheap cluster")

    centroid = X[cluster_labels == good_cluster].mean(axis=0)
    distances = X.apply(lambda row: distance.euclidean(centroid, row), axis=1)
    idx_outliers = np.where(cluster_labels == -1)[0]
    df_outliers = df.iloc[idx_outliers].copy()
    df_outliers['distance'] = distances
    df_outliers['distance_scaled'] = scaler.fit_transform(df_outliers[['distance']])
    df_outliers['hybrid_score'] = df_outliers['cost_benefit'] / (df_outliers['distance_scaled'] + 1)

    top_cluster = df_outliers.nsmallest(top_n, 'distance')
    st.table(top_cluster[['Player', 'performance', 'Market Value (in millions of euros)', 'cost_benefit', 'distance']].reset_index(drop=True))

    st.markdown("#### 2. Highest individual cost-benefit")
    top_cb = df.sort_values(by='cost_benefit', ascending=False).head(top_n)
    st.table(top_cb[['Player', 'cost_benefit', 'Cluster']].reset_index(drop=True))

    st.markdown("#### 3. Highest hybrid score")
    top_hybrid = df_outliers.sort_values(by='hybrid_score', ascending=False).head(top_n)
    st.table(top_hybrid[['Player', 'performance', 'Market Value (in millions of euros)', 'cost_benefit', 'distance', 'hybrid_score']].reset_index(drop=True))

    st.subheader("See Player Details")
    player = st.selectbox("Select a player:", df['Player'].unique())
    st.write(df[df['Player'] == player].T)

if st.session_state["authenticated"]:
    app()
else:
    login()