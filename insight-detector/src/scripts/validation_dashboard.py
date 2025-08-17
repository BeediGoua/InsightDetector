#!/usr/bin/env python3
# validation_dashboard.py

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Configuration des chemins pour exécution depuis src/scripts/
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Configuration
st.set_page_config(
    page_title="InsightDetector - Validation Dashboard",
    page_icon="",
    layout="wide"
)

@st.cache_data
def load_data():
    """Charge toutes les données nécessaires avec gestion d'erreurs robuste"""
    BASE_DIR = Path(__file__).parent
    RESULTS_DIR = BASE_DIR / "data" / "results"
    
    # Vérification de l'existence du répertoire
    if not RESULTS_DIR.exists():
        st.error(f"Répertoire de données non trouvé: {RESULTS_DIR}")
        st.info("Assurez-vous d'avoir exécuté le notebook d'orchestration d'abord")
        return None, None, None
    
    # Résumés et évaluations
    try:
        data_file = RESULTS_DIR / "all_summaries_and_scores.json"
        if not data_file.exists():
            st.error(f"Fichier de données manquant: {data_file}")
            st.info("Exécutez d'abord la cellule de sauvegarde dans orchestration_notebook.ipynb")
            return None, None, None
            
        with open(data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        summaries = data.get("summaries", [])
        evaluations = data.get("evaluations", [])
        
        if not summaries or not evaluations:
            st.warning("Données vides détectées")
            st.info(f"Résumés: {len(summaries)}, Évaluations: {len(evaluations)}")
            
    except json.JSONDecodeError as e:
        st.error(f"Erreur de lecture JSON: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement: {e}")
        return None, None, None
    
    # Annotations humaines (optionnel)
    annotations_dir = RESULTS_DIR / "human_annotations"
    annotations = []
    if annotations_dir.exists():
        annotation_files = list(annotations_dir.glob("annotations_*.json"))
        if annotation_files:
            try:
                with open(max(annotation_files), "r", encoding="utf-8") as f:
                    annotations = json.load(f)
                st.sidebar.success(f"Annotations humaines chargées: {len(annotations)}")
            except Exception as e:
                st.sidebar.warning(f"Erreur chargement annotations: {e}")
    
    return summaries, evaluations, annotations

def create_metrics_overview(evaluations):
    """Vue d'ensemble des métriques"""
    df = pd.DataFrame(evaluations)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            " Résumés traités",
            len(df),
            delta=None
        )
    
    with col2:
        avg_factuality = df['Factualité'].mean()
        st.metric(
            "Factualité moyenne", 
            f"{avg_factuality:.3f}",
            delta=f"+{(avg_factuality-0.5):.3f}" if avg_factuality > 0.5 else f"{(avg_factuality-0.5):.3f}"
        )
    
    with col3:
        avg_coherence = df['Cohérence'].mean()
        st.metric(
            " Cohérence moyenne",
            f"{avg_coherence:.3f}",
            delta=f"+{(avg_coherence-0.5):.3f}" if avg_coherence > 0.5 else f"{(avg_coherence-0.5):.3f}"
        )
    
    with col4:
        avg_composite = df['Score composite'].mean()
        st.metric(
            "Score global moyen",
            f"{avg_composite:.3f}",
            delta=f"+{(avg_composite-0.5):.3f}" if avg_composite > 0.5 else f"{(avg_composite-0.5):.3f}"
        )

def create_distribution_plots(evaluations):
    """Graphiques de distribution des scores"""
    df = pd.DataFrame(evaluations)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution des scores principaux
        fig = px.histogram(
            df, 
            x=['Factualité', 'Cohérence', 'Lisibilité', 'Score composite'],
            title=" Distribution des Scores Principaux",
            nbins=20
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot comparatif
        metrics_data = []
        for metric in ['Factualité', 'Cohérence', 'Lisibilité', 'Engagement']:
            for score in df[metric]:
                metrics_data.append({
                    'Métrique': metric,
                    'Score': score
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        fig = px.box(
            df_metrics,
            x='Métrique',
            y='Score',
            title=" Comparaison des Métriques (Box Plot)"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def summary_explorer(summaries, evaluations):
    """Explorateur de résumés interactif"""
    st.subheader(" Explorateur de Résumés")
    
    df_eval = pd.DataFrame(evaluations)
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        factuality_range = st.slider(
            "Plage Factualité",
            0.0, 1.0, (0.0, 1.0), 0.1
        )
    
    with col2:
        coherence_range = st.slider(
            "Plage Cohérence", 
            0.0, 1.0, (0.0, 1.0), 0.1
        )
    
    with col3:
        sort_by = st.selectbox(
            "Trier par",
            ['Score composite', 'Factualité', 'Cohérence', 'Lisibilité'],
            index=0
        )
    
    # Filtrage
    mask = (
        (df_eval['Factualité'] >= factuality_range[0]) &
        (df_eval['Factualité'] <= factuality_range[1]) &
        (df_eval['Cohérence'] >= coherence_range[0]) &
        (df_eval['Cohérence'] <= coherence_range[1])
    )
    
    filtered_df = df_eval[mask].sort_values(sort_by, ascending=False)
    
    st.write(f"**{len(filtered_df)} résumés correspondants**")
    
    # Sélection d'un résumé spécifique
    if len(filtered_df) > 0:
        selected_idx = st.selectbox(
            "Sélectionner un résumé",
            range(len(filtered_df)),
            format_func=lambda x: f"#{x+1} - {filtered_df.iloc[x]['summary_id']} (Score: {filtered_df.iloc[x]['Score composite']:.3f})"
        )
        
        # Afficher le résumé sélectionné
        if selected_idx is not None:
            selected_summary_id = filtered_df.iloc[selected_idx]['summary_id']
            summary_data = next(s for s in summaries if s['summary_id'] == selected_summary_id)
            eval_data = filtered_df.iloc[selected_idx]
            
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("###  Résumé Généré")
                st.info(summary_data['ensemble_summary']['summary'])
                
                # Métadonnées
                st.markdown("###  Métadonnées")
                metadata = summary_data.get('metadata', {})
                st.json({
                    'Longueur résumé': summary_data['ensemble_summary']['length'],
                    'Longueur source': metadata.get('source_length', 'N/A'),
                    'Temps génération': f"{metadata.get('runtime_seconds', 'N/A')}s",
                    'Stratégie': metadata.get('fusion_strategy', 'N/A')
                })
            
            with col2:
                st.markdown("###  Scores Détaillés")
                
                # Graphique radar des scores
                scores = {
                    'Factualité': eval_data['Factualité'],
                    'Cohérence': eval_data['Cohérence'], 
                    'Lisibilité': eval_data['Lisibilité'],
                    'Engagement': eval_data['Engagement']
                }
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=list(scores.values()),
                    theta=list(scores.keys()),
                    fill='toself',
                    name='Scores'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Scores numériques
                for metric, score in scores.items():
                    st.metric(metric, f"{score:.3f}")

def human_vs_automatic_comparison(annotations, evaluations):
    """Comparaison évaluations humaines vs automatiques"""
    if not annotations:
        st.info(" Aucune annotation humaine disponible pour comparaison")
        return
    
    st.subheader("👥 Comparaison Humain vs Automatique")
    
    # Préparer les données
    human_data = []
    for ann in annotations:
        summary_id = ann['summary_id']
        eval_data = next((e for e in evaluations if e['summary_id'] == summary_id), None)
        if eval_data:
            human_data.append({
                'summary_id': summary_id,
                'human_factuality': ann['human_scores']['factuality'] / 5,  # Normaliser 1-5 -> 0-1
                'human_coherence': ann['human_scores']['coherence'] / 5,
                'human_overall': ann['human_scores']['overall'] / 5,
                'auto_factuality': eval_data['Factualité'],
                'auto_coherence': eval_data['Cohérence'],
                'auto_composite': eval_data['Score composite']
            })
    
    if not human_data:
        st.warning(" Aucune correspondance trouvée entre annotations humaines et automatiques")
        return
    
    df_comparison = pd.DataFrame(human_data)
    
    # Graphiques de corrélation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.scatter(
            df_comparison,
            x='auto_factuality',
            y='human_factuality',
            title="Factualité: Humain vs Auto",
            trendline="ols"
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="red"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df_comparison,
            x='auto_coherence',
            y='human_coherence', 
            title="Cohérence: Humain vs Auto",
            trendline="ols"
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="red"))
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.scatter(
            df_comparison,
            x='auto_composite',
            y='human_overall',
            title="Score Global: Humain vs Auto", 
            trendline="ols"
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="red"))
        st.plotly_chart(fig, use_container_width=True)
    
    # Calcul des corrélations
    st.markdown("###  Corrélations")
    correlations = {
        'Factualité': df_comparison[['human_factuality', 'auto_factuality']].corr().iloc[0,1],
        'Cohérence': df_comparison[['human_coherence', 'auto_coherence']].corr().iloc[0,1],
        'Score Global': df_comparison[['human_overall', 'auto_composite']].corr().iloc[0,1]
    }
    
    col1, col2, col3 = st.columns(3)
    for i, (metric, corr) in enumerate(correlations.items()):
        with [col1, col2, col3][i]:
            color = "normal" if corr > 0.7 else "inverse"
            st.metric(f"r({metric})", f"{corr:.3f}", delta=None)

def main():
    """Application principale"""
    st.title(" InsightDetector - Dashboard de Validation")
    st.markdown("*Analyse et validation des résumés générés par l'IA*")
    
    # Chargement des données
    with st.spinner("Chargement des données..."):
        summaries, evaluations, annotations = load_data()
    
    if summaries is None:
        st.stop()
    
    # Sidebar pour navigation
    st.sidebar.title(" Navigation")
    page = st.sidebar.selectbox(
        "Choisir une section",
        [" Vue d'ensemble", " Explorateur", "👥 Comparaison Humain/Auto", "⚙️ Paramètres"]
    )
    
    if page == " Vue d'ensemble":
        st.header(" Vue d'Ensemble des Performances")
        create_metrics_overview(evaluations)
        st.markdown("---")
        create_distribution_plots(evaluations)
        
    elif page == " Explorateur":
        summary_explorer(summaries, evaluations)
        
    elif page == " Comparaison Humain/Auto":
        human_vs_automatic_comparison(annotations, evaluations)
        
    elif page == " Paramètres":
        st.header(" Configuration")
        st.info("Fonctionnalités de configuration à venir...")
        
        # Export des données
        if st.button(" Exporter les données"):
            export_data = {
                'summaries': summaries,
                'evaluations': evaluations,
                'annotations': annotations
            }
            st.download_button(
                "⬇ Télécharger JSON complet",
                json.dumps(export_data, ensure_ascii=False, indent=2),
                "insightdetector_export.json",
                "application/json"
            )

if __name__ == "__main__":
    main()