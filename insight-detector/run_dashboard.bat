@echo off
echo 🧠 Lancement du Dashboard InsightDetector
echo.
echo Vérification de l'environnement...
python -c "import streamlit, plotly; print('✅ Dépendances OK')" 2>nul || (
    echo ❌ Installez les dépendances avec:
    echo pip install streamlit plotly
    pause
    exit /b 1
)

echo.
echo 🚀 Démarrage du dashboard sur http://localhost:8501
echo Appuyez sur Ctrl+C pour arrêter
echo.
streamlit run validation_dashboard.py --server.port 8501
pause