# Insighs-Analysis

conda create -n blumana_env jupyter
pip3 install torch torchvision torchaudio
pip install -r requirements.txt



streamlit run pipeline.py --server.fileWatcherType none --server.enableXsrfProtection=false