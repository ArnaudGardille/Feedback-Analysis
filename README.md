# Insighs-Analysis

conda create -n blumana_env jupyter
pip3 install torch torchvision torchaudio
pip install -r requirements.txt



streamlit run pipeline.py --server.fileWatcherType none --server.enableXsrfProtection=false


python src/trustpilot.py --company www.amazon.com --language en --to-page 948