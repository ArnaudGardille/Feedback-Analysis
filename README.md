# Insighs-Analysis

conda create -n vigie_env python=3.10 jupyter
pip3 install torch torchvision torchaudio
pip install -r requirements.txt



streamlit run pipeline.py --server.fileWatcherType none --server.enableXsrfProtection=false


python src/trustpilot.py --company www.amazon.com --language en --to-page 948