# Insighs-Analysis


project_preparation_fr.ipynb : création des catégories
push_feedbacks.ipynb: On extrait les feedbacks du csv, et on les envoit dans Bubble
aspect_extraction_fr.ipynb: extraction des aspects
prepare_visu.ipynb:  on prépare les aspects pour l'affichage
insight_clustering.ipynb: On extrait les insights dans chaque catégorie

bubble.py
context.py
mistral.py
models.py
pipeline.py
preprocessing.py
trustpilot.py
utilities.py




conda create -n vigie_env python=3.10 jupyter
pip3 install torch torchvision torchaudio
pip install -r requirements.txt


watch nvidia-smi

streamlit run pipeline.py --server.fileWatcherType none --server.enableXsrfProtection=false


python src/trustpilot.py --company www.amazon.com --language en --to-page 948