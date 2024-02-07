# Blumana-prompts

- prompt_insights.txt
Extraire les insights, ainsi que leur type et leur tags

- prompt_feedbacks.txt
Classifier les insights

- prompt_regroupement.txt
Regrouper les insights par cluster

python src/trustpilot.py --company www.darty.com --to-page 449


'''
python3 -m llama_cpp.server --model /media/maitre/HDD1/Models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf --n_gpu_layers 25 --chat_format functionary --n_ctx 32000
'''

'''
python3 -m llama_cpp.server --model /media/maitre/HDD1/Models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf --n_gpu_layers 35 --chat_format functionary --n_ctx 32000
'''