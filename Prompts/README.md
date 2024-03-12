# prompts

- prompt_insights.txt
Extraire les insights, ainsi que leur type et leur tags

- prompt_feedbacks.txt
Classifier les insights

- prompt_regroupement.txt
Regrouper les insights par cluster

python src/trustpilot.py --company www.darty.com --to-page 449


'''
python3 -m llama_cpp.server --model /media/maitre/HDD1/Models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf --n_gpu_layers -1 --chat_format functionary --n_ctx 8000 --use_mlock 1
'''

'''
python3 -m llama_cpp.server --model /media/maitre/HDD1/Models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf --chat_format functionary --n_ctx 32000 
'''

python3 -m llama_cpp.server --model /media/maitre/HDD1/Models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf --n_gpu_layers 25 --chat_format functionary --n_ctx 0 

--chat_format llama-2 

python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-125m


docker run -p 8080:8080 -v /path/to/models:/models --gpus all ggerganov/llama.cpp:server-cuda -m models/7B/ggml-model.gguf -c 512 --host 0.0.0.0 --port 8080 --n-gpu-layers 99

docker run -p 8080:8080 -v /media/maitre/HDD1/Models/:/models --gpus all ggerganov/llama.cpp:server-cuda -m models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf -c 512 --host 0.0.0.0 --port 8080 --n-gpu-layers -1



cd ~/Documents/Development/llama.cpp
./server -m  /media/maitre/HDD1/Models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf -c 2048 

--n_threads=6  --n_batch=2048

python3 -m llama_cpp.server --model /media/maitre/HDD1/Models/mixtral-8x7b-instruct-v0.1.Q5_K_M.gguf --chat_format functionary --n_ctx 0 


sudo docker run --gpus all     -e HF_TOKEN=$HF_TOKEN -p 8000:8000     ghcr.io/mistralai/mistral-src/vllm:latest     --host 0.0.0.0     --model mistralai/Mistral-7B-Instruct-v0.2


sky launch -c  mistral-7b mistral-7b-v0.2.yaml --region eu-west-3
sky launch -c  mixtral  mixtral-8X7b-v0.1.yaml --region eu-west-3

IP=$(sky status --ip mixtral)

curl http://$IP:8000/v1/completions   -H "Content-Type: application/json"   -d '{
      "model": "mistralai/Mistral-7B-Instruct-v0.2",
      "prompt": "My favourite condiment is",
      "max_tokens": 25
  }'



python -u -m vllm.entrypoints.openai.api_server        --host 8000        --model mistralai/Mistral-7B-Instruct-v0.2 --max-model-len 8000



'''
python3 -m llama_cpp.server --model /media/maitre/HDD1/Models/mixtral-8x7b-instruct-v0.1.Q3_K_M.gguf --n_gpu_layers -1 --chat_format functionary --n_ctx 8000 --use_mlock 1
'''
