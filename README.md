# Personal Knowledge Agent

University information security project. Save notes and links from a browser or Telegram, process them in a worker, and search the results.

## Local photo and semantic search

Copy `.env.example` to `.env`, configure `PKA_WEB_USERS`, and put the Qwen GGUF file named by `MODEL_PATH` in `./models`. Set `LOCAL_EMBEDDINGS_URL=http://embeddings:8000/v1/embeddings` and leave the other `EMBEDDINGS_*` settings empty.

```sh
docker compose --profile embeddings up --build -d --wait embeddings
docker compose --profile embeddings up --build -d
```

The first command downloads the pinned EmbeddingGemma 2 weights into a persistent volume and waits for the model to load. Open the web app at `http://127.0.0.1:8000` to save a note or upload a JPEG or PNG up to 8 MiB and 16 megapixels. After the worker marks the note ready, use Semantic search to find it by meaning or visual details. Notes indexed with a different embedding model remain available through keyword search.

### NVIDIA GPU

On a host with a compatible NVIDIA driver and [NVIDIA Container Toolkit](https://docs.docker.com/engine/containers/gpu/), use the GPU override:

```sh
docker compose -f compose.yaml -f compose.gpu.yaml --profile embeddings up --build -d --wait embeddings
docker compose -f compose.yaml -f compose.gpu.yaml --profile embeddings up --build -d
```

EmbeddingGemma 2 selects CUDA when PyTorch can see a GPU. It uses the CPU when CUDA is unavailable or the model runs out of GPU memory during loading. For hosts without NVIDIA container support, use the standard startup commands above.

```sh
docker compose -f compose.yaml -f compose.gpu.yaml --profile embeddings exec -T embeddings \
  python -c 'from urllib.request import urlopen; print(urlopen("http://127.0.0.1:8000/health").read().decode())'
```
