# GPU compute — NVIDIA L40 (Tailscale)

Nodo remoto per inferenza VLM/LLM (Paper 2, suite A1–A4).  
**Non committare password o chiavi in questo repository.**

## Endpoint

| Campo | Valore |
|-------|--------|
| Host (Tailscale) | `100.86.223.16` |
| Hostname | `dagati-llm169` |
| User | `ml` |
| RAM | **256 GB** (~254 GB available) |
| GPU | **NVIDIA L40** — 46 GB VRAM |
| Inferenza | **Ollama** (`:11434`) |
| Disco | ~531 GB totali, ~411 GB liberi |

## Modelli Ollama (2026-08-12)

| Modello | Size | Uso |
|---------|------|-----|
| `qwen2.5vl:3b` | 3.2 GB | Vision leggero, routing study |
| `qwen2.5vl:7b` | 6.0 GB | Vision primario + varianti prompt |
| `llama3.2-vision:11b` | 7.8 GB | Vision medio |
| `gemma3:12b` | 8.1 GB | Vision medio |
| `qwen2.5vl:32b` | 21 GB | Vision massima latenza |

Elenco live: `./scripts/gpu/ssh-gpu.sh 'ollama list'`

## Dataset vision (no camera live)

VisA + MVTec AD — vedi [dataset-manufacturing-a1.md](dataset-manufacturing-a1.md).

```bash
cd XAIR_Runtime/experiments/datasets/manufacturing-a1/scripts
./download_visa.sh && ./download_mvtec.sh
python3 build_manifest.py --total 2000 --seed 42
```

## Accesso

```bash
cp config/compute-gpu.env.example config/compute-gpu.env
# Preferire SSH key: ssh-copy-id ml@100.86.223.16
./scripts/gpu/ssh-gpu.sh 'nvidia-smi'
export OLLAMA_HOST=http://100.86.223.16:11434
```

## Pipeline

**Ollama su L40** → AIS JSON → **XAIR su dev host** → metriche SER/CRR (Paper 1).
