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

## Modelli Ollama (2026-08-11)

| Modello | Size | Uso Paper 2 |
|---------|------|-------------|
| `qwen2.5vl:7b` | 6.0 GB | Vision + AIS da immagini VisA/MVTec |
| `qwen2.5-coder:7b` | 4.7 GB | Smoke test / latenza bassa |
| `qwen3-coder:30b` | 18 GB | AIS strutturato principale |
| `qwen3.5:latest` | 6.6 GB | VLM leggero + tools |

Elenco live: `./scripts/gpu/ssh-gpu.sh 'ollama list'`

## Dataset vision (no camera live)

Paper 2 usa **VisA + MVTec AD** — vedi [`XAIR_Runtime/experiments/datasets/manufacturing-a1/README.md`](../../../XAIR_Runtime/experiments/datasets/manufacturing-a1/README.md).

```bash
cd XAIR_Runtime/experiments/datasets/manufacturing-a1/scripts
./download_visa.sh && ./download_mvtec.sh
python3 build_manifest.py --total 100 --seed 42
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
