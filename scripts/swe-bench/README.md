# Comparing Standard vs. Reproduction Setting on SWE-Bench Verified

This (sub-)repository is designed to evaluate and compare the performance and efficiency curves of an AI coding agent (**mini-SWE-agent**) across two distinct experimental environments:

1. **Standard Setting (Baseline):** The agent only has access to the GitHub issue description and the unmodified repository. It cannot run issue-specific gold tests to verify its work during development because those tests are not yet present in the repo.
2. **Reproduction Setting (Oracle Feedback):** The test patches (reproducer tests written specifically for the bug) are pre-applied to the repository *before* the agent starts. The agent can execute these tests in real-time, observing test failures, editing code, and iterating until the test suite passes.


---

## 🖥️ Hardware & Software Requirements

This study was evaluated on a high-performance HPC cluster node with the following specification:

*   **CPU Allocation**: 64 Cores (Intel(R) Xeon(R) Platinum 8468, 2.10GHz base / 3.80GHz max)
*   **RAM Allocation**: 1024 GB DDR5 RAM
*   **Storage**: 10 TB GPFS Shared Storage
*   **OS**: Red Hat Enterprise Linux 9.6 (Plow)
*   **Container Engine**: Podman v5.4.0 (running in rootless configuration)
*   **Python Environment**: Python 3.10 (managed via `uv` v0.11.15)

The LLM was hosted via vllm on a separate node using 8 H100 GPUs and 1024 GB DDR5 RAM.

---

## 🚀 Step-by-Step Reproduction Guide

Follow these steps on your remote server to run the complete standard and reproduction evaluations.

### 📊 Workflow Pipeline

The following diagram illustrates the complete execution pipeline, corresponding exactly to the step-by-step instructions below:

```mermaid
graph LR
    Step1["Step 1: Setup uv & Rootless Container Socket"] --> Step2["Step 2: Launch stable vLLM Model Server"]
    Step2 --> Step3["Step 3: Clone, Patch & Verify mini-swe-agent"]
    Step3 --> Step4["Step 4: Run Standard & Reproduction Sweeps on SWE-Bench Verified"]
    Step4 --> Step5["Step 5: Run SWE-Bench Verified Evaluation Harness"]
    Step5 --> Step6["Step 6: Trajectory Replay & Plot Curves"]

    style Step1 fill:#FFE6EA,stroke:#FFAEC0,color:#5C1E36
    style Step2 fill:#FFF0E6,stroke:#FFC3A0,color:#5A2E17
    style Step3 fill:#FEF9E7,stroke:#F7DC6F,color:#7D6608
    style Step4 fill:#E6F4EA,stroke:#8ED0A9,color:#1B4D3E
    style Step5 fill:#EBF5FB,stroke:#AED6F1,color:#1B4F72
    style Step6 fill:#F5EEF8,stroke:#D7BDE2,color:#4A235A
```

---

### Step 1: Setup Unified Package Manager (`uv`) and Rootless Container Registry

To run the evaluation, we use `uv` (a fast Rust-based package and Python manager) and `podman` (running in rootless mode to avoid requiring root/sudo privileges). 

We assume that system utilities (`git`, `curl`, `podman`) are already pre-installed by system administrators. We split this environment setup into **first-time setup** (done once) and **every-session activation** (done every time you log in).

#### 🔧 Step 1.1: First-Time Setup (Done Once)

> [!WARNING]
> **Node-Level Isolation Danger**: The `load_env.sh` and evaluation scripts aggressively manage local node resources in `/tmp` and kill all Podman processes owned by you. **If you submit batch jobs to the cluster, you MUST submit them using the "Exclusive Node" flag (e.g. `#BSUB -x` or `-n <max_cores>`)**. If two of your jobs land on the same physical compute node, they will assassinate each other's containers and collide on `/tmp` socket files!

First, create the isolated project folder where all tasks and environments will live, and navigate into it. Then run these commands inside that folder to install `uv`, create a persistent local symlink making `podman` respond to `docker` commands, and generate a `load_env.sh` utility helper:

<details><summary>code</summary>

```bash
mkdir -p $HOME/projects/swe-bench
cd $HOME/projects/swe-bench

# Install uv — UV_UNMANAGED_INSTALL prevents the installer from modifying ~/.bashrc etc.
curl -LsSf https://astral.sh/uv/install.sh | UV_VERSION="0.11.15" UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh

# Create the docker→podman wrapper (silences cgroup warnings that break agent output parsing)
mkdir -p ~/.local/bin
cat << 'EOF' > ~/.local/bin/docker
#!/bin/bash
REAL_PODMAN=$(which -a podman | grep -v "local/bin" | head -n 1)
exec "${REAL_PODMAN:-/usr/bin/podman}" --log-level=error "$@"
EOF
chmod +x ~/.local/bin/docker

cat << 'EOF' > load_env.sh
#!/bin/bash

# Note: Podman falls back to rootless single mapping into the namespace if subuid mappings are missing.
# The custom local docker wrapper (silencing warnings) makes this fallback fully transparent.

export PATH="$HOME/.local/bin:$PATH"

# Podman runtime metadata (sockets, locks) — tiny, stays on disk
export XDG_RUNTIME_DIR="/tmp/run-$(id -u)"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# Bypass GPFS parallel filesystem lock & permission issues by routing rootless storage to local /tmp
# A full 500-instance run pulls 200-500 GB of images; local SSD /tmp partition has 980+ GB free.
export CONTAINERS_STORAGE_CONF="$XDG_RUNTIME_DIR/storage.conf"
mkdir -p "/tmp/containers-$(id -u)/storage"
chmod 700 "/tmp/containers-$(id -u)"
FUSE_OVERLAY=$(which fuse-overlayfs 2>/dev/null || echo "/usr/bin/fuse-overlayfs")
cat << CONF_EOF > "$CONTAINERS_STORAGE_CONF"
[storage]
driver = "overlay"
runroot = "/tmp/run-$(id -u)/containers"
graphroot = "/tmp/containers-$(id -u)/storage"
rootless_storage_path = "/tmp/containers-$(id -u)/storage"

[storage.options.overlay]
mount_program = "$FUSE_OVERLAY"
ignore_chown_errors = "true"
CONF_EOF

# Terminate any existing podman system service daemon to guarantee it restarts using the correct local /tmp configuration
pkill -u "$(id -u)" -f "podman system service" 2>/dev/null
pkill -u "$(id -u)" -f podman 2>/dev/null
pkill -u "$(id -u)" -f conmon 2>/dev/null
pkill -u "$(id -u)" -f fuse-overlayfs 2>/dev/null
sleep 0.5

# Run system migrate to cleanly reset any stale pause processes or locks local to this node's /tmp environment
podman system migrate >/dev/null 2>&1

# Start rootless Podman background service socket
podman system service --time=0 "unix://$XDG_RUNTIME_DIR/podman.sock" >/dev/null 2>&1 &
sleep 1
export DOCKER_HOST="unix://$XDG_RUNTIME_DIR/podman.sock"

# uv, pip, and HuggingFace download caches → /tmp (avoids exceeding the 100GB home directory quota)
export UV_CACHE_DIR="/tmp/uv-cache-$(id -u)"
export PIP_CACHE_DIR="/tmp/pip-cache-$(id -u)"
export HF_HOME="/tmp/hf-home-$(id -u)"
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME"

echo "✨ Ready. Container images & locks: /tmp ($(df -h /tmp | awk 'NR==2{print $4}') free)"
EOF
chmod +x load_env.sh
```

</details>

#### ✅ Step 1.2: Verify Your Active Environment

To verify that your path bindings, active symlinks, and rootless containers are active and working on the current node, execute:

```bash
cd $HOME/projects/swe-bench && source ./load_env.sh

# Verify system utilities and active docker symlink
uv --version
git --version
docker --version  # Actually points to podman via symlink

# Verify active rootless container socket access
docker ps
```

### Step 2: Model Serving & Inference Environment (vLLM Launch)
To guarantee scientific reproducibility and deterministic environment isolation, all language model completions are served locally using **vLLM** on a dedicated high-performance deep learning node. Utilizing a local, self-hosted open-weights model ensures absolute consistency across runs and eliminates the risk of silent proprietary API updates, deprecations, or external network latency/connectivity fluctuations that would compromise the benchmark's reliability.

*   **Inference Hardware**: A dedicated compute node equipped with **8x NVIDIA H100 PCIe GPUs** (80GB VRAM per GPU) and 1024GB of RAM.
*   **Model Server**: vLLM serving framework (OpenAI-compatible REST API).
*   **Reference Model**: `MiniMaxAI/MiniMax-M2.7`.
*   **Serving Parameters**: Served across all 8 GPUs using **Tensor Parallelism of size 4** (`--tensor-parallel-size 4`) combined with **Expert Parallelism** (`--enable-expert-parallel`). Because MiniMax-M2.7's architecture does not support pure TP=8, running on >4 GPUs requires the TP4+EP2 setup (which maps 4-way TP + 2-way EP on 8 GPUs). It utilizes the official `--tool-call-parser minimax_m2` and `--reasoning-parser minimax_m2` parameters, enables auto-tool choice (`--enable-auto-tool-choice`), trusts remote code (`--trust-remote-code`), and uses the custom compilation config `'{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}'` to accelerate QK-norm operations. *Tip: You can optionally add `--kv-cache-dtype fp8` to compress the KV cache by 2x and double your parallel sequence capacity. Our main runs do not involve KV compression.*

#### 🔧 Step 2.1: Install vLLM (Done Once)
Connect to a **GPU serving node**, navigate to your project directory, create a dedicated virtual environment, and install vLLM. Since the home directory is shared via GPFS, this only needs to be done once:

<details><summary>code</summary>

```bash
cd $HOME/projects/swe-bench && source ./load_env.sh
uv venv vllm_env --python 3.10
source vllm_env/bin/activate
uv pip install "vllm==0.21.0"
```

</details>

#### 🔧 Step 2.2: Create the Launch Script (Done Once)
Create `start_vllm.sh` in your swe-bench directory. The script uses TP=4 + Expert Parallelism across 8 GPUs, disables DeepGEMM (requires `nvcc`, unavailable in batch jobs), and redirects GB-scale compiler caches away from the home directory:

<details><summary>code</summary>

```bash
cd $HOME/projects/swe-bench
cat << 'EOF' > start_vllm.sh
#!/bin/bash

source ./load_env.sh

# Disable DeepGEMM (requires nvcc, not available in batch jobs)
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0

# Redirect GB-scale compiler caches away from home directory
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor-$(id -u)"
export TRITON_CACHE_DIR="/tmp/triton-$(id -u)"
export CUDA_CACHE_PATH="/tmp/nv-cache-$(id -u)"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

source vllm_env/bin/activate
vllm serve MiniMaxAI/MiniMax-M2.7 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --moe-backend triton \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --max-model-len 64000 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --compilation-config '{"mode":3,"pass_config":{"fuse_minimax_qk_norm":true}}' \
EOF
chmod +x start_vllm.sh
```

</details>

#### 🚀 Step 2.3: Start the vLLM Server (Every Session)
Submit the server as a batch job to LSF (or any other scheduler). The single quotes defer `$HOME` expansion to the compute node, bypassing login-node path issues:
```bash
bsub -G grp_moc -J vllm-server -gpu "num=8:mode=exclusive_process" -n 32 -R "span[hosts=1]" -M 1024G -W 24:00 -o vllm_server.log -e vllm_server.err 'cd $HOME/projects/swe-bench && ./start_vllm.sh'
```

*(Monitor startup in real-time: `tail -f vllm_server.log vllm_server.err`)*

#### 🔀 Step 2.3.5: Load Balancing Multiple vLLM Servers (Multi-Node Scaling)
If you spun up multiple vLLM servers (e.g. by submitting the `bsub` job multiple times to different GPU nodes), you can tie them together into a single, high-throughput, load-balanced endpoint using the native **LiteLLM Proxy** already installed in your client environment.

Since parallel client workers execute heterogeneous tasks at different speeds (e.g., git checkouts, fast shell operations, or long-running test suites), request arrival is heavily jittered. Using the concurrency-based **usage-based-routing-v2** strategy ensures that requests are dynamically routed to the node with the lowest active load, preventing queue bottlenecks and latency spikes.

<details><summary>details</summary>

1. **Get the hostnames of your active GPU nodes:**
   Check LSF to identify the hostnames of all your running vLLM servers:
   ```bash
   bjobs -w
   ```
   *(Note down the node names, e.g., `gpu-node-1`, `gpu-node-2`, etc.)*

2. **Create a LiteLLM Proxy configuration file:**
   On your CPU compute node (where you will run the client sweep), create a YAML configuration file listing all your active servers:
   ```bash
   cd $HOME/projects/swe-bench/mini-swe-agent
   
   cat << 'EOF' > litellm_proxy_config.yaml
   model_list:
     - model_name: MiniMaxAI/MiniMax-M2.7
       litellm_params:
         model: openai/MiniMaxAI/MiniMax-M2.7
         api_base: http://<gpu-node-1>:8000/v1
         api_key: vllm-token
     - model_name: MiniMaxAI/MiniMax-M2.7
       litellm_params:
         model: openai/MiniMaxAI/MiniMax-M2.7
         api_base: http://<gpu-node-2>:8000/v1
         api_key: vllm-token
     # Add more nodes here as needed...
   
   router_settings:
     routing_strategy: "usage-based-routing-v2"  # Route to node with the least active requests
     allowed_fails: 3  # Failover protection if a node crashes
   EOF
   ```
   *Replace `<gpu-node-1>`, `<gpu-node-2>`, etc., with your actual GPU compute node hostnames.*

3. **Start the LiteLLM Proxy server:**
   Inside your active client virtual environment on the CPU compute node, terminate any stale proxy processes to free up the port, and then launch the proxy in the background:
   ```bash
   cd $HOME/projects/swe-bench/mini-swe-agent
   source .venv/bin/activate
   pkill -u $USER -f litellm
   sleep 3
   nohup litellm --config litellm_proxy_config.yaml --port 8000 > litellm_proxy.log 2>&1 &
   ```
   *(This starts the load balancer at `http://localhost:8000/v1`)*

4. **Point your client to the Proxy:**
   Now, in Step 2.4, simply set your `OPENAI_API_BASE` to point to the local load balancer instead of a single GPU node (by exporting `localhost`):

</details>

#### 🌐 Step 2.4: Configure API Endpoint and Disable Cost Tracking (Every Session)
Set the GPU server address on your CPU compute node. Replace `localhost` with the hostname shown by `bjobs -l` or `bpeek` after the vLLM job starts (or use `localhost` if running the LiteLLM load balancer in Step 2.3.5):

```bash
export OPENAI_API_BASE="http://localhost:8000/v1"
export OPENAI_API_KEY="vllm-token"
export MSWEA_COST_TRACKING='ignore_errors'
```

#### 📡 Step 2.5: Verify the Model Connection (Every Session)
Confirm the CPU node can reach the vLLM server:

```bash
curl -s -X POST "$OPENAI_API_BASE/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{"model":"MiniMaxAI/MiniMax-M2.7","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}'
```
A successful response returns a JSON payload with the model's reply.

### Step 3: Clone, Setup, and Patch `mini-swe-agent`
`mini-swe-agent` achieves state-of-the-art results using a direct bash-only interaction framework. We pull pre-built evaluation environments from Epoch's GHCR registry instead of Docker Hub to avoid rate limits.

#### 🔧 Step 3.1: Clone and Install (Done Once)

<details><summary>code</summary>

```bash
cd $HOME/projects/swe-bench
git clone https://github.com/SWE-agent/mini-swe-agent.git
cd mini-swe-agent
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv pip install 'litellm[proxy]'
```

</details>

#### 🔑 Step 3.2: Authenticate with GHCR, GitHub, and Hugging Face (Every Session)
These credentials are lost when the node restarts or you get a new compute allocation. Re-run this each session. 
Use a **Personal Access Token (Classic)** with `read:packages` and `repo` scopes for GitHub. Use a standard **Read Token** from your Hugging Face settings for `HF_TOKEN`.
```bash
echo "YOUR_GITHUB_PERSONAL_ACCESS_TOKEN" | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# Rewrite GitHub HTTPS URLs to use your token (increases rate limits and avoids NAT IP blocks)
git config --global url."https://YOUR_GITHUB_USERNAME:YOUR_GITHUB_PERSONAL_ACCESS_TOKEN@github.com/".insteadOf "https://github.com/"

# Set Hugging Face token to silence warnings and boost datasets API rate limits
export HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

#### 🔧 Step 3.3: Patch Registry Paths (Done Once)
**1. Patch all `docker.io/swebench` references** to `ghcr.io/epoch-research` in all files except `.venv` and `vllm_env` directories:
Note that all patches must be run from inside the `mini-swe-agent/` directory.

<details><summary>code</summary>

```bash
cd $HOME/projects/swe-bench/mini-swe-agent
find . -type f -not -path '*/.*' -not -path './.venv/*' -not -path './vllm_env/*' \
  -exec sed -i 's|docker.io/swebench|ghcr.io/epoch-research|g' {} +
```

</details>

<br>

**2. Patch `swebench.py`** to remove the `_1776_` double-underscore hack and update the image path to the correct `swe-bench.eval` format:

<details><summary>code</summary>

```bash
cd $HOME/projects/swe-bench/mini-swe-agent
python3 << 'EOF'
path = 'src/minisweagent/run/benchmarks/swebench.py'
with open(path, 'r') as f:
    content = f.read()

target = '''    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"ghcr.io/epoch-research/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()'''

replacement = '''    if image_name is None:
        iid = instance["instance_id"]
        image_name = f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{iid}:latest".lower()'''

if target in content:
    with open(path, 'w') as f:
        f.write(content.replace(target, replacement))
    print('Successfully patched swebench.py for GHCR!')
else:
    print('Target not found or already updated!')
EOF
```

</details>

<br>

**3. Increase the image pull timeout** from 120s to 900s to handle slow concurrent pulls with 16 workers:

<details><summary>code</summary>

```bash
cd $HOME/projects/swe-bench/mini-swe-agent
python3 << 'EOF'
path = 'src/minisweagent/environments/docker.py'
with open(path, 'r') as f:
    content = f.read()

target = 'pull_timeout: int = 120'
replacement = 'pull_timeout: int = 900'

if target in content:
    with open(path, 'w') as f:
        f.write(content.replace(target, replacement))
    print('Successfully increased pull_timeout to 900 seconds!')
else:
    print('Target not found or already updated!')
EOF
```

</details>

<br>

**Verify** the patches took effect — you should see exactly one match:
```bash
cd $HOME/projects/swe-bench/mini-swe-agent
grep -rn "ghcr.io/epoch-research/swe-bench.eval" . --exclude-dir={.git,.venv}
```
```text
./src/minisweagent/run/benchmarks/swebench.py:73:        image_name = f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{iid}:latest".lower()
```

#### ✅ Step 3.4: Pre-Flight Verification (Every Session)
Run the agent on a single instance to confirm the server connection, tool parsers, and container pull all work before committing to the full 500-instance sweep:
```bash
export MSWEA_COST_TRACKING='ignore_errors'
cd $HOME/projects/swe-bench/mini-swe-agent
source .venv/bin/activate
mini-extra swebench-single \
  --model openai/MiniMaxAI/MiniMax-M2.7 \
  --subset verified \
  --split test \
  --instance django__django-16595 \
  --yolo
```

#### 🔧 Step 3.5: Create the Clean Recovery Script (Done Once)
To easily recover if a parallel run freezes, crashes, or is manually aborted, create a `clean.sh` recovery helper inside the `mini-swe-agent` directory. This script terminates orphaned background processes and resets the rootless Podman socket while keeping cached container images active:

<details><summary>code</summary>

```bash
cd $HOME/projects/swe-bench/mini-swe-agent
cat << 'EOF' > clean.sh
#!/bin/bash
# Usage: ./clean.sh [--prune-images] [--clean-failed] [--clean-failed-all]
#   --prune-images      Wipe cached container images from local /tmp (frees 200-500 GB;
#                       images will be re-pulled on the next run).
#   --clean-failed      Automatically detect and delete the output JSON and trajectory files
#                       of any instances that failed due to transient/internal errors
#                       (exit_status != "Submitted", "ContextWindowExceededError", and "LimitsExceeded"),
#                       preserving ContextWindowExceededError and LimitsExceeded runs as permanent failures to avoid re-running them.
#   --clean-failed-all  Automatically detect and delete the output JSON and trajectory files
#                       of ALL failed/incomplete instances, including ContextWindowExceededError and LimitsExceeded.

PRUNE_IMAGES=false
CLEAN_FAILED=false
export CLEAN_FAILED_ALL=false

for arg in "$@"; do
  case "$arg" in
    --prune-images)       PRUNE_IMAGES=true ;;
    --clean-failed)       CLEAN_FAILED=true ;;
    --clean-failed-all)   CLEAN_FAILED=true; CLEAN_FAILED_ALL=true ;;
  esac
done

echo "🧹 Cleaning environment..."

source "$(dirname "$0")/../load_env.sh" >/dev/null 2>&1 || source "$(dirname "$0")/load_env.sh" >/dev/null 2>&1 || true

if [[ "$(hostname)" == *"login"* ]]; then
  echo "⚠️  On login node — skipping container cleanup. Re-run on the compute node."
else
  pkill -u "$USER" -f podman 2>/dev/null; pkill -u "$USER" -f conmon 2>/dev/null; pkill -u "$USER" -f fuse-overlayfs 2>/dev/null
  sleep 1

  XDG_RUNTIME_DIR="/tmp/run-$(id -u)"
  rm -rf "$XDG_RUNTIME_DIR"

  if [ "$PRUNE_IMAGES" = true ]; then
    echo "🗑️  Pruning image cache from /tmp..."
    rm -rf "/tmp/containers-$(id -u)/storage"
  fi

  echo "🔹 Re-generating local storage configuration..."
  source "$(dirname "$0")/../load_env.sh" >/dev/null 2>&1 || source "$(dirname "$0")/load_env.sh" >/dev/null 2>&1 || true
  podman system migrate >/dev/null 2>&1
fi

pkill -u "$USER" -f mini-extra 2>/dev/null; pkill -u "$USER" -f minisweagent 2>/dev/null

if [ "$CLEAN_FAILED" = true ]; then
  if [ "$CLEAN_FAILED_ALL" = true ]; then
    echo "🗑️  Cleaning up ALL failed/incomplete runs (including ContextWindowExceededError and LimitsExceeded) and syncing preds.json..."
  else
    echo "🗑️  Cleaning up failed/incomplete runs (excluding ContextWindowExceededError and LimitsExceeded) and syncing preds.json..."
  fi
  python3 -c '
import os, shutil, json
dirs = ["./evaluation_output_standard", "./evaluation_output_reproduction"]
clean_all = os.environ.get("CLEAN_FAILED_ALL") == "true"

for out_dir in dirs:
    if os.path.exists(out_dir):
        print(f"Syncing directory: {out_dir}")
        # 1. Clear failed runs by scanning subdirectories
        for subdir in os.listdir(out_dir):
            subdir_path = os.path.join(out_dir, subdir)
            if os.path.isdir(subdir_path):
                traj_file = f"{subdir}.traj.json"
                traj_path = os.path.join(subdir_path, traj_file)
                if os.path.exists(traj_path):
                    try:
                        with open(traj_path, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        status = data.get("info", {}).get("exit_status", "")
                        if status == "Submitted":
                            continue
                        elif status in ("ContextWindowExceededError", "LimitsExceeded"):
                            if clean_all:
                                print(f"  ❌ Wiping failed run directory: {subdir} (Status: {status})")
                                shutil.rmtree(subdir_path)
                            else:
                                print(f"  🛡️  Preserving {status} run directory: {subdir} (Permanent Failure)")
                        else:
                            print(f"  ❌ Wiping failed run directory: {subdir} (Status: {status})")
                            shutil.rmtree(subdir_path)
                    except Exception:
                        pass
                else:
                    print(f"  ❌ Wiping empty/incomplete run directory: {subdir}")
                    shutil.rmtree(subdir_path)
        # 2. Gather remaining successful and preserved runs
        successful = set()
        for subdir in os.listdir(out_dir):
            subdir_path = os.path.join(out_dir, subdir)
            if os.path.isdir(subdir_path):
                traj_file = f"{subdir}.traj.json"
                traj_path = os.path.join(subdir_path, traj_file)
                if os.path.exists(traj_path):
                    try:
                        with open(traj_path, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        status = data.get("info", {}).get("exit_status", "")
                        if status in ("Submitted", "ContextWindowExceededError", "LimitsExceeded"):
                            successful.add(subdir)
                    except Exception:
                        pass
        # 3. Synchronize preds.json
        preds_path = os.path.join(out_dir, "preds.json")
        if os.path.exists(preds_path):
            try:
                with open(preds_path, "r", encoding="utf-8") as fh:
                    preds = json.load(fh)
                orig_len = len(preds)
                if isinstance(preds, dict):
                    preds = {k: v for k, v in preds.items() if k in successful}
                elif isinstance(preds, list):
                    preds = [item for item in preds if item.get("instance_id") in successful]
                print(f"  ✨ Synced preds.json: {orig_len} -> {len(preds)} entries")
                with open(preds_path, "w", encoding="utf-8") as fh:
                    json.dump(preds, fh, indent=4)
            except Exception as e:
                print(f"  ⚠️ Error syncing preds.json: {e}")
'
fi

# To start a 100% fresh run from scratch, you can manually delete the output directories:
# rm -rf ./evaluation_output_standard ./evaluation_output_reproduction

echo "✨ Done. Local /tmp space: $(df -h /tmp | awk 'NR==2{print $4 " free"}')"
EOF
chmod +x clean.sh
```

</details>

#### 🔧 Step 3.6: Patch for Reproduction Setting (Done Once)
This patch makes `mini-swe-agent` pre-apply the golden `test_patch` to the repository before the agent starts if the `--apply-test-patch` CLI flag is provided to `mini-extra swebench`. Since the codebase uses pre-built Docker containers from Epoch's registry that already contain the repository at the correct `base_commit` inside the `/testbed` directory, there is no dynamic `git clone` or `git checkout` in `docker.py`. Instead, we patch `src/minisweagent/run/benchmarks/swebench.py` to write `test_patch` and run `git apply` inside the container right before starting the agent loop. 

Run the patch from inside your `mini-swe-agent` directory:

<details><summary>code</summary>


```bash
cd $HOME/projects/swe-bench/mini-swe-agent
python3 << 'EOF'
import os
path = 'src/minisweagent/run/benchmarks/swebench.py'
with open(path, 'r') as f:
    content = f.read()

target_env = '''    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env'''

replacement_env = '''    env = get_environment(env_config)

    # Pre-apply test patch for reproduction setting (oracle feedback)
    if config.get("apply_test_patch"):
        if instance.get("test_patch"):
            import base64
            test_patch = instance["test_patch"]
            patch_b64 = base64.b64encode(test_patch.encode("utf-8")).decode("utf-8")
            logger.info(f"[{instance['instance_id']}] Pre-applying test patch for reproduction setting...")
            out = env.execute({"command": f"echo '{patch_b64}' | base64 -d > test_patch.diff"})
            out = env.execute({"command": "git apply --ignore-whitespace test_patch.diff"})
            if out["returncode"] != 0:
                # Fallback: try 3-way merge which is more forgiving of context mismatches
                out = env.execute({"command": "git apply --ignore-whitespace -3 test_patch.diff"})
            if out["returncode"] != 0:
                raise RuntimeError(f"[{instance['instance_id']}] Failed to apply test patch. Output: {out['output']}")
            env.execute({"command": "rm -f test_patch.diff"})
        else:
            logger.warning(f"[{instance['instance_id']}] apply_test_patch is True but no test_patch found in instance!")

    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command, undefined=StrictUndefined).render(**instance)
        out = env.execute({"command": startup_command})
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env'''

target_cli = '''    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")'''

replacement_cli = '''    environment_class: str | None = typer.Option(None, "--environment-class", help="Environment type to use. Recommended are docker or singularity", rich_help_panel="Advanced"),
    apply_test_patch: bool = typer.Option(False, "--apply-test-patch", help="Pre-apply golden test patch to repository before starting agent (reproduction setting)", rich_help_panel="Data selection"),
) -> None:
    # fmt: on
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to {output_path}")'''

target_configs = '''    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
    })'''

replacement_configs = '''    configs.append({
        "environment": {"environment_class": environment_class or UNSET},
        "model": {"model_name": model or UNSET, "model_class": model_class or UNSET},
        "apply_test_patch": apply_test_patch,
    })'''

if target_env in content and target_cli in content and target_configs in content:
    content = content.replace(target_env, replacement_env)
    content = content.replace(target_cli, replacement_cli)
    content = content.replace(target_configs, replacement_configs)
    with open(path, 'w') as f:
        f.write(content)
    print('✅ Successfully patched swebench.py to pre-apply test patches!')
else:
    print('⚠️ Target strings not found. File may be already updated or modified.')
EOF
```

</details>

### Step 4: Run `mini-swe-agent` in Both Experimental Settings

#### 🚀 Step 4.1: Run Standard Setting
Run the agent on the unmodified codebase. The agent has no access to the golden tests:
```bash
# If running as a batch job (e.g. bsub), add this cleanup trap to prevent zombie containers on abort:
cleanup() { echo "Job killed! Cleaning up..."; podman rm -fa 2>/dev/null || true; pkill -u "$(id -u)" -f catatonit || true; }
trap cleanup EXIT SIGINT SIGTERM

cd $HOME/projects/swe-bench && source ./load_env.sh
cd mini-swe-agent
source .venv/bin/activate
mini-extra swebench \
  --model openai/MiniMaxAI/MiniMax-M2.7 \
  --subset verified \
  --split test \
  --workers 4 \
  --output ./evaluation_output_standard
```

#### 🚀 Step 4.2: Run Reproduction Setting
Run the agent with the test patch pre-applied (requires the Step 4.2 patch to `docker.py`):
```bash
# If running as a batch job (e.g. bsub), add this cleanup trap to prevent zombie containers on abort:
cleanup() { echo "Job killed! Cleaning up..."; podman rm -fa 2>/dev/null || true; pkill -u "$(id -u)" -f catatonit || true; }
trap cleanup EXIT SIGINT SIGTERM

cd $HOME/projects/swe-bench && source ./load_env.sh
cd mini-swe-agent
source .venv/bin/activate
mini-extra swebench \
  --model openai/MiniMaxAI/MiniMax-M2.7 \
  --subset verified \
  --split test \
  --workers 4 \
  --apply-test-patch \
  --output ./evaluation_output_reproduction
```

*Both runs write action-by-action logs to `./evaluation_output_standard` and `./evaluation_output_reproduction`.*

#### 🔧 Step 4.3: Troubleshooting & Clean Recovery (As Needed)
If a run freezes, runs out of KV cache, or you manually abort it midway using `Ctrl+C`, run the following fixes before restarting:

<details><summary>fixes</summary>

1. **Clean up background processes and containers:** 
   Execute the recovery script inside your `mini-swe-agent` directory to purge zombie `podman`/`conmon` environments:
   ```bash
   cd $HOME/projects/swe-bench/mini-swe-agent
   ./clean.sh
   ```
   *Note: This script preserves your completed predictions and logs inside `./evaluation_output_standard` or `./evaluation_output_reproduction` by default.*
2. **Clean up failed/incomplete runs (Optional):**
    If you ran into cluster timeouts or `InternalServerError`s, you can pass the `--clean-failed` flag. This will automatically delete the JSON and trajectory logs of instances that failed due to transient/internal errors, while safely preserving both `ContextWindowExceededError` and `LimitsExceeded` runs as permanent failures.
   
   If you explicitly want to clean up and retry *all* failed runs, including those that exceeded the context window or limits, pass the `--clean-failed-all` flag instead:
   ```bash
   cd $HOME/projects/swe-bench/mini-swe-agent
   ./clean.sh --clean-failed
   # OR to clean everything including context window exceeded errors:
   ./clean.sh --clean-failed-all
   ```
3. **Rerun and Resume seamlessly:**
   Simply rerun your execution sweep command (Step 4.1 or Step 4.2). `mini-swe-agent` will automatically detect your existing output files, skip all completed instances, and resume right from where it left off!
4. **Starting Fresh (Optional):**
   If you ever want to discard a run entirely and start 100% fresh from scratch, manually wipe the output directories before rerunning:
   ```bash
   cd $HOME/projects/swe-bench/mini-swe-agent
   rm -rf ./evaluation_output_standard ./evaluation_output_reproduction
   ```

</details>

### Step 5: Epoch's SWE-bench Evaluation Harness

#### 🔧 Step 5.1: Install and Patch the Harness (Done Once)
Clone Epoch's fork of SWE-bench. This fork maps the `swe-bench.eval` image naming to the GHCR registry. We also patch:
1. `run_evaluation.py` to support dictionary-formatted prediction files natively (as produced by `mini-swe-agent`).
2. `docker_build.py` to pull base, environment, and instance images directly from GHCR (`ghcr.io/epoch-research`) instead of building them locally, bypassing parallel filesystem namespace limitations (`lchown` errors).
3. `test_spec.py` to replace slashes with dashes in generated container names, preventing Podman/Docker name validation failures (`invalid argument`).
4. `docker_utils.py` to make `write_to_container` shell-safe and robust, completely bypassing `lchown` namespace copy errors on GPFS by writing files directly via heredoc strings inside container shells (`cat << 'EOF'`).

<details><summary>code</summary>

```bash
cd $HOME/projects/swe-bench
git clone https://github.com/epoch-research/SWE-bench.git
cd SWE-bench
uv venv .venv
source .venv/bin/activate
uv pip install -e .

# Revert any previous edits in the repository to guarantee a clean upstream baseline before patching
git checkout swebench/harness/run_evaluation.py swebench/harness/docker_build.py swebench/harness/test_spec.py 2>/dev/null || true

# 1. Patch run_evaluation.py to support mixed predictions and write_to_container bypasses
python3 << 'EOF'
path = 'swebench/harness/run_evaluation.py'
with open(path, 'r') as f:
    content = f.read()

# A. Mixed predictions parser
target_parser = '    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}'
replacement_parser = '''    if isinstance(predictions, dict):
        new_predictions = []
        for k, v in predictions.items():
            if isinstance(v, str):
                new_predictions.append({
                    "instance_id": k,
                    "model_patch": v,
                    "model_name_or_path": "model"
                })
            elif isinstance(v, dict):
                pred_item = v.copy()
                pred_item["instance_id"] = pred_item.get("instance_id", k)
                if "model_patch" not in pred_item and "patch" in pred_item:
                    pred_item["model_patch"] = pred_item["patch"]
                new_predictions.append(pred_item)
        predictions = new_predictions
    elif isinstance(predictions, list):
        new_predictions = []
        for pred_item in predictions:
            if isinstance(pred_item, dict):
                item = pred_item.copy()
                if "model_patch" not in item and "patch" in item:
                    item["model_patch"] = item["patch"]
                new_predictions.append(item)
        predictions = new_predictions
    predictions = {pred[KEY_INSTANCE_ID]: pred for pred in predictions}'''

# B. Import write_to_container
target_import = '    copy_to_container,\n    exec_run_with_timeout,'
replacement_import = '    copy_to_container,\n    write_to_container,\n    exec_run_with_timeout,'

# C. Use write_to_container for intermediate patch
target_patch = '        copy_to_container(container, patch_file, Path("/tmp/patch.diff"))'
replacement_patch = '        write_to_container(container, pred["model_patch"] or "", Path("/tmp/patch.diff"))'

# D. Use write_to_container for eval script
target_eval = '        copy_to_container(container, eval_file, Path("/eval.sh"))'
replacement_eval = '        write_to_container(container, test_spec.eval_script, Path("/eval.sh"))'

patched = False
if target_parser in content:
    content = content.replace(target_parser, replacement_parser)
    patched = True
if target_import in content:
    content = content.replace(target_import, replacement_import)
    patched = True
if target_patch in content:
    content = content.replace(target_patch, replacement_patch)
    patched = True
if target_eval in content:
    content = content.replace(target_eval, replacement_eval)
    patched = True

if patched:
    with open(path, 'w') as f:
        f.write(content)
    print('✅ Successfully patched run_evaluation.py!')
else:
    print('ℹ️  run_evaluation.py already patched or target string not found.')
EOF

# 2. Patch docker_build.py to enable transparent pulling of base, env, and instance images
python3 << 'EOF'
path = 'swebench/harness/docker_build.py'
with open(path, 'r') as f:
    content = f.read()

# A. Patch base image pulling
target_base = '''        except docker.errors.ImageNotFound:
            pass
        # Build the base image (if it does not exist or force rebuild is enabled)'''
replacement_base = '''        except docker.errors.ImageNotFound:
            try:
                print(f"Attempting to pull base image {image_name} from registry...")
                client.images.pull(image_name)
                print(f"Successfully pulled base image {image_name}!")
                continue
            except Exception as e:
                print(f"Failed to pull base image {image_name} ({e}). Will build it from source.")
                pass
        # Build the base image (if it does not exist or force rebuild is enabled)'''

if target_base in content:
    content = content.replace(target_base, replacement_base)
    print('✅ Patched base image pulling in docker_build.py')

# B. Patch environment image pulling
target_env = '''                remove_image(client, test_spec.env_image_key, "quiet")
                image_exists = False
        except docker.errors.ImageNotFound:
            pass
        if not image_exists:'''
replacement_env = '''                remove_image(client, test_spec.env_image_key, "quiet")
                image_exists = False
        except docker.errors.ImageNotFound:
            try:
                print(f"Attempting to pull environment image {test_spec.env_image_key} from registry...")
                client.images.pull(test_spec.env_image_key)
                print(f"Successfully pulled environment image {test_spec.env_image_key}!")
                image_exists = True
            except Exception as e:
                print(f"Failed to pull environment image {test_spec.env_image_key} ({e}). Will build it from source.")
                pass
        if not image_exists:'''

if target_env in content:
    content = content.replace(target_env, replacement_env)
    print('✅ Patched env image pulling in docker_build.py')

# C. Patch instance image pulling
target_inst = '''    except docker.errors.ImageNotFound:
        pass

    # Build the instance image'''
replacement_inst = '''    except docker.errors.ImageNotFound:
        try:
            logger.info(f"Attempting to pull instance image {image_name} from registry...")
            client.images.pull(image_name)
            logger.info(f"Successfully pulled instance image {image_name}!")
            image_exists = True
        except Exception as e:
            logger.info(f"Failed to pull instance image {image_name} ({e}). Will build it from source.")
            pass

    # Build the instance image'''

if target_inst in content:
    content = content.replace(target_inst, replacement_inst)
    print('✅ Patched instance image pulling in docker_build.py')

with open(path, 'w') as f:
    f.write(content)
print('🚀 docker_build.py patch complete!')
EOF

# 3. Patch test_spec.py to prevent container naming crashes
python3 << 'EOF'
path = 'swebench/harness/test_spec.py'
with open(path, 'r') as f:
    content = f.read()

target = '''    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"eval.{self.instance_id}"
        return f"{registry_repo}.eval.{self.instance_id}.{run_id}"'''

replacement = '''    def get_instance_container_name(self, run_id=None):
        if not run_id:
            return f"eval.{self.instance_id}"
        safe_prefix = registry_repo.replace("/", "-")
        return f"{safe_prefix}.eval.{self.instance_id}.{run_id}"'''

if target in content:
    with open(path, 'w') as f:
        f.write(content.replace(target, replacement))
    print('✅ Successfully patched test_spec.py!')
else:
    print('ℹ️  test_spec.py already patched or target string not found.')
EOF

# 4. Patch docker_utils.py to make write_to_container shell-safe and robust
python3 << 'EOF'
path = 'swebench/harness/docker_utils.py'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.read().splitlines()

start_idx = -1
for i, line in enumerate(lines):
    if line.strip().startswith("def write_to_container"):
        start_idx = i
        break

if start_idx != -1:
    end_idx = -1
    for j in range(start_idx + 1, len(lines)):
        if lines[j].startswith("def ") or lines[j].startswith("class "):
            end_idx = j
            break
    if end_idx == -1:
        end_idx = len(lines)
    
    # Check if already patched
    already_patched = False
    for k in range(start_idx, end_idx):
        if "Failed to write to container at" in lines[k]:
            already_patched = True
            break
            
    if already_patched:
        print('✅ docker_utils.py is already successfully patched!')
    else:
        new_impl = [
            "def write_to_container(container: Container, data: str, dst: Path):",
            "    \"\"\"",
            "    Write a string to a file in a docker container",
            "    \"\"\"",
            "    # Make sure parent directory exists first",
            "    container.exec_run(f\"mkdir -p {dst.parent}\", user=\"root\")",
            "",
            "    # echo with heredoc to file",
            "    command = f\"cat <<'{HEREDOC_DELIMITER}' > {dst}\\n{data}\\n{HEREDOC_DELIMITER}\"",
            "    res = container.exec_run([\"/bin/bash\", \"-c\", command], user=\"root\")",
            "    if res.exit_code != 0:",
            "        # Try with /bin/sh as fallback",
            "        res_sh = container.exec_run([\"/bin/sh\", \"-c\", command], user=\"root\")",
            "        if res_sh.exit_code != 0:",
            "            raise RuntimeError(",
            "                f\"Failed to write to container at {dst}.\\n\"",
            "                f\"bash exit code: {res.exit_code}, output: {res.output.decode('utf-8', errors='ignore')}\\n\"",
            "                f\"sh exit code: {res_sh.exit_code}, output: {res_sh.output.decode('utf-8', errors='ignore')}\"",
            "            )"
        ]
        lines[start_idx:end_idx] = new_impl
        with open(path, 'w', newline='\n', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print('✅ Successfully patched docker_utils.py!')
else:
    print('❌ def write_to_container not found in docker_utils.py!')
EOF
```

</details>

#### 🚀 Step 5.2: Run the Evaluation (Done after each run)
Evaluate the agent's prediction files against the ground-truth tests. 

> [WARNING] Any batch job running this evaluation **must** be submitted with exclusive node access to prevent cross-job container assassination!

```bash
# If running as a batch job (e.g. bsub), add this cleanup trap to prevent zombie containers on abort:
cleanup() { echo "Job killed! Cleaning up..."; podman rm -fa 2>/dev/null || true; pkill -u "$(id -u)" -f catatonit || true; }
trap cleanup EXIT SIGINT SIGTERM

cd $HOME/projects/swe-bench/SWE-bench

# Standard setting
source ../load_env.sh
source .venv/bin/activate
rm -rf logs/run_evaluation/mini-swe-agent-verified-standard
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path ../mini-swe-agent/evaluation_output_standard/preds.json \
  --max_workers 16 \
  --run_id mini-swe-agent-verified-standard

# Reproduction setting
source ../load_env.sh
source .venv/bin/activate
rm -rf logs/run_evaluation/mini-swe-agent-verified-reproduction
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path ../mini-swe-agent/evaluation_output_reproduction/preds.json \
  --max_workers 16 \
  --run_id mini-swe-agent-verified-reproduction
```
*Results are written to `logs/run_evaluation/`.*

---

### Step 6: Trajectory Replay & Plot Curves

Instead of re-running the expensive agent pipeline multiple times under different action limits, **Trajectory Replay** parses the `.traj.json` logs, replays executed actions up to action count $a$, captures the resulting `git diff`, and evaluates those patches.

```mermaid
graph LR
    Traj[".traj.json Trajectories"] --> Replay["Replay Action Commands"]
    Replay --> Patch["Extract Action-a Patch"]
    Patch --> Eval["Evaluate with Podman/Docker"]

    style Traj fill:#FFE6EA,stroke:#FFAEC0,color:#5C1E36
    style Replay fill:#FFF0E6,stroke:#FFC3A0,color:#5A2E17
    style Patch fill:#E6F4EA,stroke:#8ED0A9,color:#1B4D3E
    style Eval fill:#E8EAFF,stroke:#B0B7FF,color:#2C3E50
```

#### 🔧 Step 6.1: Create the Replay Utilities (Done Once)
Run the following code to create the python utility scripts required for trajectory extraction and curve plotting:

<details><summary>code</summary>

<br>

`extract_action_patches.py` — python script to sequentially execute standard commands and extract git diffs:

```bash
cd $HOME/projects/swe-bench/SWE-bench/
cat << 'EOF' > extract_action_patches.py
#!/usr/bin/env python3
"""
extract_action_patches.py (Containerized & Stateful Version)

Trajectory Replay for SWE-bench Verified.
Generates a stateful monolithic bash script to replay agent commands sequentially 
inside the exact SWE-bench Docker containers, guaranteeing 100% environmental fidelity.
"""

import os
import sys
import json
import argparse
import subprocess
import concurrent.futures
import time
import threading
import uuid
import shlex
from typing import List, Dict, Any

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package is required. Install it using: pip install datasets")
    sys.exit(1)

def parse_trajectory_commands(file_path: str) -> List[str]:
    """Parses a mini-swe-agent trajectory file and extracts the sequence of bash commands."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    messages = []
    if isinstance(data, dict):
        messages = data.get("history", data.get("messages", data.get("trajectory", [])))
    elif isinstance(data, list):
        messages = data

    commands = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content") or ""
            
            # OpenAI/vLLM tool calls
            if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
                for tc in msg["tool_calls"]:
                    try:
                        func = tc.get("function", {})
                        if func.get("name") in ["execute", "run_command", "execute_command", "run_bash", "bash"]:
                            args = json.loads(func.get("arguments", "{}"))
                            if "command" in args:
                                commands.append(args["command"])
                    except Exception:
                        pass
                        
            # Standard markdown blocks
            for prefix in ["command:", "```mswea_bash_command", "```bash", "```sh"]:
                if prefix in content:
                    parts = content.split(prefix)
                    for part in parts[1:]:
                        cmd = part.split("```")[0].strip() if "```" in prefix else part.strip()
                        if cmd:
                            commands.append(cmd)

        # Custom Action List
        elif isinstance(msg, dict) and "action" in msg:
            action = msg["action"]
            if isinstance(action, str):
                commands.append(action)
            elif isinstance(action, dict) and "command" in action:
                commands.append(action["command"])

    return [cmd for cmd in commands if cmd.strip()]

def get_task_metadata() -> Dict[str, str]:
    """Loads SWE-bench Verified metadata from Hugging Face to get base_commits."""
    print("Retrieving SWE-bench Verified dataset from Hugging Face...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return {item["instance_id"]: item["base_commit"] for item in dataset}

def process_trajectory(traj_path: str, actions: List[int], metadata: Dict[str, str], worker_status: dict) -> Dict[str, Any]:
    pid = os.getpid()
    basename = os.path.basename(traj_path)
    instance_id = basename.replace(".traj", "").replace(".json", "")
    
    def update_status(status_str: str):
        if worker_status is not None:
            worker_status[pid] = {"instance_id": instance_id, "status": status_str, "timestamp": time.time()}
            
    update_status("Initializing containerized replay")
    
    if instance_id not in metadata:
        return {"instance_id": instance_id, "error": "Not found in dataset", "patches": None}
        
    base_commit = metadata[instance_id]
    commands = parse_trajectory_commands(traj_path)
    
    if not commands:
        return {"instance_id": instance_id, "error": "No bash commands found", "patches": None}
        
    image_name = f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{instance_id}:latest".lower()
    container_name = f"replay_{instance_id.replace('-', '_')}_{uuid.uuid4().hex[:6]}"
    patches_by_action = {}
    
    try:
        update_status("Starting Podman/Docker container")
        # Start a persistent background container
        start_res = subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", container_name, image_name, "tail", "-f", "/dev/null"],
            capture_output=True, text=True
        )
        if start_res.returncode != 0:
            return {"instance_id": instance_id, "error": f"Failed to start container: {start_res.stderr}", "patches": None}

        subprocess.run(["docker", "exec", container_name, "bash", "-c", "git config --global user.email 'replay@agent.com' && git config --global user.name 'Agent'"], capture_output=True)

        update_status("Executing monolithic stateful replay script")
        
        # Command definitions that alter bash shell state and must not be run inside a timeout subshell
        STATE_COMMANDS = {
            "cd", "pushd", "popd",         # Directory navigation
            "export", "set", "unset",      # Variable management
            "source", ".", "eval",         # Execution in current context
            "alias", "unalias", "shopt",   # Shell behavior flags
            "conda", "nvm", "rvm"          # Environment managers (often shell functions)
        }
        
        # Build the master bash script
        script_lines = ["#!/bin/bash", "set +e", "cd /testbed"]
        for i, cmd in enumerate(commands):
            first_word = cmd.strip().split()[0] if cmd.strip() else ""
            
            # Execute state-altering commands directly, wrap others in a 30s timeout to prevent test hangs
            if first_word in STATE_COMMANDS:
                script_lines.append("{\n" + cmd + "\n} || true")
            else:
                script_lines.append(f"timeout 300s bash -c {shlex.quote(cmd)} || true")
                
            # Inject checkpoint extractions cleanly in the middle of the script execution
            current_action = i + 1
            if current_action in actions:
                script_lines.append("git add -A")
                script_lines.append(f"git diff --cached {base_commit} > /tmp/patch_action_{current_action}.diff")
                script_lines.append("git reset HEAD") # Unstage to prevent interference with future commands

        script_lines.append("git add -A")
        script_lines.append(f"git diff --cached {base_commit} > /tmp/patch_final.diff")
        script_lines.append("git reset HEAD")

        master_script = "\n".join(script_lines) + "\n"
        
        # FIX: Added a 30-minute overall timeout to prevent orphaned background jobs from hanging Python forever
        try:
            subprocess.run(
                ["docker", "exec", "-i", container_name, "bash"],
                input=master_script, text=True, capture_output=True, timeout=1800, errors="replace"
            )
        except subprocess.TimeoutExpired:
            update_status("Trajectory timed out, extracting available patches")
            pass # Proceed to extraction; any patches saved before the hang will still be captured!
        
        update_status("Extracting generated checkpoint patches")
        # Retrieve the generated diff files
        for a in actions:
            target_file = f"/tmp/patch_action_{a}.diff" if a <= len(commands) else "/tmp/patch_final.diff"
            
            cat_res = subprocess.run(["docker", "exec", container_name, "cat", target_file], capture_output=True, text=True, errors="replace")
            if cat_res.returncode == 0:
                patches_by_action[a] = {
                    "instance_id": instance_id,
                    "model_patch": cat_res.stdout,
                    "model_name_or_path": f"mini-swe-agent-action-{a}"
                }
            
    except Exception as e:
        return {"instance_id": instance_id, "error": str(e), "patches": None}
    finally:
        update_status("Destroying container")
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        
        if worker_status is not None and pid in worker_status:
            try:
                del worker_status[pid]
            except Exception:
                pass
            
    return {"instance_id": instance_id, "error": None, "patches": patches_by_action}

def monitor_workers(worker_status: dict, interval: int = 15):
    while True:
        time.sleep(interval)
        try:
            active = list(worker_status.items())
        except Exception:
            continue
        if not active: continue
            
        print("\n" + "="*80)
        print(f"📊 Active Container Workers (at {time.strftime('%H:%M:%S')})")
        print("="*80)
        now = time.time()
        for pid, info in sorted(active, key=lambda x: x[1].get("timestamp", 0)):
            elapsed = now - info.get("timestamp", now)
            warn = " ⚠️ [LONG RUN]" if elapsed > 120 else ""
            print(f"  • PID {pid} | {info.get('instance_id')} | {info.get('status')} ({elapsed:.0f}s){warn}")
        print("="*80 + "\n", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_dir", type=str, required=True)
    parser.add_argument("--actions", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./replayed_predictions")
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args()
    
    actions = [int(a.strip()) for a in args.actions.split(",")]
    os.makedirs(args.output_dir, exist_ok=True)
    
    metadata = get_task_metadata()
    traj_files = [os.path.join(root, f) for root, _, files in os.walk(args.traj_dir) for f in files if f.endswith(".traj.json")]
    
    if not traj_files:
        print(f"No trajectories found in {args.traj_dir}")
        sys.exit(1)
        
    predictions_by_action = {a: [] for a in actions}
    completed_ids_per_action = {a: set() for a in actions}
    
    for a in actions:
        out_file = os.path.join(args.output_dir, f"preds_action_{a}.json")
        if os.path.exists(out_file):
            try:
                with open(out_file, "r") as f:
                    existing = json.load(f)
                predictions_by_action[a] = existing
                for e in existing: completed_ids_per_action[a].add(e["instance_id"])
            except Exception:
                pass
                
    completed_ids = set.intersection(*[completed_ids_per_action[a] for a in actions]) if actions else set()
    for a in actions:
        predictions_by_action[a] = [p for p in predictions_by_action[a] if p["instance_id"] in completed_ids]
        
    pending_files = [p for p in traj_files if os.path.basename(p).replace(".traj.json", "") not in completed_ids]
    
    if not pending_files:
        print("All trajectories already processed.")
        sys.exit(0)

    print(f"🚀 {len(completed_ids)} done, {len(pending_files)} remaining. Running containers on {args.max_workers} workers...")
    
    import multiprocessing
    manager = multiprocessing.Manager()
    worker_status = manager.dict()
    
    threading.Thread(target=monitor_workers, args=(worker_status, 15), daemon=True).start()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_trajectory, p, actions, metadata, worker_status) for p in pending_files]
        for idx, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                res = future.result()
                if res["error"]:
                    print(f"[{idx+1}/{len(pending_files)}] ⚠️ {res['instance_id']}: {res['error']}")
                else:
                    print(f"[{idx+1}/{len(pending_files)}] ✅ {res['instance_id']}")
                    for a, patch_dict in res["patches"].items():
                        predictions_by_action[a].append(patch_dict)
                        
                for a in actions:
                    with open(os.path.join(args.output_dir, f"preds_action_{a}.json"), "w") as f:
                        json.dump(predictions_by_action[a], f, indent=4)
            except Exception as e:
                print(f"Worker crashed: {e}")

    print("\n✅ Successfully finished containerized trajectory replay!")

if __name__ == "__main__":
    main()

EOF
```

`plot_comparison.py` — python script to visualize the benchmarking curves:

```bash
cd $HOME/projects/swe-bench/SWE-bench/
cat << 'EOF' > plot_comparison.py
#!/usr/bin/env python3
"""
plot_comparison.py

Generates a NeurIPS-compliant single-column comparative line graph 
showing the SWE-bench Verified optimization curves over action counts.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def hash_text(text: str):
    hash_val = 0
    for ch in text:
        hash_val = (hash_val * 281 ^ ord(ch) * 997) & 0xFFFFFFFF 
    return hash_val

def get_line_format(post_gen_selection_method: str, feedback_method: str):
    rng = np.random.default_rng(hash_text(post_gen_selection_method + feedback_method) + 1)
    color = {
        'last': 'tab:gray',
        'mbr_hard': 'tab:blue',
        'mbr_soft': 'tab:orange',
        'er_hard': 'tab:red',
        'er_soft': 'tab:green',
        'codet_hard': 'tab:cyan',
        'codet_soft': 'tab:pink',
        'Oracle-Test Accuracy with Oracle-Test Feedback': 'm',
        'Oracle-Test Accuracy with Self-Test Feedback': 'k',
        'Self-Test Accuracy with Self-Test Feedback': 'c',
        'QiskitHumanEvalSim': 'k',
        'QiskitHumanEvalSimX': 'r',
        'Qwen3 (235B-A22B-Instruct)': '#1D1C1A',
        'Qwen3-Next (80B-A3B-Instruct)': "#2E9CC5",
        'Qwen3-Coder-Next (80B-A3B-Instruct)': "#2DAB2D",
    }
    random_color = rng.choice(matplotlib.pyplot.rcParams['axes.prop_cycle'].by_key()['color'])
    
    line_style = {
        'codes_on_tests': '-',
        'thompson_sampling_tf': '--',
        'thompson_sampling_cf': ':',
        'thompson_sampling_cfs': '-.',
        'last': '--',
        'mbr_hard': '-',
        'mbr_soft': ':',
        'er_hard': '-',
        'er_soft': '-.',
        'codet_hard': '--',
        'codet_soft': ':',
        'Oracle-Test Accuracy with Oracle-Test Feedback': ':',
        'Oracle-Test Accuracy with Self-Test Feedback': '-',
        'Self-Test Accuracy with Self-Test Feedback': '--',
        'QiskitHumanEvalSim': '-',
        'QiskitHumanEvalSimX': ':',
        'Qwen3 (235B-A22B-Instruct)': '-',
        'Qwen3-Next (80B-A3B-Instruct)': "--",
        'Qwen3-Coder-Next (80B-A3B-Instruct)': ":",
    }
    random_line_style = rng.choice(['-', '--', '-.', ':'])

    return {
        'color': color.get(post_gen_selection_method, random_color), 
        'linestyle': line_style.get(feedback_method, random_line_style)
    }

def get_score(run_id: str) -> float:
    """Extracts the success rate percentage from a SWE-bench summary JSON file."""
    action = run_id.split("-")[-1]
    report_path = f"mini-swe-agent-action-{action}.{run_id}.json"
    if not os.path.exists(report_path):
        return 0.0
        
    try:
        with open(report_path, "r") as f:
            data = json.load(f)
        resolved = int(data.get("resolved_instances", 0))
        total = int(data.get("submitted_instances", data.get("total_instances", 500)))
        return round((resolved / total) * 100, 2) if total > 0 else 0.0
    except Exception:
        return 0.0

def calculate_smooth_offsets(x_arr, y_arr, idx, offset_dist=7.5):
    """
    Calculates an offset vector perpendicular to the local tangent vector of the curve.
    Adjusts for the visual aspect ratio mismatch between X (0-250) and Y (0-100) scales.
    """
    # 1. Estimate local delta using central, forward, or backward differences
    if idx == 0:
        dx = x_arr[1] - x_arr[0]
        dy = (y_arr[1] - y_arr[0]) * 2.5 # Scale Y to conceptually match 250 max-width visual spacing
    elif idx == len(x_arr) - 1:
        dx = x_arr[-1] - x_arr[-2]
        dy = (y_arr[-1] - y_arr[-2]) * 2.5
    else:
        dx = x_arr[idx+1] - x_arr[idx-1]
        dy = (y_arr[idx+1] - y_arr[idx-1]) * 2.5
    
    # 2. Derive normal unit vector pointing perpendicular to the curve orientation
    tangent = np.array([dx, dy])
    norm = np.linalg.norm(tangent)
    
    if norm == 0:
        return (0, offset_dist), (0, -offset_dist)
    
    tangent_unit = tangent / norm
    normal_unit = np.array([-tangent_unit[1], tangent_unit[0]])
    
    # 3. Dynamic text anchors to keep labels tightly wrapped but completely legible
    angle = np.arctan2(normal_unit[1], normal_unit[0])
    
    if np.abs(angle) > np.pi * 0.75:
        r_ha, r_va = 'right', 'center'
        s_ha, s_va = 'left', 'center'
    elif np.abs(angle) < np.pi * 0.25:
        r_ha, r_va = 'left', 'center'
        s_ha, s_va = 'right', 'center'
    elif angle >= 0:
        r_ha, r_va = 'center', 'bottom'
        s_ha, s_va = 'center', 'top'
    else:
        r_ha, r_va = 'center', 'top'
        s_ha, s_va = 'center', 'bottom'

    # Standard setting points down (-normal), Reproduction points up (+normal)
    r_offset = normal_unit * offset_dist
    s_offset = -normal_unit * offset_dist
    
    return (s_offset[0], s_offset[1]), s_ha, s_va, (r_offset[0], r_offset[1]), r_ha, r_va

def main():
    actions_file = os.path.join("replayed_predictions", "actions.txt")
    if not os.path.exists(actions_file):
        print(f"❌ Error: {actions_file} not found. Run ./replay_trajectories.sh first.")
        sys.exit(1)
    with open(actions_file, "r") as f:
        actions = [int(a.strip()) for a in f.read().strip().split(",")]

    standard_scores = [get_score(f"mini-swe-agent-verified-standard-action-{a}") for a in actions]
    reproduction_scores = [get_score(f"mini-swe-agent-verified-reproduction-action-{a}") for a in actions]

    x_data = np.array(actions)
    y_std = np.array(standard_scores)
    y_rep = np.array(reproduction_scores)

    # --- NeurIPS Publication Styling Setup ---
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'text.usetex': False, 
        'axes.edgecolor': '#333333',
        'axes.linewidth': 0.8,
    })
    
    fig, ax = plt.subplots(figsize=(5.5, 3.8), dpi=300)
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')

    # Standard Setting Format Configuration (Dark Black baseline)
    standard_fmt = get_line_format('last', 'last') 
    standard_color = '#1D1C1A'
    ax.plot(
        x_data, y_std, 
        marker='o', markersize=4, linewidth=1.5, 
        color=standard_color, linestyle=standard_fmt['linestyle'],
        label='Standard Setting (No Oracle)'
    )

    # Reproduction Setting Format Configuration
    reproduction_fmt = get_line_format('Oracle-Test Accuracy with Oracle-Test Feedback', 'Oracle-Test Accuracy with Oracle-Test Feedback')
    ax.plot(
        x_data, y_rep, 
        marker='s', markersize=4, linewidth=1.5, 
        color=reproduction_fmt['color'], linestyle=reproduction_fmt['linestyle'],
        label='Reproduction Setting (Oracle Feedback)'
    )

    # Axis Labels Configuration
    ax.set_xlabel('Action Count (Discrete Actions Executed)', fontsize=10, labelpad=6)
    ax.set_ylabel('SWE-bench Verified Score (%)', fontsize=10, labelpad=6)

    ax.set_xlim(0, max(actions) + 5)
    ax.set_ylim(0, 100)
    ax.set_xticks(actions)
    ax.tick_params(axis='both', which='major', labelsize=8.5, colors='#333333')
    
    ax.grid(True, linestyle='--', linewidth=0.5, color='#cccccc', alpha=0.7)
    ax.legend(loc='lower right', frameon=True, facecolor='#ffffff', edgecolor='#d1d5db', fontsize=8)

    # White box styling with a compact padding constraint
    bbox_props = dict(boxstyle="square,pad=0.1", fc="#ffffff", ec="none", alpha=0.85)

    # Smooth label alignment loop
    for i in range(len(x_data)):
        if i < 4 or (i - 4) % 2 == 1:
            a = x_data[i]
            s_score = y_std[i]
            r_score = y_rep[i]
            
            # Tightened distance scalar (7.5) moves data strings noticeably closer to dot coordinates
            s_xy, s_ha, s_va, r_xy, r_ha, r_va = calculate_smooth_offsets(x_data, y_rep, i, offset_dist=7.5)

            if s_score > 0:
                ax.annotate(
                    f"{s_score}%", (a, s_score), textcoords="offset points", 
                    xytext=s_xy, ha=s_ha, va=s_va,
                    fontsize=7, fontweight='bold', color='#2C3E50',
                    bbox=bbox_props
                )
            if r_score > 0:
                ax.annotate(
                    f"{r_score}%", (a, r_score), textcoords="offset points", 
                    xytext=r_xy, ha=r_ha, va=r_va,
                    fontsize=7, fontweight='bold', color='#5C1E36',
                    bbox=bbox_props
                )

    plt.tight_layout()
    output_filename = 'scores_comparison.pdf'
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.02)
    print(f"🎉 Successfully saved automated comparative plot to: '{os.path.abspath(output_filename)}'")

if __name__ == "__main__":
    main()

EOF
source .venv/bin/activate
uv pip install matplotlib
```

</details>

#### 🔧 Step 6.2: Create the Replay Scripts (Done Once)

[WARNING] Any batch job running these replay or evaluation scripts **must** be submitted with exclusive node access to prevent cross-job container assassination and GPFS lock collisions!

Run the following code to create the replay script:

<details><summary>code</summary>

<br>

`replay_trajectories.sh` — creates all intermediate states for trajectory replay:

```bash
cd $HOME/projects/swe-bench/SWE-bench/
cat << 'EOF' > replay_trajectories.sh
#!/bin/bash
# replay_trajectories.sh (Containerized Version)

set -e
source .venv/bin/activate

ACTIONS="${1:-10,20,30,40,50,60,80,100,120,140,170,200,230,250}"

echo "=========================================================="
echo "🚀 Starting Containerized Trajectory Replay"
echo "=========================================================="

if ! python3 -c "import datasets" &> /dev/null; then
    uv pip install datasets
fi

# IMPORTANT: Hard limit to 16 workers to prevent container locks
MAX_WORKERS=16
echo "Using ${MAX_WORKERS} parallel container workers"

echo -e "\n🔄 Replaying Standard Setting Trajectories..."
python3 -u extract_action_patches.py \
  --traj_dir ../mini-swe-agent/evaluation_output_standard \
  --actions "${ACTIONS}" \
  --output_dir ./replayed_predictions/standard \
  --max_workers ${MAX_WORKERS}

echo -e "\n🔄 Replaying Reproduction Setting Trajectories..."
python3 -u extract_action_patches.py \
  --traj_dir ../mini-swe-agent/evaluation_output_reproduction \
  --actions "${ACTIONS}" \
  --output_dir ./replayed_predictions/reproduction \
  --max_workers ${MAX_WORKERS}

echo "${ACTIONS}" > ./replayed_predictions/actions.txt
echo -e "\n✅ Done!"
EOF
chmod +x replay_trajectories.sh
```

`evaluate_actions.sh` — batch-evaluates all extracted action predictions:
```bash
cd $HOME/projects/swe-bench/SWE-bench
cat << 'EOF' > evaluate_actions.sh
#!/bin/bash
# evaluate_actions.sh
# Runs batch evaluations in parallel over all extracted action prediction files.
# Reads action values dynamically from replayed_predictions/actions.txt.

set -e

# Automatically clean up Podman containers and catatonit zombies if LSF kills the job
cleanup() {
    echo "Job killed! Cleaning up Podman containers..."
    podman rm -fa 2>/dev/null || true
    pkill -u "$(id -u)" -f catatonit || true
}
trap cleanup EXIT SIGINT SIGTERM

# Auto-load the cluster environment, Podman socket, and path settings
source "$(dirname "$0")/../load_env.sh" >/dev/null 2>&1 || source "$(dirname "$0")/load_env.sh" >/dev/null 2>&1 || true
source .venv/bin/activate

MAX_WORKERS=${LSB_DJOB_NUMPROC:-48}

echo "=========================================================="
echo "🚀 Starting Full Action Sweeps Evaluation with ${MAX_WORKERS} Workers"
echo "=========================================================="

# Read actions dynamically or from arguments
if [ "$#" -eq 1 ]; then
  IFS=',' read -ra ACTIONS <<< "$1"
else
  ACTIONS_FILE="./replayed_predictions/actions.txt"
  if [ ! -f "${ACTIONS_FILE}" ]; then
    echo "❌ Error: ${ACTIONS_FILE} not found. Run ./replay_trajectories.sh first."
    exit 1
  fi
  # Convert comma-separated actions into a bash array
  IFS=',' read -ra ACTIONS < "${ACTIONS_FILE}"
fi
echo "Evaluating actions: ${ACTIONS[*]}"

# Auto-detect allocated cores from LSF to avoid Errno 11
MAX_WORKERS=${LSB_DJOB_NUMPROC:-48}
echo "Using ${MAX_WORKERS} parallel workers for Podman containers"

# 1. Evaluate Standard Predictions
echo -e "\n📊 Evaluating Standard Setting Predictions..."
for a in "${ACTIONS[@]}"; do
  echo "----------------------------------------------------------"
  echo "Running evaluation for Standard setting at Action ${a}..."
  echo "----------------------------------------------------------"
  python3 -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path ./replayed_predictions/standard/preds_action_${a}.json \
    --max_workers ${MAX_WORKERS} \
    --run_id mini-swe-agent-verified-standard-action-${a}
done

# 2. Evaluate Reproduction Predictions
echo -e "\n📊 Evaluating Reproduction Setting Predictions..."
for a in "${ACTIONS[@]}"; do
  echo "----------------------------------------------------------"
  echo "Running evaluation for Reproduction setting at Action ${a}..."
  echo "----------------------------------------------------------"
  python3 -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path ./replayed_predictions/reproduction/preds_action_${a}.json \
    --max_workers ${MAX_WORKERS} \
    --run_id mini-swe-agent-verified-reproduction-action-${a}
done

echo -e "\n✅ Finished all action-based evaluations!"
echo "Check output logs in 'logs/run_evaluation' to collect final scores."
EOF
chmod +x evaluate_actions.sh
```

</details>

#### 🚀 Step 6.3: Run the Replay Workflow (Done after each run)

**1. Extract action-level patches:**
*(Patches are extracted to `./replayed_predictions/standard/` and `./replayed_predictions/reproduction/`)*
```bash
cd $HOME/projects/swe-bench/SWE-bench
source .venv/bin/activate

./replay_trajectories.sh                    # default actions: 10,20,30,40,50,60,80,100,120,140,170,200,230,250
#./replay_trajectories.sh 5,10,15,20,25,30  # or pass custom actions
```
The chosen actions are persisted to `./replayed_predictions/actions.txt` and automatically picked up by all downstream scripts.

**2. Evaluate all action predictions:**
*(Evaluation reports and scores are saved in `logs/run_evaluation/`)*

```bash
cd $HOME/projects/swe-bench/SWE-bench
source .venv/bin/activate

./evaluate_actions.sh
```

**3. Plot the optimization curves:**
*(Output saved to `scores_comparison.pdf`)*

```bash
cd $HOME/projects/swe-bench/SWE-bench
source .venv/bin/activate

python plot_comparison.py
```
This automatically parses the evaluation summary `*.json` files generated in your current directory and creates `scores_comparison.pdf` comparing standard vs. reproduction success rates across action counts.

---

## 🔗 Official References & Resources

*   **Agentic System**: [mini-swe-agent Documentation](https://mini-swe-agent.com/latest/)
*   **Docker Optimization Post**: [Epoch AI - Running SWE-bench in One Hour](https://epoch.ai/blog/swebench-docker#impact-size)
*   **Image Packages**: [Epoch Research SWE-bench GHCR Registry](https://github.com/orgs/epoch-research/packages?repo_name=SWE-bench)
*   **Benchmark Documentation**: [Official SWE-bench Site](https://www.swebench.com)
*   **Official Benchmark Repository**: [Princeton NLP - SWE-bench Harness](https://github.com/princeton-nlp/SWE-bench)
