# Ollama Local Privacy Guide

> 🔐 **BONUS MODULE — PROMPT 0.5.1 (Part B)**
> Not required to complete any feature. Adds Ollama privacy hardening
> for students building with sensitive data or in regulated environments.

## Is your model actually running locally?

### The critical distinction (Ollama v0.12+)

| Model tag | Where it runs | Data leaves machine? |
|-----------|--------------|----------------------|
| `ollama run llama3.2` | **Local** — your machine | **No** |
| `ollama run gpt-oss:120b-cloud` | Ollama's servers | **Yes** — prompts sent remotely |

The `-cloud` suffix is the signal. Without it, standard Ollama models run
entirely on your hardware. With it, your prompts leave your machine.

### Verify it yourself — three methods

**Method 1 (simplest):** Pull your model, disconnect WiFi, run a prompt.
If it works normally → confirmed local.

```bash
ollama pull llama3.2
# disconnect network
ollama run llama3.2 "What is 2 + 2?"
# works offline → confirmed local inference
```

**Method 2 (packet capture):**
```bash
# macOS — watch for outbound traffic while running a prompt
sudo tcpdump -i en0 -nn 'not port 443 and host not 239.255.255.250'

# Linux
sudo tcpdump -i eth0 -nn 'not port 443'
```
You should see zero traffic during inference. Traffic only appears when
pulling models (`ollama pull ...`), not during chat.

**Method 3 (firewall rule):**
```powershell
# Windows PowerShell — block ALL Ollama outbound
New-NetFirewallRule -DisplayName "Block Ollama Outbound" `
  -Direction Outbound `
  -Program "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" `
  -Action Block

# Remove when you need to pull new models
Remove-NetFirewallRule -DisplayName "Block Ollama Outbound"
```

```bash
# Linux iptables — block HTTPS for your user only
sudo iptables -A OUTPUT -p tcp -m owner --uid-owner $(id -u) \
  --dport 443 -j DROP
```

---

## Disable history logging

Ollama stores your chat history in **plain text** at:

| Platform | Path |
|----------|------|
| macOS/Linux | `~/.ollama/history` |
| Windows | `%LOCALAPPDATA%\Ollama\history.txt` |

For sensitive workloads, disable it before starting Ollama:

```bash
# One session
export OLLAMA_KEEP_HISTORY=false && ollama serve

# Permanent (add to ~/.zshrc or ~/.bashrc)
echo 'export OLLAMA_KEEP_HISTORY=false' >> ~/.zshrc
```

You can also set this in your `.env` alongside the bootcamp flag:
```
ENABLE_OLLAMA_PRIVACY_CHECKS=true
```

When `ENABLE_OLLAMA_PRIVACY_CHECKS=true`, the server logs a warning at
startup if `OLLAMA_KEEP_HISTORY` has not been disabled.

---

## Team / organisational deployment

If running Ollama on a shared GPU server, **never expose port 11434 directly
to the internet** — there is no built-in authentication.

**Option A: Nginx reverse proxy with Basic Auth**
```nginx
server {
    listen 443 ssl;
    ssl_certificate     /etc/letsencrypt/live/your-domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain/privkey.pem;

    location / {
        auth_basic           "Ollama";
        auth_basic_user_file /etc/nginx/.htpasswd;
        proxy_pass           http://127.0.0.1:11434;
        proxy_set_header     Host $host;
    }
}
```

Create the password file: `htpasswd -c /etc/nginx/.htpasswd yourname`

**Option B: SSH tunnelling (simplest for dev teams)**
```bash
# Developer runs this on their laptop — connects through SSH, no port exposure
ssh -L 11434:localhost:11434 user@gpu-server

# App.js / .env point at localhost:11434 — encrypted via SSH
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Air-gapped deployment (HIPAA / GDPR / regulated industries)

For environments where **no internet access** is permitted on the production
machine:

```bash
# Step 1 — on an internet-connected machine, pull the model
ollama pull llama3.2

# Step 2 — locate the model files
# macOS/Linux: ~/.ollama/models/
# Windows:     %USERPROFILE%\.ollama\models\

# Step 3 — archive them
tar czf ollama-models.tar.gz ~/.ollama/models/

# Step 4 — transfer via approved media (USB, internal SFTP, etc.)

# Step 5 — on the air-gapped machine, extract to the same path
mkdir -p ~/.ollama
tar xzf ollama-models.tar.gz -C ~/

# Step 6 — verify (Ollama finds models by blob hash, not filename)
ollama list
```

**Why this matters for compliance:**
- Local inference = **no multi-party data processing agreement** needed under GDPR
- HIPAA: no PHI leaves the system boundary
- Document this in your compliance audit trail: list model name, version,
  and the date you transferred it

---

## What the startup privacy check does

When `ENABLE_OLLAMA_PRIVACY_CHECKS=true` in `.env`, the Ollama provider
logs these warnings at server startup:

| Condition | Warning |
|-----------|---------|
| Model name contains `-cloud` | Your prompts will leave the machine |
| `OLLAMA_KEEP_HISTORY` not set | Chat history is stored in plain text |

These are informational warnings, not errors — the server still starts.
They are designed to surface easy-to-miss privacy gotchas before the
first real query lands, not to block usage.

---

## Quick reference

```bash
# Check which model you're running
ollama list

# Confirm a model has no cloud suffix
ollama show llama3.2  # shows model details including parameters file

# Run without history logging
OLLAMA_KEEP_HISTORY=false ollama serve

# Verify network isolation during inference (macOS)
sudo tcpdump -i en0 -c 20
# should show nothing when chatting (only shows traffic on model pull)
```
