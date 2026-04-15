# scilab-rag — Production Deployment Plan

## Architecture Overview

scilab-rag runs on a dedicated **VPS 3**, separate from the scilab-microservices infrastructure.
It connects to VPS 1 (infra) over WireGuard for shared services (PostgreSQL, RabbitMQ).
Neo4j lives locally on VPS 3 because it is only used by the RAG pipeline.

```
[Local Machine]
     │
     │ SSH tunnel
     ▼
[VPS 1 — Infrastructure]  10.13.13.1
  ├── PostgreSQL  :5432
  ├── Redis       :6379
  ├── RabbitMQ    :5672
  └── Keycloak, MinIO, etc.
     ▲
     │ WireGuard tunnel (UDP 51820)
     │
[VPS 3 — scilab-rag]  10.13.13.3
  ├── scilab-rag  :8080  (FastAPI, uvicorn)
  └── Neo4j       :7687  (bolt, local only)
```

---

## What Is New vs scilab-microservices

| Aspect | scilab-microservices | scilab-rag |
|--------|----------------------|------------|
| Language | .NET | Python 3.11 |
| Services | Multiple (6+) | Single app + Neo4j |
| Graph DB | None | **Neo4j 5** (new) |
| WireGuard role | VPS 2 peer | **VPS 3 peer** (new) |
| Migrations | EF Core | **Alembic** (PostgreSQL) |

---

## 1. VPS 3 Provisioning

### Cloud-Init (first boot)

```yaml
#cloud-config
package_update: true
packages:
  - git
  - ufw
  - curl
  - wireguard
  - wireguard-tools
runcmd:
  # Docker
  - install -m 0755 -d /etc/apt/keyrings
  - curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  - chmod a+r /etc/apt/keyrings/docker.asc
  - |
    tee /etc/apt/sources.list.d/docker.sources <<EOF
    Types: deb
    URIs: https://download.docker.com/linux/ubuntu
    Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
    Components: stable
    Signed-By: /etc/apt/keyrings/docker.asc
    EOF
  - apt update
  - apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  - systemctl enable docker
  - systemctl start docker
  # WireGuard
  - echo "net.ipv4.ip_forward=1" >> /etc/sysctl.conf
  - sysctl -p
  - mkdir -p /etc/wireguard
  - systemctl enable wg-quick@wg0
  # Firewall — only SSH and app port exposed publicly
  - ufw allow OpenSSH
  - ufw allow 80
  - ufw allow 443
  - ufw --force enable
  # Clone repo
  - mkdir -p /root/scilab
  - git clone https://github.com/scilab-org/scilab-rag.git /root/scilab/scilab-rag
```

---

## 2. WireGuard — Add VPS 3 as Peer on VPS 1

Run **on VPS 1** to generate a key pair for VPS 3 and add it as a peer:

```bash
# Generate VPS 3 key pair
wg genkey | tee /etc/wireguard/vps3_private.key | wg pubkey > /etc/wireguard/vps3_public.key
chmod 600 /etc/wireguard/vps3_private.key

VPS3_PRIVATE=$(cat /etc/wireguard/vps3_private.key)
VPS3_PUBLIC=$(cat /etc/wireguard/vps3_public.key)
VPS1_PUBLIC=$(cat /etc/wireguard/vps1_public.key)

# Add VPS 3 as a peer to VPS 1's running tunnel (no restart needed)
wg set wg0 peer "$VPS3_PUBLIC" allowed-ips 10.13.13.3/32

# Persist the peer in wg0.conf
cat >> /etc/wireguard/wg0.conf <<EOF

[Peer]
# VPS 3 — scilab-rag
PublicKey = ${VPS3_PUBLIC}
AllowedIPs = 10.13.13.3/32
EOF

# Generate the config to copy to VPS 3
cat > /etc/wireguard/vps3_wg0.conf <<EOF
[Interface]
Address = 10.13.13.3/24
PrivateKey = ${VPS3_PRIVATE}

[Peer]
# VPS 1
PublicKey = ${VPS1_PUBLIC}
Endpoint = $(curl -4 -s ifconfig.me):51820
AllowedIPs = 10.13.13.1/32
PersistentKeepalive = 25
EOF

echo "=== Copy this to VPS 3 as /etc/wireguard/wg0.conf ==="
cat /etc/wireguard/vps3_wg0.conf
```

### Allow VPS 3 to reach infra services on VPS 1

```bash
# On VPS 1 — extend UFW to allow VPS 3's WireGuard IP
sudo ufw allow in on wg0 from 10.13.13.3 to any port 5432  # PostgreSQL
sudo ufw allow in on wg0 from 10.13.13.3 to any port 5672  # RabbitMQ
sudo ufw reload
```

> The existing rules from the scilab-microservices setup allow `any` on wg0.
> These scoped rules are additive — no existing VPS 2 traffic is affected.

### Start WireGuard on VPS 3

```bash
# On VPS 3 — paste the vps3_wg0.conf content
sudo nano /etc/wireguard/wg0.conf

sudo wg-quick up wg0
sudo systemctl enable wg-quick@wg0

# Verify
sudo wg show
ping 10.13.13.1
```

---

## 3. Environment Variables

Create `/root/scilab/scilab-rag/.env` on VPS 3. All values are required unless noted.

```env
# ── OpenRouter ──────────────────────────────────────────────────────────────
OPENROUTER_API_KEY=
OPENROUTER_API_URL_BASE=https://openrouter.ai/api/v1
OPENROUTER_API_URL_CHAT=https://openrouter.ai/api/v1/chat/completions

# Per-model API keys (can reuse OPENROUTER_API_KEY for all if on one plan)
OPEN_ROUTER_API_KEY_EMBED_MODEL=
OPEN_ROUTER_API_KEY_IMAGE_MODEL=
OPEN_ROUTER_API_KEY_SUMMARY_MODEL=
OPEN_ROUTER_API_KEY_CHAT_MODEL=
OPEN_ROUTER_API_KEY_EXTRACT_MODEL=

# ── URLs ────────────────────────────────────────────────────────────────────
FRONTEND_URL=https://<your-frontend-domain>
GATEWAY_URL=https://<your-api-gateway-domain>

# ── Neo4j (local — Docker service name) ─────────────────────────────────────
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<strong-random-password>
NEO4J_DATABASE=neo4j

# ── PostgreSQL (VPS 1 via WireGuard) ────────────────────────────────────────
POSTGRESQL_URI=postgresql://<user>:<password>@10.13.13.1:5432/<dbname>

# ── RabbitMQ (VPS 1 via WireGuard) ──────────────────────────────────────────
RABBITMQ_URI=amqp://<user>:<password>@10.13.13.1:5672/

# RabbitMQ topology — must match MassTransit topology in scilab-microservices
RABBITMQ_INGEST_EXCHANGE=EventSourcing.Events.Lab:PaperIngestionEvent
RABBITMQ_INGEST_QUEUE=paper-ingestion
RABBITMQ_COMPLETED_EXCHANGE=EventSourcing.Events.Lab:PaperIngestionCompletedEvent

# ── Keycloak (VPS 1 via WireGuard, or public URL) ───────────────────────────
KEYCLOAK_SERVER_URL=
KEYCLOAK_REALM=
KEYCLOAK_CLIENT_ID=
KEYCLOAK_CLIENT_SECRET=
```

> **Note on Neo4j URI:** Inside the Docker Compose network, the app reaches Neo4j via
> the service name `neo4j`, not `localhost`. The URI must be `bolt://neo4j:7687`.

---

## 4. First Deploy (manual)

```bash
# On VPS 3
cd /root/scilab/scilab-rag

# Copy .env (do not commit secrets)
nano .env   # fill in all values above

# Start Neo4j first and wait for it to be healthy
docker compose up -d neo4j
docker compose ps   # wait until neo4j is healthy (~30s)

# Build the app image
docker compose build scilab-rag

# Run DB migrations (PostgreSQL via Alembic)
docker compose run --rm scilab-rag alembic upgrade head

# Start the app
docker compose up -d scilab-rag

# Verify
docker compose ps
docker compose logs -f scilab-rag
```

### Verify connectivity

```bash
# From VPS 3 — confirm WireGuard tunnel reaches infra services
nc -zv 10.13.13.1 5432   # PostgreSQL
nc -zv 10.13.13.1 5672   # RabbitMQ

# Neo4j browser (SSH port-forward locally)
# ssh -L 7474:127.0.0.1:7474 root@<VPS3_IP> -N
# then open http://localhost:7474
```

---

## 5. GitHub Actions CI/CD

Two workflows are included in `.github/workflows/`:

| Workflow | File | Trigger |
|----------|------|---------|
| Deploy | `deploy.yml` | Push to `main` (app/alembic/Dockerfile changes) |
| Docker Cleanup | `docker-cleanup.yml` | Weekly (Sunday 3AM UTC) |

### Required GitHub Secrets

Go to the repo → **Settings → Secrets and variables → Actions** and add:

| Secret | Value |
|--------|-------|
| `VPS_HOST` | VPS 3 public IP |
| `VPS_USER` | `root` (or your SSH user) |
| `VPS_SSH_KEY` | Contents of `~/.ssh/id_ed25519` (the private key) |
| `VPS_PORT` | `22` (or custom SSH port) |

### What the deploy workflow does on each push

1. SSH into VPS 3
2. `git reset --hard origin/main`
3. `docker compose build --no-cache scilab-rag`
4. `docker compose run --rm scilab-rag alembic upgrade head`
5. `docker compose up -d --no-deps scilab-rag`
6. Health check (30s wait, inspect container state)
7. Prune dangling images
8. Create a GitHub issue on failure

Neo4j is **not** rebuilt on each deploy — only the app image is rebuilt.

---

## 6. Missing Infrastructure Checklist

Before CI/CD can deploy, these must exist on VPS 1:

- [ ] PostgreSQL database created for scilab-rag (separate from microservices DB if needed)
- [ ] RabbitMQ user/vhost created for scilab-rag if using separate credentials
- [ ] WireGuard peer for VPS 3 added to `/etc/wireguard/wg0.conf` on VPS 1
- [ ] UFW rules on VPS 1 allow VPS 3 (`10.13.13.3`) on ports 5432 and 5672
- [ ] Keycloak client registered for scilab-rag (if using Keycloak auth)

On VPS 3:

- [ ] `/root/scilab/scilab-rag/.env` created with all values filled
- [ ] WireGuard configured and tunnel verified (`ping 10.13.13.1` works)
- [ ] SSH public key from GitHub Actions added to `~/.ssh/authorized_keys`
- [ ] Docker and Docker Compose installed

---

## 7. Step-by-Step Checklist (Do This Now)

### Phase 1 — Provision VPS 3

- [ ] Create VPS 3 (Ubuntu 22.04+) with cloud-init from section 1
- [ ] Note VPS 3's public IP

### Phase 2 — WireGuard

- [ ] On VPS 1: generate VPS 3 key pair and add peer (section 2)
- [ ] On VPS 1: add UFW rules for port 5432 and 5672 from `10.13.13.3`
- [ ] On VPS 3: paste `vps3_wg0.conf` → `/etc/wireguard/wg0.conf`
- [ ] On VPS 3: `wg-quick up wg0` and verify `ping 10.13.13.1`
- [ ] On VPS 3: verify `nc -zv 10.13.13.1 5432` and `nc -zv 10.13.13.1 5672`

### Phase 3 — Environment

- [ ] Create PostgreSQL DB for scilab-rag on VPS 1 (if not sharing with microservices)
- [ ] Fill in `.env` on VPS 3 (all fields in section 3)
- [ ] Confirm `NEO4J_URI=bolt://neo4j:7687` (service name, not localhost)

### Phase 4 — First Manual Deploy

- [ ] `docker compose up -d neo4j` and wait for healthy
- [ ] `docker compose build scilab-rag`
- [ ] `docker compose run --rm scilab-rag alembic upgrade head`
- [ ] `docker compose up -d scilab-rag`
- [ ] Confirm `docker compose ps` shows both containers healthy

### Phase 5 — CI/CD

- [ ] Add GitHub Secrets: `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`, `VPS_PORT`
- [ ] Push a commit to `main` touching `app/` to trigger the deploy workflow
- [ ] Verify the Actions run completes without error

---

## Troubleshooting

### App cannot reach Neo4j

```bash
# Check Neo4j is healthy
docker compose ps neo4j

# Test from inside the app container
docker compose exec scilab-rag python -c \
  "from neo4j import GraphDatabase; d=GraphDatabase.driver('bolt://neo4j:7687', auth=('neo4j','<pass>')); d.verify_connectivity(); print('OK')"
```

### App cannot reach PostgreSQL / RabbitMQ

```bash
# Check WireGuard is up
sudo wg show

# Test ports
nc -zv 10.13.13.1 5432
nc -zv 10.13.13.1 5672
```

### Alembic migration fails

```bash
# Run interactively to see full error
docker compose run --rm scilab-rag alembic upgrade head

# Check POSTGRESQL_URI in .env is correct and DB exists
docker compose run --rm scilab-rag python -c \
  "import os; print(os.getenv('POSTGRESQL_URI'))"
```

### DNS stops working after WireGuard starts

See the DNS troubleshooting section in the scilab-microservices WireGuard README — the fix is identical (scope the MASQUERADE rule or pin `/etc/resolv.conf`).

```bash
# Quick fix
sudo rm /etc/resolv.conf
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 1.1.1.1" | sudo tee -a /etc/resolv.conf
sudo chattr +i /etc/resolv.conf
```
