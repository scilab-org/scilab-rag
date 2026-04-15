# scilab-rag — Production Deployment Plan

## Architecture Overview

scilab-rag runs on a dedicated **VPS 3** behind nginx with TLS.
All external services are cloud-hosted or already running on VPS 1.

```
Internet
  │  :443
  ▼
[nginx]  ← TLS termination (Let's Encrypt, rag.hyperdatalab.site)
  │  :8080 (internal)
  ▼
[scilab-rag]  FastAPI / uvicorn

External services (configured via .env):
  ├── Neo4j Aura          — cloud-managed graph DB
  ├── PostgreSQL          — VPS 1 (direct connection)
  ├── RabbitMQ            — VPS 1 (direct connection)
  └── Keycloak            — https://auth.hyperdatalab.site
```

> **Security note:** PostgreSQL and RabbitMQ are currently accessed via VPS 1's public IP.
> Consider restricting those ports on VPS 1's firewall to VPS 3's IP only, or migrating
> to a WireGuard tunnel (see scilab-microservices deploy/prod/README.md for the pattern).

---

## File Structure

```
deploy/prod/
  docker-compose.yml        — production compose (certbot + nginx + app)
  .env                      — live secrets (gitignored, never commit)
  .env.example              — template to copy when setting up a new VPS
  nginx/
    nginx.conf              — nginx reverse proxy + TLS config
  docker-volumes/           — created at runtime, gitignored
    nginx/certbot/conf/     — Let's Encrypt certificates
    nginx/cache/            — nginx cache
    nginx/log/              — nginx access/error logs
```

The `Dockerfile` lives at the **repo root**. The compose `build.context` is set to `../../`
so Docker resolves it correctly from this directory.

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
  # Firewall
  - ufw allow OpenSSH
  - ufw allow 80
  - ufw allow 443
  - ufw --force enable
  # Clone repo
  - mkdir -p /root/scilab
  - git clone https://github.com/scilab-org/scilab-rag.git /root/scilab/scilab-rag
```

---

## 2. Environment Variables

Copy `.env.example` to `.env` and fill in all values:

```bash
cp /root/scilab/scilab-rag/deploy/prod/.env.example \
   /root/scilab/scilab-rag/deploy/prod/.env
nano /root/scilab/scilab-rag/deploy/prod/.env
```

| Variable | Where to get it |
|----------|-----------------|
| `RAG_DOMAIN` | Your DNS record for scilab-rag (e.g. `rag.hyperdatalab.site`) |
| `CERTBOT_EMAIL` | Email for Let's Encrypt expiry notices |
| `OPENROUTER_API_KEY` | OpenRouter dashboard |
| `OPEN_ROUTER_API_KEY_*` | OpenRouter dashboard (per-model keys) |
| `FRONTEND_URL` | Your frontend domain |
| `GATEWAY_URL` | Your API gateway domain |
| `NEO4J_URI / _USERNAME / _PASSWORD / _DATABASE` | Neo4j Aura console |
| `POSTGRESQL_URI` | VPS 1 credentials |
| `RABBITMQ_URI` | VPS 1 credentials |
| `KEYCLOAK_*` | Keycloak admin console |

---

## 3. First Deploy (manual)

> DNS for `RAG_DOMAIN` must already point to VPS 3's public IP before running certbot.

```bash
cd /root/scilab/scilab-rag/deploy/prod

# Step 1 — obtain TLS certificate (certbot runs standalone on :80, then exits)
docker compose run --rm certbot

# Step 2 — start nginx
docker compose up -d nginx

# Step 3 — build the app image
docker compose build scilab-rag

# Step 4 — run PostgreSQL migrations
docker compose run --rm scilab-rag alembic upgrade head

# Step 5 — start the app
docker compose up -d scilab-rag

# Verify
docker compose ps
docker compose logs -f scilab-rag
curl -I https://rag.hyperdatalab.site/health
```

### Certificate renewal

Certbot `restart: "no"` means it only runs on explicit invocation.
To renew, stop nginx (frees :80), renew, restart nginx:

```bash
cd /root/scilab/scilab-rag/deploy/prod
docker compose stop nginx
docker compose run --rm certbot renew
docker compose start nginx
```

Or set up a cron job on VPS 3:
```
0 3 1 * * cd /root/scilab/scilab-rag/deploy/prod && docker compose stop nginx && docker compose run --rm certbot renew && docker compose start nginx
```

---

## 4. GitHub Actions CI/CD

Two workflows live in `.github/workflows/`:

| Workflow | Trigger |
|----------|---------|
| `deploy.yml` | Push to `main` (app/alembic/Dockerfile changes) |
| `docker-cleanup.yml` | Weekly — Sunday 3AM UTC |

### Required GitHub Secrets

Go to the repo → **Settings → Secrets and variables → Actions**:

| Secret | Value |
|--------|-------|
| `VPS_HOST` | VPS 3 public IP |
| `VPS_USER` | `root` (or your SSH user) |
| `VPS_SSH_KEY` | Full contents of the private key (PEM, including headers) |
| `VPS_PORT` | `22` |

### SSH key setup on VPS 3

The key in `VPS_SSH_KEY` must be in `~/.ssh/authorized_keys` on VPS 3:

```bash
# Generate a dedicated deploy key (on your local machine)
ssh-keygen -t ed25519 -C "github-actions-scilab-rag" -f ~/.ssh/scilab_rag_deploy

# Copy public key to VPS 3
ssh-copy-id -i ~/.ssh/scilab_rag_deploy.pub root@<VPS3_IP>

# Add the private key content to GitHub secret VPS_SSH_KEY
cat ~/.ssh/scilab_rag_deploy
```

### What the deploy workflow does on each push to `main`

1. SSH into VPS 3
2. `git reset --hard origin/main`
3. `cd deploy/prod && docker compose build --no-cache scilab-rag`
4. `docker compose run --rm scilab-rag alembic upgrade head`
5. `docker compose up -d --no-deps scilab-rag`
6. Health check (30s wait, inspect container state)
7. Prune dangling images
8. Open a GitHub issue on failure

---

## 5. Step-by-Step Checklist

### Phase 1 — Provision VPS 3
- [ ] Create VPS 3 (Ubuntu 22.04+) using the cloud-init above
- [ ] Confirm SSH access: `ssh root@<VPS3_IP>`
- [ ] Confirm Docker is running: `docker ps`
- [ ] Confirm repo was cloned: `ls /root/scilab/scilab-rag`

### Phase 2 — Environment
- [ ] Copy `.env.example` to `.env` in `deploy/prod/`
- [ ] Fill in all values (Neo4j Aura URI, PostgreSQL URI, RabbitMQ URI, all API keys)
- [ ] Verify Neo4j connectivity: `curl -s <NEO4J_URI>` or test from the app

### Phase 3 — First Manual Deploy
- [ ] Add DNS A record: `rag.hyperdatalab.site` → VPS 3 public IP (wait for propagation)
- [ ] Fill `CERTBOT_EMAIL` and `RAG_DOMAIN` in `.env`
- [ ] `cd /root/scilab/scilab-rag/deploy/prod`
- [ ] `docker compose run --rm certbot` — obtain TLS certificate
- [ ] `docker compose up -d nginx` — start reverse proxy
- [ ] `docker compose build scilab-rag`
- [ ] `docker compose run --rm scilab-rag alembic upgrade head`
- [ ] `docker compose up -d scilab-rag`
- [ ] `docker compose ps` — confirm all services `Up`
- [ ] `curl -I https://rag.hyperdatalab.site/health` — confirm HTTPS works

### Phase 4 — CI/CD
- [ ] Generate a deploy SSH key pair (see section 4)
- [ ] Add `VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`, `VPS_PORT` to GitHub secrets
- [ ] Push a commit touching `app/` to trigger the deploy workflow
- [ ] Confirm the Actions run completes green

---

## Troubleshooting

### Build fails — cannot find Dockerfile

The compose `build.context` is `../../` (repo root). If you run `docker compose` from a
different directory, Docker won't find the Dockerfile. Always run from `deploy/prod/`.

### App fails to start — check logs

```bash
docker compose logs --tail=50 scilab-rag
```

### Migration fails

```bash
# Run interactively
docker compose run --rm scilab-rag alembic upgrade head

# Check the POSTGRESQL_URI is reachable from VPS 3
docker compose run --rm scilab-rag python -c \
  "import os; print(os.getenv('POSTGRESQL_URI'))"
```

### Health check failing

The Dockerfile exposes `:8080` and the healthcheck calls `/health`. Confirm that endpoint
exists in the app. If it doesn't, either add it or change the healthcheck to a known route.

### Restart cleanly

```bash
cd /root/scilab/scilab-rag/deploy/prod
docker compose down
docker compose up -d scilab-rag
```
