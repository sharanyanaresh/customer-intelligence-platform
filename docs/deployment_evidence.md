# Deployment Evidence

## Local Docker Compose Evidence

Use this command to start both local services:

```powershell
docker-compose up --build -d
```

Use these commands to capture evidence:

```powershell
docker-compose ps
curl.exe http://127.0.0.1:8000/health
curl.exe http://127.0.0.1:8001/health
```

Expected evidence:

- `customer-intel-ml-api` is running and healthy on port `8000`.
- `customer-intel-rag-api` is running and healthy on port `8001`.
- ML `/health` returns `status`, `model_version`, `index_version`, and `uptime_seconds`.
- RAG `/health` returns `status`, `index_version`, and `prompt_version`.

## Cloud Run Evidence

Cloud deployment should use:

- Region: `us-central1`
- Min instances: `0`
- Max instances: `1`
- Memory: `512Mi`
- Service: `customer-intel-ml-api`

Paste the deployed endpoint URL here after deployment:

```text
Cloud Run URL: <paste-url-here-after-deployment>
```

Screenshots to save for submission:

1. Cloud Run service details page showing URL, region, and revision.
2. Browser or terminal output from `GET /health`.
3. Cloud Run logs showing a successful request.
4. Local `docker-compose ps` output if cloud deployment is skipped.

