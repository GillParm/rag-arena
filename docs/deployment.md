# Deploying to Cloud Run

Both images are built for Cloud Run: they listen on `$PORT`, run as a non-root user and keep
no state on local disk that matters between instances. This guide has not been run against a
live Google Cloud project from this repository; it lists the steps and the settings that
matter.

## What production needs

| Piece | Local | Production |
| --- | --- | --- |
| Postgres with pgvector | `pgvector/pgvector:pg17` in Compose | Cloud SQL for PostgreSQL 17 (pgvector is a supported extension; the first migration runs `CREATE EXTENSION vector`) |
| API | `backend/Dockerfile` on port 8300 | Cloud Run service, 2 vCPU and 4 GiB (the cross-encoder needs the memory) |
| Web | `web/Dockerfile` on port 3300 | Cloud Run service, 1 vCPU and 512 MiB |
| Secrets | `.env` | Secret Manager, mounted as `GEMINI_API_KEY` |

The API needs only the Gemini key. The Anthropic key is used for building the eval set and
judging, which run offline from a laptop or CI, not in the deployed service.

## Steps

```bash
PROJECT=my-project
REGION=us-central1
REPO=$REGION-docker.pkg.dev/$PROJECT/rag-arena

# 1. Images (from the repo root)
docker build -f backend/Dockerfile -t $REPO/api:latest .
docker build -t $REPO/web:latest web
docker push $REPO/api:latest
docker push $REPO/web:latest

# 2. Database
gcloud sql instances create rag-arena-db --database-version=POSTGRES_17 \
  --region=$REGION --tier=db-custom-1-3840
gcloud sql databases create rag_arena --instance=rag-arena-db
gcloud sql users create rag --instance=rag-arena-db --password=...

# 3. Load the index once, from any machine that can reach the database
#    (Cloud SQL Auth Proxy on localhost:5433, for example)
export RAG_ARENA_DATABASE_URL=postgresql://rag:...@localhost:5433/rag_arena
rag-arena corpus download
rag-arena index build

# 4. API
gcloud run deploy rag-arena-api --image=$REPO/api:latest --region=$REGION \
  --cpu=2 --memory=4Gi --min-instances=0 --max-instances=3 --concurrency=8 \
  --add-cloudsql-instances=$PROJECT:$REGION:rag-arena-db \
  --set-env-vars=RAG_ARENA_DATABASE_URL="postgresql://rag:...@/rag_arena?host=/cloudsql/$PROJECT:$REGION:rag-arena-db" \
  --set-env-vars=RAG_ARENA_CORS_ORIGINS='["https://rag-arena-web-....run.app"]' \
  --set-secrets=GEMINI_API_KEY=gemini-api-key:latest

# 5. Web
gcloud run deploy rag-arena-web --image=$REPO/web:latest --region=$REGION \
  --cpu=1 --memory=512Mi --set-env-vars=API_URL=https://rag-arena-api-....run.app
```

## Notes

- The API image bakes in the embedding model and the reranker (about 1.2 GB), so cold starts
  do not download from Hugging Face. Build with `--build-arg BAKE_MODELS=false` for a small
  image that downloads on first use instead.
- The BM25 index is rebuilt from Postgres on start-up when it is missing, which takes a few
  seconds per chunk set. Startup also loads the default chunk set and the reranker so the
  first question is not slow; give the service a start-up probe of at least 60 seconds.
- `/api/ask` is rate limited per client IP in memory, which is per instance. For a public
  demo, put Cloud Armor or API Gateway in front for a global limit, and set a budget alert on
  the Gemini key.
- The leaderboard is read from `results/runs/main/summary.json`, copied into the API image at
  build time. Rebuild the image after a new benchmark run.
- Terraform is not included.
