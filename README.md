# scilab-rag

Graph RAG service for ingesting PDFs into a Neo4j-backed knowledge graph and querying it via an API.

## Stack
- FastAPI
- LlamaIndex
- Neo4j (APOC)
- Docling

## Prerequisites
- Conda (Miniconda or Anaconda)
- Docker Desktop (for Neo4j)

## Setup

### 1) Create and activate a conda environment
```bash
conda create -n scilab-rag python=3.11 -y
conda activate scilab-rag
```

### 2) Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3) Start Neo4j with APOC
Use Docker to run Neo4j with the APOC plugin. Example (adjust as needed):
```bash
docker run \
    -p 7474:7474 -p 7687:7687 \
    -v $PWD/data:/data -v $PWD/plugins:/plugins \
    --name neo4j-apoc \
    -e NEO4J_apoc_export_file_enabled=true \
    -e NEO4J_apoc_import_file_enabled=true \
    -e NEO4J_apoc_import_file_use__neo4j__config=true \
    -e NEO4JLABS_PLUGINS=\[\"apoc\"\] \
    neo4j:5.16
```

## Run the API

### Option A: Task runner
```bash
pip install taskipy
```
```bash
task dev
```

### Option B: VS Code debugger
Use the Python debugger in VS Code to launch the app.

## API Overview
- `POST /pdf` Upload a PDF to be ingested
- `POST /ingest` Parse and build the knowledge graph
- `POST /chat` Query the knowledge graph
- `GET /status` Check graph status

## Notes
- Ensure Neo4j is running before ingesting or querying.
- Update environment variables in your config if needed for Neo4j credentials.
