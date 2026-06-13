# Deployment Guide

This guide covers deploying the Agentic EDA Pipeline in various environments.

## Local Deployment

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/Sujay1709/agentic-eda-pipeline.git
cd agentic-eda-pipeline

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (if using AI features)
ollama serve

# 5. Pull model in another terminal
ollama pull mistral

# 6. Run the app
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## Docker Deployment

### Prerequisites
- Docker & Docker Compose installed
- At least 8GB RAM available

### Using Docker Compose (Recommended)

```bash
# Clone and navigate to project
git clone https://github.com/Sujay1709/agentic-eda-pipeline.git
cd agentic-eda-pipeline

# Build and start services
docker-compose up --build

# For background execution
docker-compose up -d
```

Services will be available at:
- **Streamlit Web UI**: http://localhost:8501
- **Ollama API**: http://localhost:11434

Pull a model if needed:
```bash
docker exec ollama-service ollama pull mistral
```

**Stopping services:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f eda-pipeline
```

### Using Docker Directly

```bash
# Build image
docker build -t agentic-eda-pipeline:latest .

# Run container
docker run -d \
  --name eda-pipeline \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  agentic-eda-pipeline:latest

# Access the app
# http://localhost:8501

# Stop container
docker stop eda-pipeline
docker rm eda-pipeline
```

---

## Cloud Deployment

### AWS (Using EC2 + Docker)

```bash
# 1. Launch EC2 instance
# - AMI: Ubuntu 22.04 LTS
# - Instance Type: t3.large (minimum)
# - Storage: 30GB
# - Security Group: Allow ports 8501, 11434

# 2. SSH into instance and setup
ssh -i your-key.pem ubuntu@your-instance-ip

# 3. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# 4. Clone and deploy
git clone https://github.com/Sujay1709/agentic-eda-pipeline.git
cd agentic-eda-pipeline
docker-compose up -d

# 5. Access via EC2 public IP:8501
```

### Google Cloud Platform (Cloud Run)

```bash
# 1. Set up gcloud CLI and authenticate
gcloud auth login

# 2. Create Dockerfile optimizations for Cloud Run
# Already in repository

# 3. Build and push to Artifact Registry
gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/eda-pipeline

# 4. Deploy to Cloud Run
gcloud run deploy eda-pipeline \
  --image gcr.io/YOUR-PROJECT-ID/eda-pipeline \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 3600 \
  --port 8501
```

### Heroku

```bash
# 1. Create Heroku app
heroku create your-app-name

# 2. Add buildpack
heroku buildpacks:add heroku/python

# 3. Create Procfile (add to repo)
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# 4. Deploy
git push heroku master

# 5. View logs
heroku logs --tail
```

---

## Production Configuration

### Environment Variables

Create a `.env` file with production settings:

```bash
# .env (production)
EDA_MODEL=mistral
OLLAMA_HOST=http://ollama:11434
EDA_AI_NARRATIVE=true
EDA_MAX_RETRIES=5
LOG_LEVEL=INFO
```

### Performance Tuning

For better performance in production:

```yaml
# docker-compose.yml adjustments
services:
  eda-pipeline:
    environment:
      - STREAMLIT_LOGGER_LEVEL=warning
      - STREAMLIT_CLIENT_MAXUPLOADSIZE=200  # MB
    resources:
      limits:
        memory: 4G
      reservations:
        memory: 2G
  
  ollama:
    environment:
      - OLLAMA_NUM_GPU=1  # If using GPU
    resources:
      limits:
        memory: 8G
```

### SSL/TLS Certificate

For HTTPS deployment with Nginx:

```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Monitoring & Logging

### Health Checks

```bash
# Streamlit health
curl http://localhost:8501/_stcore/health

# Ollama health
curl http://localhost:11434/api/tags
```

### Log Management

```bash
# View application logs
docker-compose logs -f eda-pipeline

# Save logs to file
docker-compose logs > deployment.log

# Cleanup old logs
find logs/ -mtime +30 -delete
```

### Metrics & Monitoring

Consider using:
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **ELK Stack** for log aggregation
- **DataDog** or **New Relic** for APM

---

## Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find process using port 8501
lsof -i :8501
# Kill process
kill -9 <PID>
```

**Out of memory:**
```bash
# Increase Docker memory limits
docker update --memory 4g container-name
```

**Ollama connection errors:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
docker restart ollama-service
```

**Slow processing:**
```bash
# Enable GPU support (if available)
docker run --gpus all ...

# Check resource usage
docker stats
```

---

## Backup & Recovery

### Backup Strategy

```bash
# Backup outputs
tar -czf outputs_backup_$(date +%Y%m%d).tar.gz outputs/

# Backup data
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/

# Store in cloud
aws s3 cp outputs_backup_*.tar.gz s3://your-bucket/backups/
```

### Restore from Backup

```bash
tar -xzf outputs_backup_20240101.tar.gz
docker-compose restart eda-pipeline
```

---

## Scaling

### Horizontal Scaling (Multiple Instances)

Use Docker Swarm or Kubernetes:

```bash
# Docker Swarm
docker swarm init
docker stack deploy -c docker-compose.yml eda-pipeline

# Kubernetes (using Helm chart - future enhancement)
helm install eda-pipeline ./helm-chart
```

### Vertical Scaling

Increase resources per instance:
- More CPU cores
- More RAM (up to 16GB+ for large datasets)
- NVMe SSD storage
- GPU acceleration

---

## Support & Community

- **Issues**: https://github.com/Sujay1709/agentic-eda-pipeline/issues
- **Discussions**: https://github.com/Sujay1709/agentic-eda-pipeline/discussions
- **Contributing**: See CONTRIBUTING.md
