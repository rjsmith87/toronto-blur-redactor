# Toronto 311: AI Privacy Wall

**AI-powered privacy gateway that automatically redacts faces and license plates from municipal photo submissions before they enter Salesforce Agentforce — ensuring GDPR/SOC2 compliance at the edge.**

Built with YOLOv8, MediaPipe, Flask, and Salesforce Einstein Vector Store. Deployed on Heroku.

---

## The Problem

Cities collect thousands of photo-based service requests daily (potholes, graffiti, debris). These images frequently contain faces and license plates — PII that creates legal liability the moment it enters a CRM. Traditional approaches rely on manual review or post-ingestion redaction, both of which are slow and error-prone.

## The Solution

This project implements a **zero-trust Privacy Wall** — a computer vision microservice that sits between citizen photo uploads and the Salesforce platform. Every image is scrubbed of biometric and vehicle identifiers *before* it touches the CRM, eliminating PII liability at the architectural level.

Key capabilities:
- **Real-time face detection** via MediaPipe with Gaussian blur redaction
- **License plate detection** via YOLOv8 (ONNX Runtime) with region masking
- **Deterministic fallback** — when AI confidence drops below 0.6, a geometric head-zone blur activates as a safety net
- **Agentforce integration** — redacted images are analyzed by an AI agent that maps them to Toronto's 371-category municipal service taxonomy using semantic vector search

---

## Tech Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Computer Vision** | YOLOv8 (ONNX Runtime) | Vehicle and license plate detection |
| **Face Detection** | MediaPipe | Real-time facial landmark detection |
| **Image Processing** | OpenCV, Pillow | Blur application and image manipulation |
| **Web Framework** | Flask + Gunicorn | REST API serving the redaction pipeline |
| **AI Orchestration** | Salesforce Agentforce | Photo analysis and service categorization |
| **Vector Search** | Einstein Vector Store | Semantic matching against municipal taxonomy |
| **Prompt Engineering** | Agentforce Prompt Templates | Vision-based categorization prompts |
| **Backend Logic** | Apex (REST handler) | Secure external data ingestion |
| **Auth** | JWT Bearer Flow (RSA-256) | Salesforce OAuth machine-to-machine auth |
| **Hosting** | Heroku (Python 3.11) | Microservice deployment |
| **Frontend** | Lightning Web Components | Salesforce UI for agent interaction |

---

## Architecture

```
                          PRIVACY WALL
                    ┌─────────────────────┐
                    │   Heroku (Python)    │
                    │                     │
  ┌──────────┐     │  ┌───────────────┐  │     ┌──────────────────────────┐
  │  Citizen  │     │  │  MediaPipe    │  │     │     Salesforce Org       │
  │  Photo    │────▶│  │  Face Detect  │  │     │                          │
  │  Upload   │     │  └───────┬───────┘  │     │  ┌────────────────────┐  │
  └──────────┘     │          │          │     │  │  Agentforce Agent  │  │
                    │  ┌───────▼───────┐  │     │  │  (Austin311Analysis│  │
                    │  │  YOLOv8       │  │     │  │   Prompt Template) │  │
                    │  │  Plate Detect  │  │     │  └────────┬───────────┘  │
                    │  └───────┬───────┘  │     │           │              │
                    │          │          │     │  ┌────────▼───────────┐  │
                    │  ┌───────▼───────┐  │     │  │  Einstein Vector   │  │
                    │  │  Gaussian     │  │     │  │  Store (371 svc    │  │
                    │  │  Blur Engine  │──│────▶│  │  request types)    │  │
                    │  └───────────────┘  │     │  └────────────────────┘  │
                    │                     │ JWT │                          │
                    │  Fallback: Global   │────▶│  AustinAgentREST.cls    │
                    │  Head-Zone Blur     │     │  (Apex REST endpoint)   │
                    └─────────────────────┘     └──────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────────┐
  │  Data Pipeline: Toronto Open Data → ETL → 371 Service Types →          │
  │  Vector Embeddings → Einstein Vector Store                             │
  └─────────────────────────────────────────────────────────────────────────┘
```

---

## How It Works

1. **Upload** — A citizen submits a photo through the 311 portal
2. **Detect** — MediaPipe identifies faces; YOLOv8 identifies vehicles and plates
3. **Redact** — Detected regions are masked with Gaussian blur. If detection confidence is low, a deterministic global head-zone blur is applied as a safety net
4. **Authenticate** — The service authenticates to Salesforce via JWT Bearer Flow (RSA-256)
5. **Upload** — The redacted image is pushed to Salesforce as a ContentDocument
6. **Analyze** — Agentforce's AI agent examines the scrubbed image and maps it to the correct service category using semantic vector search across 371 Toronto service request types
7. **Route** — The service request is created and routed to the appropriate municipal department

---

## Data Engineering

The AI agent is grounded in real municipal logic through a multi-step ETL pipeline:

- **Source:** City of Toronto Open Data portal
- **Extraction:** 371 unique service request types cleaned and normalized
- **Vectorization:** Custom schema designed for Einstein Vector Store embeddings
- **Result:** The agent performs semantic matching — not keyword search — against the full municipal taxonomy, enabling accurate categorization even for ambiguous or novel submissions

---

## Repository Structure

```
toronto-blur-redactor/
├── app.py                  # Flask app — redaction pipeline + Salesforce integration
├── requirements.txt        # Python dependencies
├── Procfile                # Heroku process definition
├── Aptfile                 # System-level dependencies for Heroku
├── tpir.json               # Toronto service request taxonomy (371 types)
├── tpir_schema.json        # Vector store schema definition
├── force-app/              # Salesforce metadata
│   └── main/default/
│       ├── classes/        # Apex classes (AustinAgentREST.cls, etc.)
│       ├── flows/          # Orchestration flows
│       └── objects/        # Custom objects (Service_Request_Type__c)
├── heroku/                 # Heroku deployment configuration
├── lwc/                    # Lightning Web Components
└── static/                 # Frontend assets
```

---

## Setup

### Prerequisites
- Python 3.11+
- Heroku CLI
- Salesforce org with Agentforce enabled

### Environment Variables
```
SF_CONSUMER_KEY=<connected-app-consumer-key>
SF_USERNAME=<salesforce-username>
SF_LOGIN_URL=https://login.salesforce.com
SF_PRIVATE_KEY=<rsa-private-key>
SF_API_VERSION=62.0
```

### Local Development
```bash
pip install -r requirements.txt
python app.py
```

### Deploy to Heroku
```bash
heroku create
git push heroku main
```

---

## License

See [LICENSE](LICENSE) for details.
