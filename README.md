# 🇨🇦 Toronto 311: AI Agent Privacy Wall
**Architecture:** Multi-Cloud Privacy Pipeline (Heroku + Salesforce Agentforce)

## Executive Summary
This project addresses the inherent PII (Personally Identifiable Information) liability in municipal reporting. By implementing a "Privacy Wall" architecture, sensitive biometric data (faces) and vehicle identifiers (license plates) are redacted at the edge before data ingestion into the CRM.

## Core Innovation: The Privacy Wall
The architecture utilizes a zero-trust gateway to ensure compliance with data sovereignty and privacy regulations (GDPR/SOC2).

### 1. Edge Redaction Engine
A Python-based microservice hosted on Heroku utilizes YOLOv8 and MediaPipe to perform real-time computer vision analysis. 

### 2. Deterministic Geometric Fallback
A key innovation in this pipeline is the **Geometric Safety Net**. To account for edge cases where AI confidence intervals fall below 0.6, I implemented a deterministic fallback logic. This secondary layer applies a calculated **Gaussian Global Head-Zone Blur** based on image dimensions, ensuring that no PII enters the Salesforce environment even if specific feature detection is obstructed by low lighting or extreme angles.

### 3. Agentforce Intelligence
The redacted payload is processed by the **Austin311Analysis** Prompt Template. This allows the AI Agent to perform complex visual categorization against the Toronto 311 taxonomy without exposing the municipality to raw biometric data.

---

## Technical Architecture

### Intelligence Layer (Salesforce)
| Component | Identifier | Functional Role |
| :--- | :--- | :--- |
| **Prompt Template** | Austin311Analysis | Generative Vision & Taxonomy Mapping |
| **Orchestration Flow** | Analyze_311_Photo_Flow | Multi-Cloud Transaction Management |
| **Apex REST Handler** | AustinAgentREST.cls | Secure External Data Ingestion |
| **Apex Grounding** | SearchServiceRequestTypes.cls | Metadata-Driven Query Service |
| **Custom Object** | Service_Request_Type__c | Municipal Service Taxonomy Schema |

### Privacy Layer (Heroku)
- **Environment:** Python 3.11
- **Computer Vision:** OpenCV, MediaPipe, Ultralytics (YOLOv8)
- **Authentication:** RSA-256 JWT Bearer Tokens

---

## Repository Structure
- **/heroku**: ML Microservice and Redaction Logic.
- **/force-app**: Salesforce Metadata (Apex, Flows, Objects, Prompt Templates).
- **/scripts**: Data Engineering scripts for Toronto Open Data ETL.
