# Toronto 311 Privacy-First AI Redactor

This service acts as a **Privacy Wall** between citizen-submitted media and the Salesforce CRM. It ensures that PII (Faces and License Plates) is destroyed at the source before data storage.

## 🛡️ Privacy & Security
- **PII Destruction:** Uses OpenCV Gaussian Blur (Kernel 99) via MediaPipe and YOLOv8.
- **JWT Auth:** Asymmetric RSA-256 token exchange with Salesforce.
- **Compliance:** Original pixels never hit Salesforce; only redacted ContentVersions are created.

## 📊 Data Engineering & Grounding
The AI is grounded using official **Toronto 311 Open Data**. 
- **Taxonomy:** 371 unique Service Request Types.
- **Transformation:** Python logic used to normalize raw municipal data into the Salesforce `Service_Request_Type__c` schema.

## 🏗️ Architecture
- `heroku/`: ML Orchestration Layer.
- `force-app/`: Salesforce Metadata (Flows, Apex, Objects).
- `scripts/`: Data transformation logic.
