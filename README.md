# Toronto 311 Privacy-First AI Redactor

This service redacts PII (Faces and License Plates) from citizen-submitted media before it touches the Salesforce CRM.

## Tech Stack
- **AI Models:** YOLOv8 (Vehicles), MediaPipe (Faces)
- **Security:** JWT Bearer Auth with Salesforce
- **Cloud:** Heroku
