public class RedactLinkedImagesJob implements Queueable, Database.AllowsCallouts {

    private List<Id> linkIds;
    private static final String HEROKU_ENDPOINT = 'https://toronto-blur-redactor-c9c04341d42d.herokuapp.com/redact';

    public RedactLinkedImagesJob(List<Id> linkIds) {
        this.linkIds = linkIds;
    }

    public void execute(QueueableContext context) {

        List<ContentDocumentLink> links = [
            SELECT Id, ContentDocumentId, LinkedEntityId
            FROM ContentDocumentLink
            WHERE Id IN :linkIds
        ];

        Set<Id> docIds = new Set<Id>();
        Map<Id, Id> docToCase = new Map<Id, Id>();
        for (ContentDocumentLink l : links) {
            if (l.ContentDocumentId != null && l.LinkedEntityId != null
                && String.valueOf(l.LinkedEntityId).startsWith('500')) {
                docIds.add(l.ContentDocumentId);
                docToCase.put(l.ContentDocumentId, l.LinkedEntityId);
            }
        }
        if (docIds.isEmpty()) return;

        List<ContentVersion> versions = [
            SELECT Id, Title, FileType, VersionData, ContentDocumentId, Is_Redacted__c
            FROM ContentVersion
            WHERE ContentDocumentId IN :docIds
            ORDER BY CreatedDate DESC
        ];

        Set<String> imageTypes = new Set<String>{'JPG','JPEG','PNG','HEIC'};
        Set<Id> processedDocs = new Set<Id>();

        for (ContentVersion v : versions) {
            if (processedDocs.contains(v.ContentDocumentId)) continue;
            processedDocs.add(v.ContentDocumentId);

            if (v.Is_Redacted__c) continue;
            if (v.FileType == null || !imageTypes.contains(v.FileType)) continue;

            // Call Heroku to redact the image
            Blob redactedBytes = callHerokuRedact(v.VersionData, v.Title, docToCase.get(v.ContentDocumentId));
            
            if (redactedBytes != null) {
                ContentVersion redacted = new ContentVersion(
                    Title = 'REDACTED - ' + v.Title,
                    VersionData = redactedBytes,
                    PathOnClient = 'REDACTED-' + v.Title + '.' + v.FileType.toLowerCase(),
                    Is_Redacted__c = true
                );
                insert redacted;
            }
        }
    }

    private Blob callHerokuRedact(Blob imageData, String title, Id caseId) {
        try {
            HttpRequest req = new HttpRequest();
            req.setEndpoint(HEROKU_ENDPOINT);
            req.setMethod('POST');
            req.setHeader('Content-Type', 'application/json');
            req.setTimeout(120000);

            Map<String, Object> payload = new Map<String, Object>{
                'imageBase64' => EncodingUtil.base64Encode(imageData),
                'metadata' => new Map<String, Object>{
                    'title' => title,
                    'caseId' => caseId
                }
            };
            req.setBody(JSON.serialize(payload));

            Http http = new Http();
            HttpResponse res = http.send(req);

            if (res.getStatusCode() == 200) {
                Map<String, Object> result = (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
                if (result.containsKey('redactedBase64')) {
                    return EncodingUtil.base64Decode((String) result.get('redactedBase64'));
                }
            }
            System.debug('Heroku redact failed: ' + res.getStatusCode() + ' ' + res.getBody());
            return imageData; // Fallback to original if redaction fails
        } catch (Exception e) {
            System.debug('Heroku callout error: ' + e.getMessage());
            return imageData; // Fallback to original on error
        }
    }
}