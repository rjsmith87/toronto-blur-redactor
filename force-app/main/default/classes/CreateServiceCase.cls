public class CreateServiceCase {

    @InvocableMethod(
        label='Create 311 Service Case'
        description='Creates a Case for a citizen complaint with AI triage fields populated. Links to the matched Service Request Type.'
    )
    public static List<CaseResult> createCases(List<CaseRequest> requests) {
        List<CaseResult> results = new List<CaseResult>();
        
        for (CaseRequest req : requests) {
            CaseResult result = new CaseResult();
            
            try {
                Case c = new Case(
                    Subject = req.subject,
                    Description = req.description,
                    Origin = 'Chat',
                    Status = 'New',
                    Priority = req.priority != null ? req.priority : 'Medium',
                    AI_Triage_Category__c = req.category,
                    AI_Triage_Subtype__c = req.subtype,
                    AI_Triage_Summary__c = req.aiSummary,
                    AI_Confidence__c = req.confidence,
                    AI_Routed_Queue__c = req.queueId,
                    Safety_Flag__c = req.safetyFlag != null ? req.safetyFlag : false
                );
                
                // Set owner to queue if provided
                if (String.isNotBlank(req.queueId)) {
                    try {
                        c.OwnerId = Id.valueOf(req.queueId);
                    } catch (Exception e) {
                        // Invalid queue ID, leave default owner
                    }
                }
                
                insert c;
                
                result.success = true;
                result.caseId = c.Id;
                result.caseNumber = [SELECT CaseNumber FROM Case WHERE Id = :c.Id].CaseNumber;
                result.message = 'Case created successfully';
                
            } catch (Exception e) {
                result.success = false;
                result.message = 'Error creating case: ' + e.getMessage();
            }
            
            results.add(result);
        }
        
        return results;
    }

    public class CaseRequest {
        @InvocableVariable(required=true description='Case subject - brief description of the complaint')
        public String subject;
        
        @InvocableVariable(required=true description='Full description of the citizen complaint')
        public String description;
        
        @InvocableVariable(description='Priority level: Low, Medium, High, Critical')
        public String priority;
        
        @InvocableVariable(description='AI-determined category from Service Request Type')
        public String category;
        
        @InvocableVariable(description='AI-determined subtype from Service Request Type')
        public String subtype;
        
        @InvocableVariable(description='AI-generated summary of the complaint for quick agent review')
        public String aiSummary;
        
        @InvocableVariable(description='AI confidence score 0-100')
        public Decimal confidence;
        
        @InvocableVariable(description='Queue ID for routing')
        public String queueId;
        
        @InvocableVariable(description='True if complaint involves safety concern')
        public Boolean safetyFlag;
    }

    public class CaseResult {
        @InvocableVariable(description='Whether case creation succeeded')
        public Boolean success;
        
        @InvocableVariable(description='Salesforce Case ID')
        public String caseId;
        
        @InvocableVariable(description='Case number for citizen reference')
        public String caseNumber;
        
        @InvocableVariable(description='Success or error message')
        public String message;
    }
}