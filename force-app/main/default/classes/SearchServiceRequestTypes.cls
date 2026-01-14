public class SearchServiceRequestTypes {

    @InvocableMethod(
        label='Search 311 Service Request Types'
        description='Searches Toronto 311 service request codes by keyword, category, or description. Returns matching service types for complaint triage.'
    )
    public static List<SearchResult> search(List<SearchRequest> requests) {
        List<SearchResult> results = new List<SearchResult>();
        
        // Fetch all active service types
        List<Service_Request_Type__c> allTypes = [
            SELECT Id, Name, Service_Code__c, Category__c, Subtype__c,
                   Default_Priority__c, SLA_Hours__c, Keywords__c,
                   Clarifying_Questions__c, Owning_Queue_Id__c
            FROM Service_Request_Type__c
            WHERE Active__c = true
        ];
        
        for (SearchRequest req : requests) {
            SearchResult result = new SearchResult();
            result.serviceTypes = new List<ServiceTypeInfo>();
            
            // Split on commas to handle multiple keywords
            List<String> searchTerms = req.searchTerm.toLowerCase().split(',');
            Set<Id> matchedIds = new Set<Id>();
            List<Service_Request_Type__c> matches = new List<Service_Request_Type__c>();
            
            for (String rawTerm : searchTerms) {
                String searchTerm = rawTerm.trim();
                if (String.isBlank(searchTerm)) continue;
                
                for (Service_Request_Type__c t : allTypes) {
                    if (matchedIds.contains(t.Id)) continue;
                    if (matches.size() >= 5) break;
                    
                    Boolean matched = false;
                    
                    // Check Name
                    if (t.Name != null && t.Name.toLowerCase().contains(searchTerm)) {
                        matched = true;
                    }
                    // Check Category
                    if (!matched && t.Category__c != null && t.Category__c.toLowerCase().contains(searchTerm)) {
                        matched = true;
                    }
                    // Check Subtype
                    if (!matched && t.Subtype__c != null && t.Subtype__c.toLowerCase().contains(searchTerm)) {
                        matched = true;
                    }
                    // Check Keywords
                    if (!matched && t.Keywords__c != null && t.Keywords__c.toLowerCase().contains(searchTerm)) {
                        matched = true;
                    }
                    
                    if (matched) {
                        matches.add(t);
                        matchedIds.add(t.Id);
                    }
                }
                
                if (matches.size() >= 5) break;
            }
            
            for (Service_Request_Type__c t : matches) {
                ServiceTypeInfo info = new ServiceTypeInfo();
                info.recordId = t.Id;
                info.name = t.Name;
                info.serviceCode = t.Service_Code__c;
                info.category = t.Category__c;
                info.subtype = t.Subtype__c;
                info.priority = t.Default_Priority__c;
                info.slaHours = t.SLA_Hours__c;
                info.clarifyingQuestions = t.Clarifying_Questions__c;
                info.queueId = t.Owning_Queue_Id__c;
                result.serviceTypes.add(info);
            }
            
            result.matchCount = matches.size();
            result.resultsJson = JSON.serialize(result.serviceTypes);
            results.add(result);
        }
        
        return results;
    }

    public class SearchRequest {
        @InvocableVariable(required=true description='Search term from citizen complaint - can be keywords, category, or issue description')
        public String searchTerm;
    }

    public class SearchResult {
        @InvocableVariable(description='Number of matching service types found')
        public Integer matchCount;
        
        @InvocableVariable(description='List of matching 311 service request types')
        public List<ServiceTypeInfo> serviceTypes;
        
        @InvocableVariable(description='JSON string of matching service types for agent processing')
        public String resultsJson;
    }

    public class ServiceTypeInfo {
        @InvocableVariable(description='Salesforce record ID')
        public String recordId;
        
        @InvocableVariable(description='Service request type name')
        public String name;
        
        @InvocableVariable(description='Toronto 311 service code')
        public String serviceCode;
        
        @InvocableVariable(description='Category (e.g., Roads, Parks, Water)')
        public String category;
        
        @InvocableVariable(description='Subtype for routing')
        public String subtype;
        
        @InvocableVariable(description='Default priority level')
        public String priority;
        
        @InvocableVariable(description='SLA in hours for resolution')
        public Decimal slaHours;
        
        @InvocableVariable(description='Questions to ask citizen for clarification')
        public String clarifyingQuestions;
        
        @InvocableVariable(description='Queue ID for case routing')
        public String queueId;
    }
}