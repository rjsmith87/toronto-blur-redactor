@RestResource(urlMapping='/austin/*')
global class AustinAgentREST {
    
    @HttpPost
    global static String handleRequest() {
        RestRequest req = RestContext.request;
        RestResponse res = RestContext.response;
        
        Map<String, Object> body = (Map<String, Object>) JSON.deserializeUntyped(req.requestBody.toString());
        String message = (String) body.get('message');
        String sessionId = (String) body.get('sessionId');
        
        try {
            Invocable.Action action = Invocable.Action.createCustomAction('generateAiAgentResponse', 'Austin_Service_Agent');
            action.setInvocationParameter('userMessage', message);
            if (String.isNotBlank(sessionId)) {
                action.setInvocationParameter('sessionId', sessionId);
            }
            
            List<Invocable.Action.Result> results = action.invoke();
            
            if (results[0].isSuccess()) {
                Object output = results[0].getOutputParameters().get('agentResponse');
                Map<String, Object> parsed = (Map<String, Object>) JSON.deserializeUntyped(String.valueOf(output));
                
                Map<String, Object> response = new Map<String, Object>{
                    'success' => true,
                    'message' => parsed.get('value'),
                    'sessionId' => results[0].getOutputParameters().get('sessionId')
                };
                return JSON.serialize(response);
            } else {
                return JSON.serialize(new Map<String, Object>{
                    'success' => false,
                    'error' => String.valueOf(results[0].getErrors())
                });
            }
        } catch (Exception e) {
            return JSON.serialize(new Map<String, Object>{
                'success' => false,
                'error' => e.getMessage()
            });
        }
    }
}