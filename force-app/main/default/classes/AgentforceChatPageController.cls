public class AgentforceChatPageController {
    
    @RemoteAction
    public static ChatResult chat(String message, String fileBase64, String fileName) {
        try {
            String redactedFileId = null;
            String redactionInfo = '';
            String redactedBase64 = null;
            String redactedFileName = null;
            Integer facesBlurred = 0;
            Integer platesBlurred = 0;
            
            // Step 1: Redact image (callout to Heroku) - MUST BE FIRST
            if (String.isNotBlank(fileBase64)) {
                Map<String, Object> redactResult = redactImage(fileBase64, fileName);
                if (redactResult != null) {
                    redactedBase64 = (String) redactResult.get('redactedBase64');
                    facesBlurred = (Integer) redactResult.get('facesBlurred');
                    platesBlurred = (Integer) redactResult.get('platesBlurred');
                    redactedFileName = (String) redactResult.get('fileName');
                    if (facesBlurred > 0 || platesBlurred > 0) {
                        redactionInfo = ' (🔒 Auto-redacted: ' + facesBlurred + ' face(s), ' + platesBlurred + ' plate(s) blurred)';
                    }
                }
            }
            
            // Step 2: Save ContentVersion (DML) - before Prompt Builder
            String contentDocumentId = null;
            if (String.isNotBlank(redactedBase64)) {
                ContentVersion cv = new ContentVersion();
                cv.Title = 'Redacted_' + (String.isNotBlank(redactedFileName) ? redactedFileName : 'image.jpg');
                cv.PathOnClient = String.isNotBlank(redactedFileName) ? redactedFileName : 'image.jpg';
                cv.VersionData = EncodingUtil.base64Decode(redactedBase64);
                cv.Description = 'Auto-redacted: ' + facesBlurred + ' faces, ' + platesBlurred + ' plates';
                insert cv;
                redactedFileId = cv.Id;
                
                // Get the ContentDocumentId for Prompt Builder
                cv = [SELECT ContentDocumentId FROM ContentVersion WHERE Id = :cv.Id];
                contentDocumentId = cv.ContentDocumentId;
            }
            
            // Step 3: Analyze photo with Prompt Builder (using ContentDocumentId)
            String photoAnalysis = null;
            if (String.isNotBlank(contentDocumentId)) {
                photoAnalysis = analyzePhoto(contentDocumentId);
            }
            
            // Step 4: Build message for Austin and invoke
            String fullMessage = '';
            
            if (String.isNotBlank(photoAnalysis) && String.isNotBlank(message)) {
                fullMessage = '[Photo Analysis: ' + photoAnalysis + ']\n\nCitizen description: ' + message;
            } else if (String.isNotBlank(photoAnalysis)) {
                fullMessage = '[Photo Analysis: ' + photoAnalysis + ']\n\nThe citizen uploaded a photo without additional description. Please confirm the issue type and collect the location.';
            } else if (String.isNotBlank(redactedBase64)) {
                fullMessage = String.isNotBlank(message) 
                    ? '[Photo uploaded but analysis unavailable]\n\nCitizen description: ' + message
                    : 'A citizen uploaded a photo of a municipal issue but image analysis is temporarily unavailable. Please ask them to describe what they see in the photo and provide the location.';
            } else if (String.isNotBlank(message)) {
                fullMessage = message;
            } else {
                fullMessage = 'A citizen has started a 311 service request. Please greet them and ask how you can help.';
            }
            String botResponse = invokeAustin(fullMessage);
            
            // Fallback if Austin fails
            if (String.isBlank(botResponse)) {
                botResponse = String.isNotBlank(photoAnalysis) ? photoAnalysis : 'Thanks for reaching out to Toronto 311! Could you describe the issue or upload a photo?';
            }
            
            // Add redaction info to response
            if (String.isNotBlank(redactionInfo)) {
                botResponse = botResponse + '\n\n' + redactionInfo;
            }
            
            return new ChatResult(true, botResponse, redactedFileId, redactedBase64);
            
        } catch (Exception e) {
            System.debug('Chat error: ' + e.getMessage() + '\n' + e.getStackTraceString());
            return new ChatResult(false, 'Error: ' + e.getMessage(), null, null);
        }
    }
    
    private static String analyzePhoto(String contentDocumentId) {
        try {
            ConnectApi.EinsteinPromptTemplateGenerationsInput input = new ConnectApi.EinsteinPromptTemplateGenerationsInput();
            input.additionalConfig = new ConnectApi.EinsteinLlmAdditionalConfigInput();
            input.additionalConfig.applicationName = 'PromptBuilderPreview';
            input.isPreview = false;
            
            // Pass as Map with 'id' key - required format for Object inputs
            Map<String, String> photoInput = new Map<String, String>();
            
            photoInput.put('id', contentDocumentId);
            
            Map<String, ConnectApi.WrappedValue> valueMap = new Map<String, ConnectApi.WrappedValue>();
            ConnectApi.WrappedValue wrappedValue = new ConnectApi.WrappedValue();
            wrappedValue.value = photoInput;
            valueMap.put('Input:Photo', wrappedValue);
            input.inputParams = valueMap;
            
            ConnectApi.EinsteinPromptTemplateGenerationsRepresentation result = 
                ConnectApi.EinsteinLLM.generateMessagesForPromptTemplate('Analyze_311_Photo', input);
            
            if (result != null && result.generations != null && !result.generations.isEmpty()) {
                return result.generations[0].text;
            }
            return null;
        } catch (Exception e) {
            System.debug('Photo analysis error: ' + e.getMessage());
            return null;
        }
    }
    
    private static Map<String, Object> redactImage(String base64Image, String fileName) {
        try {
            HttpRequest req = new HttpRequest();
            req.setEndpoint('callout:TorontoBlurClient/redact');
            req.setMethod('POST');
            req.setHeader('Content-Type', 'application/json');
            req.setTimeout(60000);
            
            Map<String, String> body = new Map<String, String>{
                'imageBase64' => base64Image
            };
            req.setBody(JSON.serialize(body));
            
            Http http = new Http();
            HttpResponse res = http.send(req);
            
            if (res.getStatusCode() == 200) {
                Map<String, Object> resBody = (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
                String redactedBase64 = (String) resBody.get('redactedBase64');
                Integer facesBlurred = (Integer) resBody.get('facesBlurred');
                Integer platesBlurred = (Integer) resBody.get('platesBlurred');
                
                return new Map<String, Object>{
                    'fileName' => (String.isNotBlank(fileName) ? fileName : 'image.jpg'),
                    'facesBlurred' => facesBlurred,
                    'platesBlurred' => platesBlurred,
                    'redactedBase64' => redactedBase64
                };
            }
            
        } catch (Exception e) {
            System.debug('Redaction error: ' + e.getMessage());
        }
        
        return null;
    }
    
    private static String invokeAustin(String userMessage) {
        try {
            Invocable.Action action = Invocable.Action.createCustomAction('generateAiAgentResponse', 'Austin_Service_Agent');
            action.setInvocationParameter('userMessage', userMessage);
            List<Invocable.Action.Result> results = action.invoke();
            System.debug('Austin results: ' + results);
            if (results != null && !results.isEmpty()) {
                System.debug('Austin isSuccess: ' + results[0].isSuccess());
                System.debug('Austin outputs: ' + results[0].getOutputParameters());
                if (results[0].isSuccess()) {
                    Object output = results[0].getOutputParameters().get('agentResponse');
                    if (output != null) {
                        Map<String, Object> parsed = (Map<String, Object>) JSON.deserializeUntyped(String.valueOf(output));
                        return (String) parsed.get('value');
                    }
                } else {
                    System.debug('Austin errors: ' + results[0].getErrors());
                }
            }
            return null;
        } catch (Exception e) {
            System.debug('Austin invocation error: ' + e.getMessage());
            return null;
        }
    }
    
    public class ChatResult {
        @AuraEnabled public Boolean success;
        @AuraEnabled public String message;
        @AuraEnabled public String attachmentId;
        @AuraEnabled public String redactedImageBase64;
        
        public ChatResult(Boolean success, String message, String attachmentId, String redactedImageBase64) {
            this.success = success;
            this.message = message;
            this.attachmentId = attachmentId;
            this.redactedImageBase64 = redactedImageBase64;
        }
    }
}