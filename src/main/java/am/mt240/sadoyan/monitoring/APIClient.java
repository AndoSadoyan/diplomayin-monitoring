package am.mt240.sadoyan.monitoring;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Map;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

public class APIClient {

    private static final String BASE_URL = "http://localhost:8088";
    private static final HttpClient client = HttpClient.newHttpClient();
    private static final ObjectMapper mapper = new ObjectMapper();

    public static Map<String, Float[]> getEmbeddings(String roomId) throws Exception {
        String url = BASE_URL + "/students/embeddings";
        if (roomId != null && !roomId.isEmpty()) {
            url += "?roomId=" + roomId;
        }
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(url))
                .header("Content-Type", "application/json")
                .GET()
                .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

        if(response.statusCode() == 200) {
            return mapper.readValue(response.body(), new TypeReference<Map<String, Float[]>>() {});
        } else {
            throw new RuntimeException(response.body());
        }
    }

    public static String checkin(String studentId, String roomId, long timestamp, float confidenceScore) {
        try {
            ObjectNode requestBody = mapper.createObjectNode();
            requestBody.put("studentId", studentId);
            requestBody.put("roomId", roomId);
            requestBody.put("timestamp", timestamp);
            requestBody.put("confidenceScore", confidenceScore);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(BASE_URL + "/attendance/checkin"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(requestBody)))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            if(response.statusCode() == 200) {
                ObjectNode responseBody = mapper.readValue(response.body(), ObjectNode.class);
                return responseBody.get("sessionId").asText();
            } else {
                System.err.println("Checkin failed: " + response.body());
                return null;
            }
        } catch (Exception e) {
            System.err.println("Error during checkin: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    public static void heartbeat(String sessionId, long timestamp, float confidenceScore) {
        try {
            ObjectNode requestBody = mapper.createObjectNode();
            requestBody.put("sessionId", sessionId);
            requestBody.put("timestamp", timestamp);
            requestBody.put("confidenceScore", confidenceScore);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(BASE_URL + "/attendance/heartbeat"))
                    .header("Content-Type", "application/json")
                    .PUT(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(requestBody)))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            if(response.statusCode() != 200) {
                System.err.println("Heartbeat failed: " + response.body());
            }
        } catch (Exception e) {
            System.err.println("Error during heartbeat: " + e.getMessage());
        }
    }

    public static void checkout(String sessionId, long timestamp) {
        try {
            ObjectNode requestBody = mapper.createObjectNode();
            requestBody.put("sessionId", sessionId);
            requestBody.put("timestamp", timestamp);

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(BASE_URL + "/attendance/checkout"))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(requestBody)))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());

            if(response.statusCode() != 200) {
                System.err.println("Checkout failed: " + response.body());
            }
        } catch (Exception e) {
            System.err.println("Error during checkout: " + e.getMessage());
        }
    }
}