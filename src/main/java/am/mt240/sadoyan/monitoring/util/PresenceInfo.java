package am.mt240.sadoyan.monitoring.util;

public class PresenceInfo {
    public static final long MAX_GAP_MS = 10000; // 10 seconds without detection = exit

    private String sessionId;
    private long firstSeen;
    private long lastSeen;
    private float lastConfidenceScore;
    private boolean syncedToBackend = false;

    public PresenceInfo(float confidenceScore) {
        this.firstSeen = System.currentTimeMillis();
        this.lastSeen = this.firstSeen;
        this.lastConfidenceScore = confidenceScore;
    }

    public void updateLastSeen(float confidenceScore) {
        this.lastSeen = System.currentTimeMillis();
        this.lastConfidenceScore = confidenceScore;
    }

    public boolean isStillPresent(long currentTime) {
        return (currentTime - lastSeen) < MAX_GAP_MS;
    }

    public String getSessionId() {
        return sessionId;
    }

    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }

    public long getFirstSeen() {
        return firstSeen;
    }

    public void setFirstSeen(long firstSeen) {
        this.firstSeen = firstSeen;
    }

    public long getLastSeen() {
        return lastSeen;
    }

    public void setLastSeen(long lastSeen) {
        this.lastSeen = lastSeen;
    }

    public float getLastConfidenceScore() {
        return lastConfidenceScore;
    }

    public void setLastConfidenceScore(float lastConfidenceScore) {
        this.lastConfidenceScore = lastConfidenceScore;
    }

    public boolean isSyncedToBackend() {
        return syncedToBackend;
    }

    public void setSyncedToBackend(boolean syncedToBackend) {
        this.syncedToBackend = syncedToBackend;
    }
}