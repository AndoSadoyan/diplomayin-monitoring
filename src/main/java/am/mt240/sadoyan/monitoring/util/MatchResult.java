package am.mt240.sadoyan.monitoring.util;

public class MatchResult {
    private String studentId;
    private float confidenceScore;

    public MatchResult(String studentId, float confidenceScore) {
        this.studentId = studentId;
        this.confidenceScore = confidenceScore;
    }

    public String getStudentId() {
        return studentId;
    }

    public void setStudentId(String studentId) {
        this.studentId = studentId;
    }

    public float getConfidenceScore() {
        return confidenceScore;
    }

    public void setConfidenceScore(float confidenceScore) {
        this.confidenceScore = confidenceScore;
    }
}
