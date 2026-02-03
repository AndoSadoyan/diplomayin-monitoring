package am.mt240.sadoyan.monitoring;

import ai.onnxruntime.*;
import am.mt240.sadoyan.monitoring.util.FaceAlignment;
import am.mt240.sadoyan.monitoring.util.MatchResult;
import am.mt240.sadoyan.monitoring.util.PresenceInfo;
import am.mt240.sadoyan.monitoring.util.ResourceUtils;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;

import javax.swing.*;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.*;

import static am.mt240.sadoyan.monitoring.util.PresenceInfo.MAX_GAP_MS;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class MonitoringTerminal {
    private static final String ROOM_ID = "12101";
    private static final long EMBEDDING_REFRESH_INTERVAL_MS = 120000; // Refresh embeddings every 5 minutes

    private OpenCVFrameGrabber grabber;
    private OrtEnvironment env;
    private OrtSession session;
    private OrtSession detSession;
    private ScrfdDetector scrfd;
    private volatile Map<String, Float[]> knownEmbeddings = new ConcurrentHashMap<>();
    private final Map<String, PresenceInfo> activeStudents = new ConcurrentHashMap<>();
    private volatile long lastEmbeddingRefresh = 0;

    public MonitoringTerminal() {
        try {
            env = OrtEnvironment.getEnvironment();
            String detPath = ResourceUtils.copyResourceToTempFile("models/scrfd_10g_bnkps.onnx", ".onnx");
            detSession = env.createSession(detPath, new OrtSession.SessionOptions());
            scrfd = new ScrfdDetector(env, detSession); // defined below

            String recPath = ResourceUtils.copyResourceToTempFile("models/arcfaceresnet100-insightface.onnx", ".onnx");
            session = env.createSession(recPath, new OrtSession.SessionOptions());

            refreshEmbeddings();

            // Start webcam
            grabber = new OpenCVFrameGrabber(0);
            grabber.start();

            startMonitoringLoop();
            startBackendUpdateThread();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void startMonitoringLoop() {
        CanvasFrame canvas = new CanvasFrame("Monitoring");
        canvas.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        canvas.setCanvasSize(grabber.getImageWidth(), grabber.getImageHeight());
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();

        new Thread(() -> {
            try {
                while (canvas.isVisible()) {
                    long now = System.currentTimeMillis();
                    if (now - lastEmbeddingRefresh > EMBEDDING_REFRESH_INTERVAL_MS) {
                        refreshEmbeddings();
                    }

                    Frame frameGrab = grabber.grab();
                    if (frameGrab == null) continue;

                    Mat frame = converter.convert(frameGrab);
                    if (frame == null || frame.empty()) continue;

                    // 1) Detect faces + 5 landmarks ONCE per frame
                    List<ScrfdDetector.Detection> dets = scrfd.detect(frame, 0.6f);

                    // 2) Process each detected face
                    for (ScrfdDetector.Detection det : dets) {
                        if (det == null || det.bbox == null || det.kps5 == null) continue;

                        // Optional minimum size filter
                        if (det.bbox.width() < 80 || det.bbox.height() < 80) continue;

                        // Clamp bbox to frame (prevents drawing errors)
                        Rect bbox = clampRect(det.bbox, frame);

                        // Draw bbox
                        rectangle(frame, bbox, new Scalar(0, 255, 0, 0), 2, LINE_8, 0);

                        // 3) Compute embedding from aligned face using 5 points
                        float[] embedding = computeEmbedding(frame, det.kps5);
                        if (embedding == null) continue;

                        // 4) Match
                        MatchResult match = matchEmbedding(embedding);
                        if (match != null && match.getStudentId() != null) {
                            putText(frame, "Matched: " + match.getStudentId(),
                                    new Point(bbox.x(), Math.max(0, bbox.y() - 10)),
                                    FONT_HERSHEY_SIMPLEX, 0.7,
                                    new Scalar(0, 255, 0, 0), 2, LINE_8, false);

                            trackPresence(match.getStudentId(), match.getConfidenceScore());
                        }
                    }

                    canvas.showImage(converter.convert(frame));
                    Thread.sleep(33);
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                try { grabber.stop(); } catch (Exception ignored) {}
                canvas.dispose();
            }
        }).start();
    }

    // Keeps a rectangle inside the image boundaries (no expansion)
    private Rect clampRect(Rect r, Mat img) {
        int x1 = Math.max(r.x(), 0);
        int y1 = Math.max(r.y(), 0);
        int x2 = Math.min(r.x() + r.width(), img.cols());
        int y2 = Math.min(r.y() + r.height(), img.rows());
        int w = Math.max(1, x2 - x1);
        int h = Math.max(1, y2 - y1);
        return new Rect(x1, y1, w, h);
    }

    private float[] matToCHWFloatArray(Mat mat) {
        int channels = mat.channels();
        int width = mat.cols();
        int height = mat.rows();

        float[] chw = new float[channels * width * height];
        FloatIndexer indexer = mat.createIndexer();

        for (int c = 0; c < channels; c++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    float val = indexer.get(y, x, c);
                    chw[c * height * width + y * width + x] = val;
                }
            }
        }

        indexer.release();
        return chw;
    }

    public float[] computeEmbedding(Mat frameBgr, Point2f[] kps5) {
        try {
            if (frameBgr == null || frameBgr.empty() || kps5 == null || kps5.length != 5)
                return null;

            // 1) Align using 5 landmarks -> returns 112x112 already
            Mat aligned112 = FaceAlignment.alignFace(frameBgr, kps5);
            if (aligned112 == null || aligned112.empty()) return null;

            // 2) float32 + normalize exactly like your registration pipeline
            aligned112.convertTo(aligned112, CV_32F);
            aligned112 = subtract(aligned112, new Scalar(127.5, 127.5, 127.5, 0)).asMat();
            aligned112 = multiply(aligned112, 1.0 / 127.5).asMat();

            float[] chwData = matToCHWFloatArray(aligned112);

            OnnxTensor inputTensor = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(chwData), new long[]{1, 3, 112, 112}
            );

            OrtSession.Result result = session.run(Collections.singletonMap("input.1", inputTensor));
            float[][] output = (float[][]) result.get(0).getValue();

            aligned112.release();
            inputTensor.close();
            result.close();

            return normalize(output[0]);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }


    private float[] normalize(float[] embedding) {
        float norm = 0f;
        for (float v : embedding) norm += v * v;
        norm = (float) Math.sqrt(norm);
        for (int i = 0; i < embedding.length; i++) embedding[i] /= norm;
        return embedding;
    }

    private MatchResult matchEmbedding(float[] embedding) {
        if (embedding == null || knownEmbeddings == null || knownEmbeddings.isEmpty()) {
            return null;
        }

        String bestMatch = null;
        float bestScore = -1f;
        for (Map.Entry<String, Float[]> entry : knownEmbeddings.entrySet()) {
            float score = cosineSimilarity(embedding, entry.getValue());
            if (score > bestScore && score >= 0.5f) {
                bestScore = score;
                bestMatch = entry.getKey();
            }
        }
        if (bestMatch != null) {
            System.out.println("Matched: " + bestMatch + " (score: " + bestScore + ")");
            return new MatchResult(bestMatch, bestScore);
        }
        return null;
    }

    private float cosineSimilarity(float[] a, Float[] b) {
        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / ((float) Math.sqrt(normA) * (float) Math.sqrt(normB));
    }

    private void startBackendUpdateThread() {
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        scheduler.scheduleAtFixedRate(() -> {
            try {
                long now = System.currentTimeMillis();
                Iterator<Map.Entry<String, PresenceInfo>> iterator = activeStudents.entrySet().iterator();

                while (iterator.hasNext()) {
                    Map.Entry<String, PresenceInfo> entry = iterator.next();
                    String studentId = entry.getKey();
                    PresenceInfo info = entry.getValue();

                    if (now - info.getLastSeen() > MAX_GAP_MS) {
                        // Student left - checkout
                        if (info.getSessionId() != null) {
                            try {
                                APIClient.checkout(info.getSessionId(), now);
                                System.out.println("✓ Checked out: " + studentId);
                            } catch (Exception e) {
                                System.err.println("Failed to checkout " + studentId + ": " + e.getMessage());
                            }
                        }
                        iterator.remove();
                    } else {
                        // Student still present
                        if (info.getSessionId() == null && !info.isSyncedToBackend()) {
                            // First time seeing this student - checkin
                            try {
                                String sessionId = APIClient.checkin(studentId, ROOM_ID, info.getFirstSeen(), info.getLastConfidenceScore());
                                info.setSessionId(sessionId);
                                info.setSyncedToBackend(true);
                                System.out.println("✓ Checked in: " + studentId + " (session: " + sessionId + ")");
                            } catch (Exception e) {
                                System.err.println("Failed to checkin " + studentId + ": " + e.getMessage());
                            }
                        } else if (info.getSessionId() != null) {
                            // Update heartbeat
                            try {
                                APIClient.heartbeat(info.getSessionId(), now, info.getLastConfidenceScore());
                            } catch (Exception e) {
                                System.err.println("Failed to send heartbeat for " + studentId + ": " + e.getMessage());
                            }
                        }
                    }
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        }, 0, 2, TimeUnit.SECONDS);
    }

    private void trackPresence(String id, float confidenceScore) {
        if (activeStudents.containsKey(id)) {
            activeStudents.get(id).updateLastSeen(confidenceScore);
        } else {
            activeStudents.put(id, new PresenceInfo(confidenceScore));
        }
    }

    private void refreshEmbeddings() {
        try {
            Map<String, Float[]> newEmbeddings = APIClient.getEmbeddings(ROOM_ID);

            if (newEmbeddings == null) {
                System.err.println("⚠️  Failed to fetch embeddings from backend (null response)");
                return;
            }

            int oldSize = (knownEmbeddings != null) ? knownEmbeddings.size() : 0;

            if (newEmbeddings.isEmpty()) {
                if (oldSize > 0) {
                    System.out.println("\n⚠️  Class ended or no class scheduled in room " + ROOM_ID + " at this time.");
                    System.out.println("⚠️  Monitoring will not track any faces until next class.");
                } else {
                    System.out.println("⚠️  No class scheduled in room " + ROOM_ID + " at this time.");
                    System.out.println("⚠️  Monitoring will not track any faces.");
                }
            } else {
                if (oldSize == 0) {
                    System.out.println("\n✓ Class started! Monitoring room " + ROOM_ID + " with " + newEmbeddings.size() + " registered students.");
                } else if (newEmbeddings.size() != oldSize) {
                    System.out.println("\n🔄 Embeddings refreshed: " + newEmbeddings.size() + " students (was " + oldSize + ")");
                }
            }

            knownEmbeddings = new ConcurrentHashMap<>(newEmbeddings);
            lastEmbeddingRefresh = System.currentTimeMillis();
        } catch (Exception e) {
            System.err.println("Failed to refresh embeddings: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(MonitoringTerminal::new);
    }
}
