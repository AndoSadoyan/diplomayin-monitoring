package am.mt240.sadoyan.monitoring;

import ai.onnxruntime.*;
import am.mt240.sadoyan.monitoring.util.MatchResult;
import am.mt240.sadoyan.monitoring.util.PresenceInfo;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.global.opencv_dnn;

import javax.swing.*;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.*;

import static am.mt240.sadoyan.monitoring.util.PresenceInfo.MAX_GAP_MS;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class MonitoringTerminal {
    private static final String ROOM_ID = "12101";
    private static final long EMBEDDING_REFRESH_INTERVAL_MS = 30000; // Refresh embeddings every 5 minutes
    
    private OpenCVFrameGrabber grabber;
    private Net faceNet;
    private OrtEnvironment env;
    private OrtSession session;
    private volatile Map<String, Float[]> knownEmbeddings = new ConcurrentHashMap<>();
    private final Map<String, PresenceInfo> activeStudents = new ConcurrentHashMap<>();
    private volatile long lastEmbeddingRefresh = 0;

    public MonitoringTerminal() {
        try {
            faceNet = opencv_dnn.readNetFromCaffe(
                    getClass().getClassLoader().getResource("dnn/deploy.prototxt").getPath(),
                    getClass().getClassLoader().getResource("dnn/res10_300x300_ssd_iter_140000.caffemodel").getPath());
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(getClass().getClassLoader().getResource(
                    "models/arcfaceresnet100-insightface.onnx").getPath(), new OrtSession.SessionOptions());

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
                    // Periodically refresh embeddings (e.g., when class changes)
                    long now = System.currentTimeMillis();
                    if (now - lastEmbeddingRefresh > EMBEDDING_REFRESH_INTERVAL_MS) {
                        refreshEmbeddings();
                    }

                    Frame frameGrab = grabber.grab();
                    if (frameGrab == null) continue;

                    Mat mat = converter.convert(frameGrab);
                    int w = mat.cols(), h = mat.rows();

                    Mat blob = opencv_dnn.blobFromImage(mat, 1.0f, new Size(300, 300),
                            new Scalar(104.0f, 177.0f, 123.0f, 0.0), false, false, CV_32F);
                    faceNet.setInput(blob);
                    Mat detections = faceNet.forward();

                    FloatIndexer indexer = detections.createIndexer();
                    for (int i = 0; i < detections.size(2); i++) {
                        float confidence = indexer.get(0, 0, i, 2);
                        if (confidence > 0.6) {
                            int x1 = (int) (indexer.get(0, 0, i, 3) * w);
                            int y1 = (int) (indexer.get(0, 0, i, 4) * h);
                            int x2 = (int) (indexer.get(0, 0, i, 5) * w);
                            int y2 = (int) (indexer.get(0, 0, i, 6) * h);

                            Rect faceRect = new Rect(x1, y1, x2 - x1, y2 - y1);
                            if (faceRect.width() < 80 || faceRect.height() < 80) continue;

                            Rect croppedRect = safeRect(faceRect, mat);
                            rectangle(mat, croppedRect, new Scalar(0, 255, 0, 0), 2, LINE_8, 0);

                            Mat face = new Mat(mat, croppedRect).clone();
                            float[] embedding = computeEmbedding(face);
                            MatchResult matchResult = matchEmbedding(embedding);
                            if (matchResult != null && matchResult.getStudentId() != null) {
                                putText(mat, "Matched: " + matchResult.getStudentId(),
                                       new Point(croppedRect.x(), croppedRect.y() - 10),
                                       FONT_HERSHEY_SIMPLEX, 0.7, 
                                       new Scalar(0, 255, 0, 0), 2, LINE_8, false);
                                trackPresence(matchResult.getStudentId(), matchResult.getConfidenceScore());
                            }
                        }
                    }
                    canvas.showImage(converter.convert(mat));
                    Thread.sleep(33);
                }
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                try {
                    grabber.stop();
                } catch (Exception e) {
                    e.printStackTrace();
                }
                canvas.dispose();
            }
        }).start();
    }

    private Rect safeRect(Rect rect, Mat mat) {
        int x1 = Math.max(rect.x() - 10, 0);
        int y1 = Math.max(rect.y() - 10, 0);
        int x2 = Math.min(rect.x() + rect.width() + 10, mat.cols());
        int y2 = Math.min(rect.y() + rect.height() + 10, mat.rows());

        int w = Math.max(x2 - x1, 1);
        int h = Math.max(y2 - y1, 1);

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

    public float[] computeEmbedding(Mat faceMat) {
        try {
            if (faceMat == null || faceMat.empty())
                return null;

            // --- 3. Resize to 112x112 (ArcFace expected size) ---
            Mat resized = new Mat();
            resize(faceMat, resized, new Size(112, 112));

            // --- 4. Convert to float32 + normalize ---
            resized.convertTo(resized, CV_32F);
            resized = subtract(resized, new Scalar(127.5,127.5,127.5,0)).asMat();
            resized = multiply(resized, 1.0/127.5).asMat();

            float[] chwData = matToCHWFloatArray(resized);
            OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chwData), new long[]{1, 3, 112, 112});
            OrtSession.Result result = session.run(Collections.singletonMap("input.1", inputTensor));
            float[][] output = (float[][]) result.get(0).getValue();
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


        // todo: add backend calls to store info on presence
    }

    private void trackPresence(String id, float confidenceScore) {
        if (activeStudents.containsKey(id)) {
            activeStudents.get(id).updateLastSeen(confidenceScore);
        } else {
            activeStudents.put(id, new PresenceInfo(confidenceScore));
        }
    }

    private void refreshEmbeddings() {

        //todo: add backend call to get students embeddings with current class in this room
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(MonitoringTerminal::new);
    }
}
