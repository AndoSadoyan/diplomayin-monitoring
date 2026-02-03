package am.mt240.sadoyan.monitoring;

import ai.onnxruntime.*;
import org.bytedeco.opencv.opencv_core.*;

import java.nio.FloatBuffer;
import java.util.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class ScrfdDetector {

    public static class Detection {
        public final Rect bbox;        // in ORIGINAL frame coords
        public final float score;
        public final Point2f[] kps5;   // [L-eye, R-eye, nose, L-mouth, R-mouth] in ORIGINAL frame coords

        public Detection(Rect bbox, float score, Point2f[] kps5) {
            this.bbox = bbox;
            this.score = score;
            this.kps5 = kps5;
        }
    }

    private final OrtEnvironment env;
    private final OrtSession session;

    // Use square input for SCRFD. 640 works with your tensor counts.
    private final int inputSize = 640;

    // Output tensor names from your dump
    private static final String[] SCORE_NAMES = {"448", "471", "494"};
    private static final String[] BBOX_NAMES  = {"451", "474", "497"};
    private static final String[] KPS_NAMES   = {"454", "477", "500"};
    private static final int[] STRIDES = {8, 16, 32};
    private static final int NUM_ANCHORS = 2;

    public ScrfdDetector(OrtEnvironment env, OrtSession session) {
        this.env = env;
        this.session = session;
    }

    public List<Detection> detect(Mat frameBgr, float confThresh) throws OrtException {
        if (frameBgr == null || frameBgr.empty()) return Collections.emptyList();

        // 1) Letterbox to 640x640 (keeps aspect ratio)
        Letterbox lb = letterbox(frameBgr, inputSize, inputSize);

        // 2) Preprocess: BGR -> RGB, float32, normalize (x - 127.5) / 128.0
        // This is the most common SCRFD preprocess.
        Mat rgb = new Mat();
        cvtColor(lb.img, rgb, COLOR_BGR2RGB);

        rgb.convertTo(rgb, CV_32F);
        rgb = subtract(rgb, new Scalar(127.5, 127.5, 127.5, 0)).asMat();
        rgb = multiply(rgb, 1.0 / 128.0).asMat();

        float[] chw = matToCHW(rgb);
        rgb.release();

        OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{1, 3, inputSize, inputSize});
        String inputName = "input.1";

        OrtSession.Result out = session.run(Collections.singletonMap(inputName, input));

        List<Detection> all = new ArrayList<>();

        for (int head = 0; head < 3; head++) {
            int stride = STRIDES[head];

            float[] scores = flattenNx1(out.get(SCORE_NAMES[head]).get().getValue());
            float[][] bbox  = (float[][]) out.get(BBOX_NAMES[head]).get().getValue(); // [N][4]
            float[][] kps   = (float[][]) out.get(KPS_NAMES[head]).get().getValue();  // [N][10]

            int featH = inputSize / stride;
            int featW = inputSize / stride;
            int N = featH * featW * NUM_ANCHORS;

            // Safety check
            if (scores.length != N || bbox.length != N || kps.length != N) {
                // If this triggers, your preprocessing inputSize/strides mismatch the model.
                continue;
            }

            int idx = 0;
            for (int y = 0; y < featH; y++) {
                for (int x = 0; x < featW; x++) {
                    float cx = (x + 0.5f) * stride;
                    float cy = (y + 0.5f) * stride;

                    for (int a = 0; a < NUM_ANCHORS; a++, idx++) {
                        // Many SCRFD exports output logits -> apply sigmoid
                        float score = sigmoid(scores[idx]);
                        if (score < confThresh) continue;

                        float l = bbox[idx][0] * stride;
                        float t = bbox[idx][1] * stride;
                        float r = bbox[idx][2] * stride;
                        float b = bbox[idx][3] * stride;

                        float x1 = cx - l;
                        float y1 = cy - t;
                        float x2 = cx + r;
                        float y2 = cy + b;

                        // Keypoints: 5 points (dx,dy)*stride added to center
                        Point2f[] pts = new Point2f[5];
                        for (int p = 0; p < 5; p++) {
                            float px = cx + kps[idx][2 * p] * stride;
                            float py = cy + kps[idx][2 * p + 1] * stride;
                            // Map back to original frame coords (undo letterbox)
                            pts[p] = new Point2f(
                                    (px - lb.padX) / lb.scale,
                                    (py - lb.padY) / lb.scale
                            );
                        }

                        // Map bbox back to original frame coords
                        float ox1 = (x1 - lb.padX) / lb.scale;
                        float oy1 = (y1 - lb.padY) / lb.scale;
                        float ox2 = (x2 - lb.padX) / lb.scale;
                        float oy2 = (y2 - lb.padY) / lb.scale;

                        // Clamp to image bounds
                        int ix1 = clamp((int) ox1, 0, frameBgr.cols() - 1);
                        int iy1 = clamp((int) oy1, 0, frameBgr.rows() - 1);
                        int ix2 = clamp((int) ox2, 0, frameBgr.cols() - 1);
                        int iy2 = clamp((int) oy2, 0, frameBgr.rows() - 1);

                        int ww = Math.max(1, ix2 - ix1);
                        int hh = Math.max(1, iy2 - iy1);

                        all.add(new Detection(new Rect(ix1, iy1, ww, hh), score, pts));
                    }
                }
            }
        }

        input.close();
        out.close();
        lb.img.release();

        // 3) NMS to remove duplicates
        return nms(all, 0.4f);
    }

    // ---------- Helpers ----------

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    // ORT sometimes returns [N][1], sometimes float[]; make it robust.
    private static float[] flattenNx1(Object v) {
        if (v instanceof float[]) {
            return (float[]) v;
        }
        if (v instanceof float[][] arr) {
            float[] out = new float[arr.length];
            for (int i = 0; i < arr.length; i++) out[i] = arr[i][0];
            return out;
        }
        throw new IllegalArgumentException("Unexpected score tensor type: " + v.getClass());
    }

    private static float[] matToCHW(Mat matFloatRgb) {
        int h = matFloatRgb.rows();
        int w = matFloatRgb.cols();
        int c = matFloatRgb.channels();

        // Convert Mat float RGB to CHW float[]
        float[] out = new float[c * h * w];

        // Safer to convert to 8U? No, we already have CV_32F.
        // Use FloatIndexer:
        var idx = matFloatRgb.createIndexer();
        for (int ch = 0; ch < c; ch++) {
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    out[ch * h * w + y * w + x] = (float) idx.getDouble(y, x, ch);
                }
            }
        }
        idx.release();
        return out;
    }

    private static class Letterbox {
        Mat img;      // 640x640
        float scale;  // resize scale
        int padX;
        int padY;
    }

    private static Letterbox letterbox(Mat src, int dstW, int dstH) {
        int w = src.cols(), h = src.rows();
        float scale = Math.min(dstW / (float) w, dstH / (float) h);

        int newW = Math.round(w * scale);
        int newH = Math.round(h * scale);

        Mat resized = new Mat();
        resize(src, resized, new Size(newW, newH));

        Mat out = Mat.zeros(dstH, dstW, src.type()).asMat();

        int padX = (dstW - newW) / 2;
        int padY = (dstH - newH) / 2;

        Mat roi = new Mat(out, new Rect(padX, padY, newW, newH));
        resized.copyTo(roi);

        roi.release();
        resized.release();

        Letterbox lb = new Letterbox();
        lb.img = out;
        lb.scale = scale;
        lb.padX = padX;
        lb.padY = padY;
        return lb;
    }

    private static List<Detection> nms(List<Detection> dets, float iouThresh) {
        dets.sort((a, b) -> Float.compare(b.score, a.score));
        List<Detection> keep = new ArrayList<>();
        boolean[] removed = new boolean[dets.size()];

        for (int i = 0; i < dets.size(); i++) {
            if (removed[i]) continue;
            Detection a = dets.get(i);
            keep.add(a);
            for (int j = i + 1; j < dets.size(); j++) {
                if (removed[j]) continue;
                Detection b = dets.get(j);
                if (iou(a.bbox, b.bbox) > iouThresh) removed[j] = true;
            }
        }
        return keep;
    }

    private static float iou(Rect a, Rect b) {
        int ax2 = a.x() + a.width();
        int ay2 = a.y() + a.height();
        int bx2 = b.x() + b.width();
        int by2 = b.y() + b.height();

        int x1 = Math.max(a.x(), b.x());
        int y1 = Math.max(a.y(), b.y());
        int x2 = Math.min(ax2, bx2);
        int y2 = Math.min(ay2, by2);

        int iw = Math.max(0, x2 - x1);
        int ih = Math.max(0, y2 - y1);

        float inter = iw * ih;
        float union = a.width() * a.height() + b.width() * b.height() - inter;
        return union <= 0 ? 0 : (inter / union);
    }
}
