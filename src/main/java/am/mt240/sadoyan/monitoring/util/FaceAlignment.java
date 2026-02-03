package am.mt240.sadoyan.monitoring.util;

import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;

/**
 * Face alignment using eye landmarks.
 * Aligns face so eyes are horizontal for better embedding consistency.
 */
public class FaceAlignment {

    // Standard eye positions for 112x112 aligned face (ArcFace input size)
    private static final Point2f[] DST_POINTS = new Point2f[]{
            new Point2f(38.2946f, 51.6963f),  // Left eye
            new Point2f(73.5318f, 51.5014f),  // Right eye
            new Point2f(56.0252f, 71.7366f), // Nose tip
            new Point2f(41.5493f, 92.3655f), // Left mouth corner
            new Point2f(70.7299f, 92.2041f)  // Right mouth corner
    };

    /**
     * Aligns face using 5-point landmarks (eyes, nose, mouth corners).
     * This is a simplified version. For production, use a proper landmark detector.
     */
    public static Mat alignFace(Mat faceMat) {
        if (faceMat == null || faceMat.empty()) {
            return faceMat;
        }

        // Detect landmarks (simplified - estimate positions)
        // In production, use a landmark detector like MTCNN or RetinaFace
        Point2f[] srcPoints = estimateLandmarks(faceMat);

        if (srcPoints == null) {
            return faceMat; // Return original if landmarks can't be estimated
        }

        // Calculate similarity transform
        Mat transform = getSimilarityTransform(srcPoints, DST_POINTS);

        if (transform == null || transform.empty()) {
            System.out.println("RETURNING THE ORIGINAL FACE. ALLIGNEMENT FAILED!!!!!");
            return faceMat; // Return original if transform failed
        }

        // Apply transformation
        Mat aligned = new Mat();
        warpAffine(faceMat, aligned, transform, new Size(112, 112),
                INTER_LINEAR, BORDER_REPLICATE, new Scalar());

        transform.release();
        return aligned;
    }

    /**
     * Estimates 5 facial landmarks (simplified version).
     * For production, replace with actual landmark detection.
     */
    private static Point2f[] estimateLandmarks(Mat faceMat) {
        int w = faceMat.cols();
        int h = faceMat.rows();

        // Estimate landmark positions (assumes roughly frontal face)
        // These are normalized estimates - replace with actual detection
        return new Point2f[]{
                new Point2f(w * 0.35f, h * 0.35f),  // Left eye
                new Point2f(w * 0.65f, h * 0.35f),  // Right eye
                new Point2f(w * 0.5f, h * 0.55f),   // Nose tip
                new Point2f(w * 0.4f, h * 0.75f),   // Left mouth corner
                new Point2f(w * 0.6f, h * 0.75f)    // Right mouth corner
        };
    }

    /**
     * Calculates similarity transform matrix (rotation, scale, translation)
     * to align source points to destination points.
     * Returns a 2x3 matrix of type CV_64F.
     */
    private static Mat getSimilarityTransform(Point2f[] src, Point2f[] dst) {
        if (src == null || dst == null || src.length != dst.length || src.length < 2) {
            return null;
        }

        // Calculate centroids
        double srcCentroidX = 0, srcCentroidY = 0;
        double dstCentroidX = 0, dstCentroidY = 0;

        for (int i = 0; i < src.length; i++) {
            srcCentroidX += src[i].x();
            srcCentroidY += src[i].y();
            dstCentroidX += dst[i].x();
            dstCentroidY += dst[i].y();
        }

        srcCentroidX /= src.length;
        srcCentroidY /= src.length;
        dstCentroidX /= dst.length;
        dstCentroidY /= dst.length;

        // Calculate scale using eye distance (first two points)
        double srcEyeDx = src[1].x() - src[0].x();
        double srcEyeDy = src[1].y() - src[0].y();
        double srcEyeDist = Math.sqrt(srcEyeDx * srcEyeDx + srcEyeDy * srcEyeDy);

        double dstEyeDx = dst[1].x() - dst[0].x();
        double dstEyeDy = dst[1].y() - dst[0].y();
        double dstEyeDist = Math.sqrt(dstEyeDx * dstEyeDx + dstEyeDy * dstEyeDy);

        if (srcEyeDist < 1e-6) {
            return null; // Invalid eye distance
        }

        double scale = dstEyeDist / srcEyeDist;

        // Calculate rotation angle (using eyes)
        double srcAngle = Math.atan2(srcEyeDy, srcEyeDx);
        double dstAngle = Math.atan2(dstEyeDy, dstEyeDx);
        double angle = dstAngle - srcAngle;

        // Build transformation matrix: 2x3 matrix of type CV_64F
        Mat transform = new Mat(2, 3, CV_64F);
        DoubleIndexer indexer = transform.createIndexer();

        double cosA = Math.cos(angle);
        double sinA = Math.sin(angle);

        // First row: [scale*cosA, -scale*sinA, tx]
        double tx = dstCentroidX - scale * (cosA * srcCentroidX - sinA * srcCentroidY);
        indexer.put(0, 0, scale * cosA);
        indexer.put(0, 1, -scale * sinA);
        indexer.put(0, 2, tx);

        // Second row: [scale*sinA, scale*cosA, ty]
        double ty = dstCentroidY - scale * (sinA * srcCentroidX + cosA * srcCentroidY);
        indexer.put(1, 0, scale * sinA);
        indexer.put(1, 1, scale * cosA);
        indexer.put(1, 2, ty);

        indexer.release();
        return transform;
    }
}