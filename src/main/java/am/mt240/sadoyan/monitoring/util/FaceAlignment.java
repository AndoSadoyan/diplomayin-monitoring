package am.mt240.sadoyan.monitoring.util;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_core;

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
     */
    private static Mat getSimilarityTransform(Point2f[] src, Point2f[] dst) {
        // Calculate centroids
        Point2f srcCentroid = new Point2f(0, 0);
        Point2f dstCentroid = new Point2f(0, 0);
        
        for (int i = 0; i < src.length; i++) {
            srcCentroid.x(srcCentroid.x() + src[i].x());
            srcCentroid.y(srcCentroid.y() + src[i].y());
            dstCentroid.x(dstCentroid.x() + dst[i].x());
            dstCentroid.y(dstCentroid.y() + dst[i].y());
        }
        
        srcCentroid.x(srcCentroid.x() / src.length);
        srcCentroid.y(srcCentroid.y() / src.length);
        dstCentroid.x(dstCentroid.x() / dst.length);
        dstCentroid.y(dstCentroid.y() / dst.length);
        
        // Calculate scale
        double srcDist = 0, dstDist = 0;
        for (int i = 0; i < src.length; i++) {
            double dx1 = src[i].x() - srcCentroid.x();
            double dy1 = src[i].y() - srcCentroid.y();
            double dx2 = dst[i].x() - dstCentroid.x();
            double dy2 = dst[i].y() - dstCentroid.y();
            srcDist += Math.sqrt(dx1 * dx1 + dy1 * dy1);
            dstDist += Math.sqrt(dx2 * dx2 + dy2 * dy2);
        }
        double scale = dstDist / srcDist;
        
        // Calculate rotation angle (using first two points - eyes)
        double srcDx = src[1].x() - src[0].x();
        double srcDy = src[1].y() - src[0].y();
        double dstDx = dst[1].x() - dst[0].x();
        double dstDy = dst[1].y() - dst[0].y();
        
        double srcAngle = Math.atan2(srcDy, srcDx);
        double dstAngle = Math.atan2(dstDy, dstDx);
        double angle = dstAngle - srcAngle;
        
        // Build transformation matrix
        double cosA = Math.cos(angle);
        double sinA = Math.sin(angle);
        
        // Translation to move centroid to origin, rotate, scale, translate to destination
        Mat transform = new Mat(2, 3, CV_64F);
        transform.put(new Mat(0, 0, scale * cosA));
        transform.put(new Mat(0, 1, -scale * sinA));
        transform.put(new Mat(0, 2, dstCentroid.x() - scale * (cosA * srcCentroid.x() - sinA * srcCentroid.y())));
        transform.put(new Mat(1, 0, scale * sinA));
        transform.put(new Mat(1, 1, scale * cosA));
        transform.put(new Mat(1, 2, dstCentroid.y() - scale * (sinA * srcCentroid.x() + cosA * srcCentroid.y())));
        
        return transform;
    }
}