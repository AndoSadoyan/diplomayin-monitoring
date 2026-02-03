package am.mt240.sadoyan.monitoring.util;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_calib3d.*;

public class FaceAlignment {
    private static final Point2f[] DST = new Point2f[]{
            new Point2f(38.2946f, 51.6963f),
            new Point2f(73.5318f, 51.5014f),
            new Point2f(56.0252f, 71.7366f),
            new Point2f(41.5493f, 92.3655f),
            new Point2f(70.7299f, 92.2041f)
    };

    public static Mat alignFace(Mat bgrFrame, Point2f[] kps5) {
        if (bgrFrame == null || bgrFrame.empty()) return bgrFrame;
        if (kps5 == null || kps5.length != 5) return bgrFrame;

        Mat src = pointsToMat(kps5); // (5,1,CV_32FC2)
        Mat dst = pointsToMat(DST);

        Mat M = estimateAffinePartial2D(src, dst); // <-- works in JavaCV

        if (M == null || M.empty()) {
            src.release();
            dst.release();
            return bgrFrame;
        }

        Mat aligned = new Mat();
        warpAffine(bgrFrame, aligned, M, new Size(112, 112),
                INTER_LINEAR, BORDER_REPLICATE, new Scalar());

        src.release();
        dst.release();
        M.release();
        return aligned;
    }

    private static Mat pointsToMat(Point2f[] pts) {
        Mat m = new Mat(pts.length, 1, CV_32FC2);
        FloatIndexer idx = m.createIndexer();
        for (int i = 0; i < pts.length; i++) {
            idx.put(i, 0, 0, pts[i].x());
            idx.put(i, 0, 1, pts[i].y());
        }
        idx.release();
        return m;
    }
}

