package am.mt240.sadoyan.monitoring.util;

import java.io.*;
import java.nio.file.*;

public final class ResourceUtils {
    public static String copyResourceToTempFile(String resourcePath, String suffix) throws IOException {
        try (InputStream in = ResourceUtils.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (in == null) throw new FileNotFoundException("Missing resource: " + resourcePath);
            Path tmp = Files.createTempFile("onnx_", suffix);
            tmp.toFile().deleteOnExit();
            Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
            return tmp.toAbsolutePath().toString();
        }
    }
}
