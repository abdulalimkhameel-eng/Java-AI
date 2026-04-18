import com.fazecast.jSerialComm.SerialPort;
import org.opencv.core.*;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.opencv.imgproc.Imgproc;
import ai.onnxruntime.*;
import java.util.Collections;
import java.io.File;
import java.net.URI;
import java.net.http.*;
import java.util.Scanner;
import nu.pattern.OpenCV;

public class AI_PROGRAM {

    private static volatile String currentGPS = "WAITING_FOR_SATELLITES...";
    private static final String NTFY_TOPIC = "Invasive_species_identified";
    private static final String MODEL_PATH = "src/main/resources/model.onnx"; // ✅ fix 2
    private static final int IMG_SIZE = 224;
    private static OrtSession session;       // ✅ fix 1
    private static OrtEnvironment env;       // ✅ fix 1

    static {
        nu.pattern.OpenCV.loadLocally();
    }

    public static void main(String[] args) {
        System.err.println("---DEBUG: MAIN METHOD REACHED---");
    System.err.flush();
        System.out.println("---SYSTEMS INITIALIZING---");
        System.out.println("---SEARCHING FOR AI MODEL---");
        try {
            File modelFile = new File(MODEL_PATH);
            if (!modelFile.exists()) {
                System.err.println("CRITICAL ERROR: model.tflite not found!");
                return;
            }
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(MODEL_PATH);
            System.out.println("[AI] Model Loaded Successfully.");
            System.err.println("---PRESS ENTER TO START SYSTEM LOOP---");
           new java.util.Scanner(System.in).nextLine();
            startGpsThread();   // ✅ matches method name below
            runSystemLoop();
        } catch (Exception error) {
            error.printStackTrace();
        }
    }

    public static void startGpsThread() {
    Thread gpsWorker = new Thread(() -> {
        SerialPort vk172 = null;
        for (SerialPort port : SerialPort.getCommPorts()) {
            System.out.println("Checking port: " + port.getSystemPortName());
            if (port.getSystemPortName().contains("USB") || port.getSystemPortName().contains("ACM")) {
                vk172 = port;
                break; 
            }
        }
        if (vk172 == null) {
            System.err.println("NO GPS PORT FOUND!");
            return;
        }
        vk172.setBaudRate(9600);
        if (vk172.openPort()) {
            System.out.println("---- Connected TO: " + vk172.getSystemPortName() + " ----");
            Scanner scanner = new Scanner(vk172.getInputStream());
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine();
                if (line.startsWith("$GPRMC")) {
                    parseNmea(line);
                }
            }
        }
    });
    gpsWorker.setDaemon(true);
    gpsWorker.start();
}// start gps ends here

    public static void parseNmea(String line) { // ✅ fix 5 - own method now
        String[] parts = line.split(",");
        System.out.println("Debug NMEA: " + line);
        if (parts.length > 6 && parts[2].equals("A")) {
            currentGPS = "Lat: " + parts[3] + parts[4] + " | Lon: " + parts[5] + parts[6];
        } else {
            currentGPS = "SEARCHING_FIX_STATUS: " + parts[2];
        }
    }

    private static void runSystemLoop() {
        VideoCapture camera = new VideoCapture(0);
        Mat frame = new Mat();
        if (!camera.isOpened()) {
            System.err.println("[AI] Camera Hardware not detected! --- ATTEMPTING 2");
            camera = new VideoCapture(1);
            return;
        }
        System.out.println("[AI] Camera Active. Looking for target.");
        while (true) {
            if (camera.read(frame)) {
                Mat resized = new Mat();                                       // ✅ fix 6
                Imgproc.resize(frame, resized, new Size(IMG_SIZE, IMG_SIZE)); // ✅ fix 6
                Imgproc.cvtColor(resized, resized, Imgproc.COLOR_BGR2RGB);
                if (performInference(resized)) {                               // ✅ fix 6
                    String alert = "--- TARGET LOCATED ---, Location: " + currentGPS;
                    System.out.println(alert);
                    sendPhoneNotification(alert);
                    try { Thread.sleep(10000); } catch (InterruptedException e) {}
                }
                try { Thread.sleep(1000); } catch (InterruptedException e) {} 
    }
            }
        }
    }

    private static boolean performInference(Mat frame) {
        try {
        var inputInfo = session.getInputMetadata();
        for (var entry : inputInfo.entrySet()) {
            System.out.println("Model Input Name: " + entry.getKey());
            System.out.flush();
            System.out.println("Model Input Type: " + entry.getValue().getInfo().toString());
            System.out.flush();
        }
            float[][][][] input = new float[1][IMG_SIZE][IMG_SIZE][3];
            for (int y = 0; y < IMG_SIZE; y++) {
                for (int x = 0; x < IMG_SIZE; x++) {
                    double[] pixel = frame.get(y, x);
                    input[0][y][x][0] = (float)(pixel[0] / 255.0);
                    input[0][y][x][1] = (float)(pixel[1] / 255.0);
                    input[0][y][x][2] = (float)(pixel[2] / 255.0);
                }
            }
            OnnxTensor tensor = OnnxTensor.createTensor(env, input);
            String inputName = session.getInputNames().iterator().next();
            System.out.println("Input Info: " + session.getInputMetadata().get(inputName).getInfo().toString());
            OrtSession.Result result = session.run(
                    Collections.singletonMap(inputName, tensor)
            );
            float[][] output = (float[][]) result.get(0).getValue();
            return output[0][1] > 0.90f;
        } catch (Exception e) {
            e.printStackTrace();
            return false;
        }
    }
    private static void sendPhoneNotification(String msg) {
        try {
            HttpClient client = HttpClient.newHttpClient();
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create("https://ntfy.sh/" + NTFY_TOPIC))
                    .POST(HttpRequest.BodyPublishers.ofString(msg))
                    .build();
            client.send(request, HttpResponse.BodyHandlers.ofString());
            System.out.println("[NTFY] Alert sent to phone.");
        } catch (Exception e) {
            System.err.println("[NTFY] Error sending: " + e.getMessage());
        }
    }

} // whole program brace
