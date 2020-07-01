package com.github.brokenegg.transformer;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;

class Translator {
    static {
        System.loadLibrary("tokenize");
    }

    private static String MODEL_PATH = "brokenegg_tf23_fp16.tflite";
    private Interpreter tflite;

    public Translator(Activity activity) {
        AssetManager assetManager = activity.getAssets();
        try {
            AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_PATH);
            FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = inputStream.getChannel();
            long startOffset = fileDescriptor.getStartOffset();
            long declaredLength = fileDescriptor.getDeclaredLength();
            MappedByteBuffer buffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
            Interpreter.Options opt = new Interpreter.Options();
            // GpuDelegate gpuDelegate = new GpuDelegate();
            // opt.addDelegate(gpuDelegate);
            tflite = new Interpreter(buffer, opt);

            byte[] dat = loadModelAsBytes(assetManager, "hoge");
            initTokenizer(dat, dat.length);
        } catch (Exception ex) {

        }
    }

    private byte[] loadModelAsBytes(AssetManager assetManager, String fileName) {
        fileName = "brokenegg.en-es-ja.spm64k.model";
        try {
            InputStream inputStream = assetManager.open(fileName);
            byte[] buf = new byte[10000000];
            int len = inputStream.read(buf, 0, buf.length);
            byte[] res = new byte[len];
            System.arraycopy(buf, 0, res, 0, len);
            inputStream.close();
            return res;
        } catch (Exception ex) {

        }
        return null;
    }

    private native boolean initTokenizer(byte[] buf, int len);
    private native long[] tokenEncode(String text);
    private native String tokenDecode(long[] ids);

    public void run() {
        long[] encodedText = tokenEncode("I'm going to the school today.");
        String decodedText = tokenDecode(encodedText);
        long[][] inputData = new long[][] { encodedText };
        long[][] targetData = new long[][] { new long[10] };
        targetData[0][0] = 64001;
        long[][] outputData = new long[][] { new long[9] };
        tflite.resizeInput(0, new int[] { inputData[0].length, inputData.length });
        tflite.resizeInput(1, new int[] { targetData[0].length, targetData.length });
        tflite.allocateTensors();
        Object[] inputs = new Object[] { inputData, targetData };
        HashMap<Integer, Object> outputs = new HashMap();
        outputs.put(0, outputData);
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        Log.d("tflite", String.format("Size: %d", outputData.length));
    }
}