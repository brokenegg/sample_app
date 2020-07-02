package com.github.brokenegg.transformer;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.HashMap;

class Translator {
    static {
        System.loadLibrary("tokenize");
    }

    private static final String LOG_TAG = "translator";

    private static final String MODEL_PATH = "brokenegg_tf22.tflite";
    private static final boolean useDynamicShape = false;
    private static final long EOS_ID = 2;
    public static final long LANG_CHITCHAT = 1;
    public static final long LANG_ENGLISH = 64000;
    public static final long LANG_SPANISH = 64001;
    public static final long LANG_JAPANESE = 64002;
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
            if (!useDynamicShape) {
                NnApiDelegate delegate = new NnApiDelegate();
                opt.addDelegate(delegate);
            }
            tflite = new Interpreter(buffer, opt);
            if (!useDynamicShape) {
                tflite.allocateTensors();
            }

            byte[] dat = loadModelAsBytes(assetManager, "brokenegg.en-es-ja.spm64k.model");
            initTokenizer(dat, dat.length);
        } catch (Exception ex) {
            Log.d(LOG_TAG, "Exception %s", ex);
        }
    }

    private byte[] loadModelAsBytes(AssetManager assetManager, String fileName) {
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

    private long[] padData(long[] ids, int size) {
        if (ids.length == size) {
            return ids;
        }
        long[] res = new long[size];
        System.arraycopy(ids, 0, res, 0, ids.length);
        if (ids.length < size) {
            res[ids.length] = EOS_ID;
        }
        return res;
    }

    public String run(String text, long langId, int maxLength) {
        long[] encodedText = tokenEncode(text);
        if (!useDynamicShape) {
            encodedText = padData(encodedText, maxLength);
        }
        long[][] inputData = new long[][] { encodedText };
        long[][] targetData = new long[][] { new long[maxLength] };
        targetData[0][0] = langId;
        long[][] outputData = new long[][] { new long[maxLength - 1] };

        int i = 0;
        for (i = 0; i < maxLength - 1; i++) {
            if (i == 0 && useDynamicShape) {
                tflite.resizeInput(0, new int[]{inputData[0].length, inputData.length});
                tflite.resizeInput(1, new int[]{targetData[0].length, targetData.length});
                tflite.allocateTensors();
            }
            Object[] inputs = new Object[]{inputData, targetData};
            HashMap<Integer, Object> outputs = new HashMap();
            outputs.put(0, outputData);
            tflite.runForMultipleInputsOutputs(inputs, outputs);
            long predict = outputData[0][i];
            Log.d("tflite", String.format("Translate: %d", predict));
            if (predict == EOS_ID) {
                break;
            }
            targetData[0][i + 1] = predict;
        }
        long[] result = new long[i];
        System.arraycopy(targetData[0], 1, result, 0, i);
        String decodedText = tokenDecode(result);
        return decodedText;
    }
}