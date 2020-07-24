package com.github.brokenegg.transformer;

import android.app.Activity;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;

class Translator {
    static {
        System.loadLibrary("translator");
    }

    private static final String LOG_TAG = "translator";

    private static final String ONNX_MODEL_PATH = "brokenegg-20200719.onnx";
    public static final int ONNX_MODEL_VERSION = 2;
    private static final String SPM_MODEL_PATH = "brokenegg.en-es-ja.spm64k.model";

    private static final long EOS_ID = 2;
    public static final long LANG_CHITCHAT = 1;
    public static final long LANG_ENGLISH = 64000;
    public static final long LANG_SPANISH = 64001;
    public static final long LANG_JAPANESE = 64002;

    public Translator(Activity activity) {
        AssetManager assetManager = activity.getAssets();
        try {
            File externalFilesDir = activity.getExternalFilesDir(null);
            Path onnxModelPath = Paths.get(externalFilesDir.toString(), ONNX_MODEL_PATH);
            loadOnnxModel(onnxModelPath.toString());

            byte[] dat = loadModelAsBytes(assetManager, SPM_MODEL_PATH, 1400000);
            loadSentencePiece(dat, dat.length);
        } catch (Exception ex) {
            Log.d(LOG_TAG, "Exception %s", ex);
        }
    }

    private byte[] loadModelAsBytes(AssetManager assetManager, String fileName, int maxSize) throws IOException {
        InputStream inputStream = assetManager.open(fileName);
        try {
            byte[] buf = new byte[maxSize];
            int len = inputStream.read(buf, 0, buf.length);
            byte[] res = new byte[len];
            System.arraycopy(buf, 0, res, 0, len);
            return res;
        } finally {
            inputStream.close();
        }
    }

    private native boolean loadOnnxModel(String modelPath);
    private native boolean loadSentencePiece(byte[] buf, int len);
    private native String translate(String text, long initialToken, int modelVersion);

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
        return translate(text, langId, ONNX_MODEL_VERSION);
    }
}