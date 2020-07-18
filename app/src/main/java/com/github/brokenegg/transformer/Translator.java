package com.github.brokenegg.transformer;

import android.app.Activity;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.InputStream;

class Translator {
    static {
        System.loadLibrary("translator");
    }

    private static final String LOG_TAG = "translator";

    private static final String MODEL_PATH = "brokenegg_tf22.tflite";
    private static final boolean useDynamicShape = false;
    private static final long EOS_ID = 2;
    public static final long LANG_CHITCHAT = 1;
    public static final long LANG_ENGLISH = 64000;
    public static final long LANG_SPANISH = 64001;
    public static final long LANG_JAPANESE = 64002;

    public Translator(Activity activity) {
        AssetManager assetManager = activity.getAssets();
        try {
            File path = activity.getExternalFilesDir(null);
            byte[] dat = null; //loadModelAsBytes(assetManager, "brokenegg-20200711_torch.onnx", 350000000);
            initModel(dat, 0); //dat.length);

            dat = loadModelAsBytes(assetManager, "brokenegg.en-es-ja.spm64k.model", 1400000);
            initTokenizer(dat, dat.length);
        } catch (Exception ex) {
            Log.d(LOG_TAG, "Exception %s", ex);
        }
    }

    private byte[] loadModelAsBytes(AssetManager assetManager, String fileName, int maxSize) {
        try {
            InputStream inputStream = assetManager.open(fileName);
            byte[] buf = new byte[maxSize];
            int len = inputStream.read(buf, 0, buf.length);
            byte[] res = new byte[len];
            System.arraycopy(buf, 0, res, 0, len);
            inputStream.close();
            return res;
        } catch (Exception ex) {

        }
        return null;
    }

    private native boolean initModel(byte[] buf, int len);
    private native boolean initTokenizer(byte[] buf, int len);
    private native String translate(String text, long initialToken);

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
        return translate(text, langId);
    }
}