#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "sentencepiece_processor.h"

#define LOG_TAG "tokenize"

static bool g_initialized = false;
static sentencepiece::SentencePieceProcessor g_processor;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_github_brokenegg_transformer_Translator_initTokenizer(JNIEnv *env, jobject thiz,
                                                               jbyteArray buf, jint len) {
    if (g_initialized) return true;
    auto& sp = g_processor;

    jbyte *pbuf = env->GetByteArrayElements(buf, 0);
    std::string model(reinterpret_cast<char*>(pbuf), len);
    env->ReleaseByteArrayElements(buf, pbuf, 0);

    const auto status = sp.LoadFromSerializedProto(model);
    if (!status.ok()) {
        return false;
    }

    g_initialized = true;
    return true;
}

extern "C"
JNIEXPORT jlongArray JNICALL
Java_com_github_brokenegg_transformer_Translator_tokenEncode(JNIEnv *env, jobject thiz,
                                                             jstring text) {
    if (!g_initialized) return NULL;
    auto& sp = g_processor;

    const char *ptext = env->GetStringUTFChars(text, NULL);
    jint len = env->GetStringUTFLength(text);
    std::string ctext(ptext, len);
    env->ReleaseStringUTFChars(text, ptext);

    std::vector<int> sps;
    auto status = sp.Encode(ctext, &sps);
    if (!status.ok()) {
        return NULL;
    }

    std::vector<long> cresult(sps.cbegin(), sps.cend());

    jlongArray result = env->NewLongArray(sps.size());
    if (result == NULL) {
        return NULL;
    }

    env->SetLongArrayRegion(result, 0, cresult.size(), cresult.data());
    return result;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_github_brokenegg_transformer_Translator_tokenDecode(JNIEnv *env, jobject thiz,
                                                             jlongArray ids) {
    if (!g_initialized) return NULL;
    auto& sp = g_processor;

    jsize len = env->GetArrayLength(ids);
    jlong *pids = env->GetLongArrayElements(ids, 0);
    std::vector<int> cids(pids, pids + len);
    env->ReleaseLongArrayElements(ids, pids, 0);

    std::string cresult;
    auto status = sp.Decode(cids, &cresult);
    if (!status.ok()) {
        return NULL;
    }

    jstring result = env->NewStringUTF(cresult.c_str());

    return result;
}