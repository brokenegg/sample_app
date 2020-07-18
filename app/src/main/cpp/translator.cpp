#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <sentencepiece_processor.h>
#include <onnxruntime_cxx_api.h>


#define LOG_TAG "tokenize"

static void dumpvec(const std::vector<int64_t>& a) {
    for (int i = 0; i < a.size(); i++) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "dump %d: %d", i, (int)a[i]);
    }
}

class Translator {
private:
    Ort::Env onnx_env;
    Ort::Session *session_;
    sentencepiece::SentencePieceProcessor* processor_;
    bool processor_ready_;

public:
    Translator(void *buf, size_t len) :
        onnx_env{},
        session_{nullptr}, //{onnx_env, "/storage/emulated/0/Android/data/com.github.brokenegg.transformer/files/brokenegg-20200711.onnx", Ort::SessionOptions{nullptr}},
        processor_ready_(false) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Hoge");
        const char *a = "/storage/emulated/0/Android/data/com.github.brokenegg.transformer/files/brokenegg-20200711_torch.onnx";
        FILE *fp = fopen(a, "r");
        if (fp) {
            __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Hoge0");
            char bufe[1024];
            int lene = fread(bufe, 1, 1024, fp);
            bufe[100] = 0;
            __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Hoge1: %s", bufe);
            fclose(fp);
        }
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Hoge2");
        session_ = new Ort::Session(onnx_env,
                "/storage/emulated/0/Android/data/com.github.brokenegg.transformer/files/brokenegg-20200711_torch.onnx",
                Ort::SessionOptions{nullptr});
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Hage");
    }

    bool load_vocab(std::string model) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Huge");
        if (processor_ready_) return true;
        processor_ = new sentencepiece::SentencePieceProcessor();
        const auto status = processor_->LoadFromSerializedProto(model);
        if (!status.ok()) {
            return false;
        }
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Loaded");
        processor_ready_ = true;
        return true;
    }

    bool translate(std::vector<int64_t>& inputs, std::vector<int64_t>& targets, std::vector<int64_t>* outputs) {
        Ort::Value inputs_tensor_{nullptr};
        std::array<int64_t, 2> inputs_shape_{1, static_cast<int64_t>(inputs.size())};

        Ort::Value targets_tensor_{nullptr};
        std::array<int64_t, 2> targets_shape_{1, static_cast<int64_t>(targets.size())};

        std::array<int64_t, 2> outputs_shape_{1, static_cast<int64_t>(targets.size())};

        Ort::Value input_tensors[2] = {
                Ort::Value(nullptr),
                Ort::Value(nullptr)
        };
        Ort::Value outputs_tensor_{nullptr};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensors[0] = Ort::Value::CreateTensor<int64_t>(memory_info, inputs.data(), inputs.size(),
                                                           inputs_shape_.data(), inputs_shape_.size());
        input_tensors[1] = Ort::Value::CreateTensor<int64_t>(memory_info, targets.data(), targets.size(),
                                                            targets_shape_.data(), targets_shape_.size());
        outputs_tensor_ = Ort::Value::CreateTensor<int64_t>(memory_info, outputs->data(), outputs->size(),
                outputs_shape_.data(), outputs_shape_.size());

        const char* input_names[] = {"inputs", "targets"};
        const char* output_names[] = {"outputs"};


        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Run");
        dumpvec(inputs);
        dumpvec(targets);
        dumpvec(*outputs);
        session_->Run(Ort::RunOptions{nullptr}, input_names,
                input_tensors, 2, output_names, &outputs_tensor_, 1);
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Ran");
        return true;
    }

    bool translate(const std::string& input_text, int64_t initial_token, std::string* output_text) {
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Euge");
        if (!processor_ready_) return false;

        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Encoding");
        std::vector<int> input_ids;
        auto status = processor_->Encode(input_text, &input_ids);
        if (!status.ok()) {
            return false;
        }
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Encoded");
        input_ids.push_back(2); //EOS_ID);

        std::vector<int64_t> input_long_ids(input_ids.cbegin(), input_ids.cend());

        std::vector<int64_t> target_long_ids;
        target_long_ids.push_back(initial_token);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Initial %d", (int)initial_token);

        std::vector<int64_t> output_long_ids;

        for (int i = 1; i <= 20; i++) {
            __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Predicting token %d", i);
            output_long_ids.resize(i);
            translate(input_long_ids, target_long_ids, &output_long_ids);
            int64_t lastId = output_long_ids[i - 1];
            __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Last ID: %d", (int)lastId);
            if (lastId == 2) {
                break;
            }
            target_long_ids.push_back(lastId);
        }

        std::vector<int> output_ids(output_long_ids.cbegin(), output_long_ids.cend());

        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Decoding");
        status = processor_->Decode(output_ids, output_text);
        if (!status.ok()) {
            return false;
        }
        __android_log_print(ANDROID_LOG_WARN, LOG_TAG, "Decoded");

        return true;
    }
};

static Translator* g_translator = nullptr;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_github_brokenegg_transformer_Translator_initModel(JNIEnv *env, jobject thiz,
                                                           jbyteArray buf, jint len) {
    if (g_translator != nullptr) return true;
    //jbyte *pbuf = env->GetByteArrayElements(buf, 0);
    g_translator = new Translator(nullptr, len);
    //env->ReleaseByteArrayElements(buf, pbuf, 0);
    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_github_brokenegg_transformer_Translator_initTokenizer(JNIEnv *env, jobject thiz,
                                                               jbyteArray buf, jint len) {
    if (g_translator == nullptr) return false;
     jbyte *pbuf = env->GetByteArrayElements(buf, 0);
    std::string model(reinterpret_cast<char*>(pbuf), len);
    env->ReleaseByteArrayElements(buf, pbuf, 0);
    g_translator->load_vocab(model);
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_github_brokenegg_transformer_Translator_translate(JNIEnv *env, jobject thiz,
                                                           jstring text, jlong initialToken) {
    if (g_translator == nullptr) return NULL;
    const char *ptext = env->GetStringUTFChars(text, NULL);
    jint len = env->GetStringUTFLength(text);
    std::string input_text(ptext, len);
    env->ReleaseStringUTFChars(text, ptext);

    std::string output_text;

    if (!g_translator->translate(input_text, initialToken, &output_text)) {
        return NULL;
    }
    jstring result = env->NewStringUTF(output_text.c_str());

    return result;
}