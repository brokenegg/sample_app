#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <sentencepiece_processor.h>
#include <onnxruntime_cxx_api.h>

#define LOG_TAG "translate"

#define EOS_ID 2

class Translator
{
private:
    Ort::Env env_;
    Ort::Session session_;
    sentencepiece::SentencePieceProcessor processor_;

    static void DumpVec(const std::vector<int64_t>& x, const char* name) {
        for (size_t i = 0; i < x.size(); i++) {
            int64_t v = static_cast<int>(x[i]);
            __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "vector %s[%ld]: %ld", name, i, v);
        }
    }

public:
    Translator() :
        env_{},
        session_{nullptr},
        processor_{} {}

    void LoadModel(const ORTCHAR_T* model_path)
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Loading ONNX model");
        session_ = Ort::Session(env_,
                model_path,
                Ort::SessionOptions{nullptr});
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Loaded");
    }

    void LoadVocab(const std::string& model)
    {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Loading SentencePiece model");
        const auto status = processor_.LoadFromSerializedProto(model);
        if (!status.ok()) throw std::exception();
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Loaded");
    }

    void Translate(const std::string& input_text,
            int64_t initial_token, int model_version,
            std::string* output_text) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Encoding");
        std::vector<int> input_ids;
        auto status = processor_.Encode(input_text, &input_ids);
        if (!status.ok()) throw std::exception();
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Encoded");
        input_ids.push_back(EOS_ID);

        std::vector<int64_t> input_long_ids(input_ids.cbegin(), input_ids.cend());

        std::vector<int64_t> target_long_ids;
        target_long_ids.push_back(initial_token);

        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Initial %d", (int)initial_token);

        std::vector<int64_t> output_long_ids;

        switch (model_version) {
            case 1:
                for (size_t i = 1; i <= 20; i++) {
                    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Predicting token %ld", i);
                    TranslateStep(input_long_ids, target_long_ids, output_long_ids);
                    int64_t lastId = output_long_ids[i - 1];
                    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Last ID: %ld", lastId);
                    if (lastId == EOS_ID) {
                        break;
                    }
                    target_long_ids.push_back(lastId);
                }
                break;
            case 2:
                TranslateStep(input_long_ids, target_long_ids, output_long_ids);
                output_long_ids.erase(output_long_ids.begin());
                break;
            default:
                throw std::exception();
        }
        std::vector<int> output_ids(output_long_ids.cbegin(), output_long_ids.cend());

        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Decoding");
        status = processor_.Decode(output_ids, output_text);
        if (!status.ok()) throw std::exception();
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Decoded");
    }

    void Translate2(const std::string& input_text, int64_t initial_token, std::string* output_text) {
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Encoding");
        std::vector<int> input_ids;
        auto status = processor_.Encode(input_text, &input_ids);
        if (!status.ok()) throw std::exception();
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Encoded");
        input_ids.push_back(EOS_ID);

        std::vector<int64_t> input_long_ids(input_ids.cbegin(), input_ids.cend());

        std::vector<int64_t> target_long_ids;
        target_long_ids.push_back(initial_token);

        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Initial %d", (int)initial_token);

        std::vector<int64_t> output_long_ids;

        std::vector<int> output_ids(output_long_ids.cbegin(), output_long_ids.cend());

        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Decoding");
        status = processor_.Decode(output_ids, output_text);
        if (!status.ok()) throw std::exception();
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Decoded");
    }
private:
    void TranslateStep(std::vector<int64_t>& inputs, std::vector<int64_t>& targets, std::vector<int64_t>& outputs) {
        DumpVec(inputs, "inputs");
        DumpVec(targets, "targets");

        // Define tensor shapes
        std::array<int64_t, 2> inputs_shape{1, static_cast<int64_t>(inputs.size())};
        std::array<int64_t, 2> targets_shape{1, static_cast<int64_t>(targets.size())};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::array<Ort::Value, 2> input_tensors{
                Ort::Value::CreateTensor<int64_t>(
                        memory_info, inputs.data(), inputs.size(),
                        inputs_shape.data(), inputs_shape.size()),
                Ort::Value::CreateTensor<int64_t>(
                        memory_info, targets.data(), targets.size(),
                        targets_shape.data(), targets_shape.size())
        };

        const char* input_names[] = {"inputs", "targets"};
        const char* output_names[] = {"outputs"};

        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "Start inference");
        std::vector<Ort::Value> output_tensors = session_.Run(Ort::RunOptions{nullptr},
                      input_names, input_tensors.data(), input_tensors.size(),
                      output_names, 1);
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "End inference");

        auto& output_tensor = output_tensors[0];
        auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
        int64_t* f = output_tensor.GetTensorMutableData<int64_t>();
        size_t total_len = type_info.GetElementCount();
        outputs.resize(total_len);
        std::copy(f, f + total_len, outputs.begin());
        DumpVec(outputs, "outputs");
    }
};

static Translator* g_translator = nullptr;

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_github_brokenegg_transformer_Translator_loadSentencePiece(JNIEnv *env, jobject thiz,
                                                                   jbyteArray buf, jint len) {
    if (g_translator == nullptr) {
        g_translator = new Translator{};
    }

    jbyte *pbuf = env->GetByteArrayElements(buf, 0);
    std::string model(reinterpret_cast<char*>(pbuf), len);
    env->ReleaseByteArrayElements(buf, pbuf, 0);
    g_translator->LoadVocab(model);
    return true;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_github_brokenegg_transformer_Translator_loadOnnxModel(JNIEnv *env, jobject thiz,
                                                               jstring model_path) {
    if (g_translator == nullptr) {
        g_translator = new Translator{};
    }

    const char *p_model_path = env->GetStringUTFChars(model_path, NULL);
    jint len = env->GetStringUTFLength(model_path);
    g_translator->LoadModel(p_model_path);
    env->ReleaseStringUTFChars(model_path, p_model_path);
    return true;
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_github_brokenegg_transformer_Translator_translate(JNIEnv *env, jobject thiz,
                                                           jstring text, jlong initialToken, jint modelVersion) {
    if (g_translator == nullptr) return NULL;
    const char *ptext = env->GetStringUTFChars(text, NULL);
    jint len = env->GetStringUTFLength(text);
    std::string input_text(ptext, len);
    env->ReleaseStringUTFChars(text, ptext);

    std::string output_text;
    g_translator->Translate(input_text, initialToken, modelVersion, &output_text);
    jstring result = env->NewStringUTF(output_text.c_str());

    return result;
}
