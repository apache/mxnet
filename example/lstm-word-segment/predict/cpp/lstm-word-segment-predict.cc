/*!
 * Copyright (c) 2016
 * \file lstm-word-segment-predict.cc
 * \brief C++ predict example of mxnet : lstm word segment
 */

#include <stdio.h>

// Path for c_predict_api
#include <mxnet/c_predict_api.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>

using namespace std;

class BufferFile {
    public:
        std::string file_path_;
        int length_;
        char *buffer_;

        explicit BufferFile(std::string file_path): file_path_(file_path) {
            std::ifstream ifs(file_path_.c_str(), std::ios::in | std::ios::binary);
            if (!ifs) {
                std::cerr << "Can't open the file. Please check " << file_path << ". \n";
            }
            ifs.seekg(0, std::ios::end);
            length_ = ifs.tellg();
            ifs.seekg(0, std::ios::beg);
            std::cerr << file_path_.c_str() << " ... " << length_ << " bytes\n";

            buffer_ = new char[sizeof(char) * length_];
            ifs.read(buffer_, length_);
            ifs.close();
        }

        int GetLength() {
            return length_;
        }

        char *GetBuffer() {
            return buffer_;
        }

        ~BufferFile() {
            delete [] buffer_;
            buffer_ = NULL;
        }
};

char PrintOutputResult(const std::vector<float> &data, unordered_map<int, char> &idx2label) {
    float best_acc = 0.0;
    int best_idx = 0;

    for (int i = 0; i < static_cast<int>(data.size()); ++i) {
        if (data[i] > best_acc) {
            best_acc = data[i];
            best_idx = i;
        }
    }

    return idx2label[best_idx];
}

void ReadVocabMap(const std::string &map_file, unordered_map<string, int> &dict) {
    ifstream m_file(map_file.c_str());
    if (!m_file) {
        cerr << "open file " << map_file << " failed." << endl;
    }

    string line;
    while (getline(m_file, line)) {
        int idx = line.find_first_of('');
        if (idx != -1) {
            string w = line.substr(0, idx);
            int index = atoi(line.substr(idx+1).c_str());
            dict[w] = index;
        }
    }

    m_file.close();
}

int GetUTF8Vec(const string &text, vector<std::string> &utf8_array) {
    utf8_array.clear();
    int idx = 0;
    while (idx < text.size()) {
        if ((text[idx] & 0x80) == 0) {  // single byte character
            utf8_array.push_back(text.substr(idx, 1));
            ++idx;
        } else if ((text[idx] & 0xE0) == 0xC0) {  // double bytes
            utf8_array.push_back(text.substr(idx, 2));
            idx += 2;
        } else if ((text[idx] & 0xF0) == 0xE0) {  // triple bytes
            utf8_array.push_back(text.substr(idx, 3));
            idx += 3;
        } else {
            ++idx;
        }
    }
    return idx;
}

int main(int argc, char *argv[]) {
    unordered_map<int, char> idx2label;
    idx2label[0] = 'B';
    idx2label[1] = 'M';
    idx2label[2] = 'E';
    idx2label[3] = 'S';

    // load vocabulary
    unordered_map<string, int> vocab_dict;
    ReadVocabMap("../vocab_map", vocab_dict);

    string symbol_file = "../../checkpoint/lstm-symbol.json";
    string param_file = "../../checkpoint/lstm-0099.params";
    BufferFile symbol_data(symbol_file);
    BufferFile param_data(param_file);

    int dev_type = 2; // 1: cpu, 2: gpu
    int dev_id = 0; // arbitrary
    mx_uint num_input_nodes = 3; // data, init_c, init_h
    const char *input_key[3] = { "data", "l0_init_c", "l0_init_h" };
    const char **input_keys = input_key;

    mx_uint batch_size = 1;
    mx_uint num_hidden = 300;
    mx_uint context_size = 7;

    const mx_uint input_shape_indptr[4] = {0, 2, 4, 6};
    const mx_uint input_shape_data[6] = { batch_size, context_size, batch_size, num_hidden, batch_size, num_hidden };

    PredictorHandle out = 0; // alias for void *

    // Create Predictor
    MXPredCreate((const char *)symbol_data.GetBuffer(),
                (const char *)param_data.GetBuffer(),
                static_cast<size_t>(param_data.GetLength()),
                dev_type,
                dev_id,
                num_input_nodes,
                input_keys,
                input_shape_indptr,
                input_shape_data,
                &out);

    vector<mx_float> init_c = vector<mx_float>(batch_size * num_hidden);
    vector<mx_float> init_h = vector<mx_float>(batch_size * num_hidden);
    vector<mx_float> data = vector<mx_float>(batch_size * context_size);
    string input_str;
    vector<string> utf8_arr;
    int window = (int) (context_size - 1) / 2;
    while (getline(cin, input_str)) {
        utf8_arr.clear();
        GetUTF8Vec(input_str, utf8_arr);

        init_c.clear(); init_h.clear();
        MXPredSetInput(out, "l0_init_c", init_c.data(), batch_size * num_hidden);
        MXPredSetInput(out, "l0_init_h", init_h.data(), batch_size * num_hidden);
        for (size_t i = 0; i < utf8_arr.size(); ++i) {
            data.clear();
            data.resize(batch_size * context_size);
            for (int j = -window; j <= window; ++j) {
                if (i+j < 0 || i+j >= utf8_arr.size()) {
                    data[j+window] = vocab_dict["P"];
                } else {
                    if (vocab_dict.find(utf8_arr[i]) != vocab_dict.end()) {
                        data[j+window] = vocab_dict[utf8_arr[i+j]];
                    } else {
                        data[j+window] = vocab_dict["U"];
                    }
                }
            }
            MXPredSetInput(out, "data", data.data(), batch_size * context_size);
            // Do Predict
            MXPredForward(out);
            // Get Output
            mx_uint output_index = 0;
            mx_uint *shape = 0;
            mx_uint shape_len;

            MXPredGetOutputShape(out, output_index, &shape, &shape_len);
            size_t size = 1;
            for (mx_uint k = 0; k < shape_len; ++k) size *= shape[k];
            vector<float> result(size);

            MXPredGetOutput(out, output_index, &(result[0]), size);

            // Print Output Label
            char char_label = PrintOutputResult(result, idx2label);
            switch(char_label) {
                case 'B': cout << utf8_arr[i]; break;
                case 'M': cout << utf8_arr[i]; break;
                case 'E': cout << utf8_arr[i] << " "; break;
                case 'S': cout << utf8_arr[i] << " "; break;
            }
        }
        cout << endl;
    }
    MXPredFree(out);
}
