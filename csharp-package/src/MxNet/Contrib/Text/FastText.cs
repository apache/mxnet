using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Contrib.Text
{
    public class FastText : _TokenEmbedding
    {
        public FastText(string pretrained_file_name = "wiki.simple.vec", string embedding_root = null, Func<Shape, NDArray> init_unknown_vec = null,
                    Vocabulary vocabulary = null, Dictionary<string, int> counter = null, int? most_freq_count = null, int min_freq = 1,
                    string unknown_token = "<unk>", string[] reserved_tokens = null)
                        : base(counter, most_freq_count, min_freq, unknown_token, reserved_tokens)
        {
            throw new NotImplementedRelease2Exception();
        }

        public override string GetDownloadFileName(string pretrained_file_name)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
