using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Contrib.Text
{
    public class CompositeEmbedding : _TokenEmbedding
    {
        public CompositeEmbedding(Vocabulary vocabulary, _TokenEmbedding[] token_embeddings, Dictionary<string, int> counter = null,
                                int? most_freq_count = null, int min_freq = 1, string unknown_token = "<unk>", string[] reserved_tokens = null) 
                                : base(counter, most_freq_count, min_freq, unknown_token, reserved_tokens)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
