using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Contrib.Text
{
    public class Vocabulary
    {
        public int Length => throw new NotImplementedRelease2Exception();

        public Dictionary<string, int> TokenToIdx => throw new NotImplementedRelease2Exception();

        public string[] IdxToToken => throw new NotImplementedRelease2Exception();

        public string UnknownToken => throw new NotImplementedRelease2Exception();

        public string[] ReservedTokens => throw new NotImplementedRelease2Exception();

        public Vocabulary(Dictionary<string, int> counter= null, int? most_freq_count= null, int min_freq= 1, string unknown_token= "<unk>",
                            string[] reserved_tokens= null)
        {
            throw new NotImplementedRelease2Exception();
        }

        private void IndexUnknownAndReservedTokens(string unknown_token, string[] reserved_tokens)
        {
            throw new NotImplementedRelease2Exception();
        }

        private void IndexCounterKeys(Dictionary<string, int> counter, string unknown_token, string[] reserved_tokens, int most_freq_count,
                            int min_freq)
        {
            throw new NotImplementedRelease2Exception();
        }

        public int[] ToIndices(string[] tokens)
        {
            throw new NotImplementedRelease2Exception();
        }

        public int[] ToTokens(string[] tokens)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
