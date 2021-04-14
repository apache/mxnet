using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Contrib.Text
{
    public class _TokenEmbedding : Vocabulary
    {
        public int VecLen 
        {
            get
            {
                throw new NotImplementedRelease2Exception();
            }
        }

        public NDArrayList[] IdxToVec
        {
            get
            {
                throw new NotImplementedRelease2Exception();
            }
        }

        public _TokenEmbedding(Dictionary<string, int> counter = null, int? most_freq_count = null, int min_freq = 1, 
                                string unknown_token = "<unk>", string[] reserved_tokens = null) 
            : base(counter, most_freq_count, min_freq, unknown_token, reserved_tokens)
        {
        }

        public virtual string GetDownloadFileName(string pretrained_file_name)
        {
            return pretrained_file_name;
        }

        public virtual string GetPreTrainedFileUrl(string pretrained_file_name)
        {
            throw new NotImplementedRelease2Exception();
        }

        public string GetPreTrainedFile(string embedding_root, string pretrained_file_name)
        {
            throw new NotImplementedRelease2Exception();
        }

        public void LoadEmbedding(string pretrained_file_path, string elem_delim, Func<Shape, NDArray> init_unknown_vec, string encoding= "utf8")
        {
            throw new NotImplementedRelease2Exception();
        }
        
        public void IndexTokensFromVocabulary(Vocabulary vocabulary)
        {
            throw new NotImplementedRelease2Exception();
        }

        public void SetIdxToVecByEmbeddings(_TokenEmbedding[] token_embeddings, int vocab_len, string[] vocab_idx_to_token)
        {
            throw new NotImplementedRelease2Exception();
        }

        public void BuildEmbeddingForVocabulary(Vocabulary vocabulary)
        {
            throw new NotImplementedRelease2Exception();
        }

        public NDArray GetVecsByTokens(string[] tokens, bool lower_case_backup= false)
        {
            throw new NotImplementedRelease2Exception();
        }

        public void UpdateTokenVectors(string[] tokens, NDArray new_vectors)
        {
            throw new NotImplementedRelease2Exception();
        }

        public void CheckPretrainedFileNames(string pretrained_file_name)
        {
            throw new NotImplementedRelease2Exception();
        }
    }
}
