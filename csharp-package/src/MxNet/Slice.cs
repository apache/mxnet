/*****************************************************************************
   Copyright 2018 The MxNet.Sharp Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
******************************************************************************/
namespace MxNet
{
    public class Slice
    {
        public Slice(int begin, int? end, int step = 1)
        {
            Begin = begin;
            End = end;
            Step = step;
        }

        public int Begin { get; set; }

        public int? End { get; set; }

        public int Step { get; set; }

        public override string ToString()
        {
            if (End.HasValue)
                return string.Format("{0}:{1}", Begin, End.Value);
            return string.Format("{0}:", Begin);
        }
    }
}