using System;
using System.Collections.Generic;
using System.Text;

namespace MxNet.Gluon.RNN.Cell
{
    public class VariationalDropoutCell : ModifierCell
    {
        public VariationalDropoutCell(RecurrentCell base_cell, float drop_inputs = 0, float drop_states = 0, float drop_outputs = 0) : base(base_cell)
        {
            throw new NotImplementedException();
        }

        public override string Alias()
        {
            return "vardrop";
        }

        public override void Reset()
        {
            base.Reset();
            throw new NotImplementedException();
        }

        private void _initialize_input_masks(NDArrayOrSymbolList inputs, NDArrayOrSymbolList states)
        {
            throw new NotImplementedException();
        }

        private void _initialize_output_masks(NDArrayOrSymbol output)
        {
            throw new NotImplementedException();
        }

        public override (NDArrayOrSymbol, NDArrayOrSymbol[]) HybridForward(NDArrayOrSymbol x, params NDArrayOrSymbol[] args)
        {
            throw new NotImplementedException();
        }
    }
}
