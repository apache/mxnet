package AI::MXNet::RNN;
use strict;
use warnings;
use AI::MXNet::Function::Parameters;
use AI::MXNet::RNN::IO;
use AI::MXNet::RNN::Cell;
use List::Util qw(max);

=encoding UTF-8

=head1 NAME

    AI::MXNet::RNN - Functions for constructing recurrent neural networks.
=cut

=head1 SYNOPSIS


=head1 DESCRIPTION

    Functions for constructing recurrent neural networks.
=cut

=head2 save_rnn_checkpoint

    Save checkpoint for model using RNN cells.
    Unpacks weight before saving.

    Parameters
    ----------
    cells : AI::MXNet::RNN::Cell or array ref of AI::MXNet::RNN::Cell
        The RNN cells used by this symbol.
    prefix : str
        Prefix of model name.
    epoch : int
        The epoch number of the model.
    symbol : Symbol
        The input symbol
    arg_params : hash ref of str to AI::MXNet::NDArray
        Model parameter, hash ref of name to NDArray of net's weights.
    aux_params : hash ref of str to AI::MXNet::NDArray
        Model parameter, hash ref of name to NDArray of net's auxiliary states.

    Notes
    -----
    - prefix-symbol.json will be saved for symbol.
    - prefix-epoch.params will be saved for parameters.
=cut

method save_rnn_checkpoint(
    AI::MXNet::RNN::Cell::Base|ArrayRef[AI::MXNet::RNN::Cell::Base] $cells,
    Str                                                             $prefix,
    Int                                                             $epoch,
    AI::MXNet::Symbol                                               $symbol,
    HashRef[AI::MXNet::NDArray]                                     $arg_params,
    HashRef[AI::MXNet::NDArray]                                     $aux_params
)
{
    $cells = [$cells] unless ref $cells eq 'ARRAY';
    my %arg_params = %{ $arg_params };
    %arg_params = %{ $_->unpack_weights(\%arg_params) } for @{ $cells };
    AI::MXNet::Module->model_save_checkpoint($prefix, $epoch, $symbol, \%arg_params, $aux_params);
}


=head2 load_rnn_checkpoint

    Load model checkpoint from file.
    Pack weights after loading.

    Parameters
    ----------
    cells : AI::MXNet::RNN::Cell or ir array ref of AI::MXNet::RNN::Cell
        The RNN cells used by this symbol.
    prefix : str
        Prefix of model name.
    epoch : int
        Epoch number of model we would like to load.

    Returns
    -------
    symbol : Symbol
        The symbol configuration of computation network.
    arg_params : hash ref of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : hash ref of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.

    Notes
    -----
    - symbol will be loaded from prefix-symbol.json.
    - parameters will be loaded from prefix-epoch.params.
=cut

method load_rnn_checkpoint(
    AI::MXNet::RNN::Cell::Base|ArrayRef[AI::MXNet::RNN::Cell::Base] $cells,
    Str                                                             $prefix,
    Int                                                             $epoch
)
{
    my ($sym, $arg, $aux) = AI::MXNet::Module->load_checkpoint($prefix, $epoch);
    $cells = [$cells] unless ref $cells eq 'ARRAY';
    $arg = $_->pack_weights($arg) for @{ $cells };
    return ($sym, $arg, $aux);
}

=head2 do_rnn_checkpoint

    Make a callback to checkpoint Module to prefix every epoch.
    unpacks weights used by cells before saving.

    Parameters
    ----------
    cells : subclass of RNN::Cell
        RNN cells used by this module.
    prefix : str
        The file prefix to checkpoint to
    period : int
        How many epochs to wait before checkpointing. Default is 1.

    Returns
    -------
    callback : function
        The callback function that can be passed as iter_end_callback to fit.
=cut

method do_rnn_checkpoint(
    AI::MXNet::RNN::Cell::Base|ArrayRef[AI::MXNet::RNN::Cell::Base]  $cells,
    Str                                                              $prefix,
    Int                                                              $period
)
{
    $period = max(1, $period);
    return sub {
        my ($iter_no, $sym, $arg, $aux) = @_;
        if (($iter_no + 1) % $period == 0)
        {
            __PACKAGE__->save_rnn_checkpoint($cells, $prefix, $iter_no+1, $sym, $arg, $aux);
        }
    };
}

## In order to closely resemble the Python's usage
method RNNCell(@args)            { AI::MXNet::RNN::Cell->new(@args % 2 ? ('num_hidden', @args) : @args) }
method LSTMCell(@args)           { AI::MXNet::RNN::LSTMCell->new(@args % 2 ? ('num_hidden', @args) : @args) }
method GRUCell(@args)            { AI::MXNet::RNN::GRUCell->new(@args % 2 ? ('num_hidden', @args) : @args) }
method FusedRNNCell(@args)       { AI::MXNet::RNN::FusedCell->new(@args % 2 ? ('num_hidden', @args) : @args) }
method SequentialRNNCell(@args)  { AI::MXNet::RNN::SequentialCell->new(@args) }
method BidirectionalCell(@args)  { AI::MXNet::RNN::BidirectionalCell->new(@args) }
method DropoutCell(@args)        { AI::MXNet::RNN::DropoutCell->new(@args) }
method ZoneoutCell(@args)        { AI::MXNet::RNN::ZoneoutCell->new(@args) }
method encode_sentences(@args)   { AI::MXNet::RNN::IO->encode_sentences(@args) }
method BucketSentenceIter(@args)
{
    my $sentences  = shift(@args);
    my $batch_size = shift(@args);
    AI::MXNet::BucketSentenceIter->new(sentences => $sentences, batch_size => $batch_size, @args);
}

1;
