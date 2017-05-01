#!/usr/bin/perl
use strict;
use warnings;
use AI::MXNet ('mx');

## preparing the samples
## to train our network
sub samples {
    my($batch_size, $func) = @_;
    # get samples
    my $n = 16384;
    ## creates a pdl with $n rows and two columns with random
    ## floats in the range between 0 and 1
    my $data = PDL->random(2, $n);
    ## creates the pdl with $n rows and one column with labels
    ## labels are floats that either sum or product, etc of
    ## two random values in each corresponding row of the data pdl
    my $label = $func->($data->slice('0,:'), $data->slice('1,:'));
    # partition into train/eval sets
    my $edge = int($n / 8);
    my $validation_data = $data->slice(":,0:@{[ $edge - 1 ]}");
    my $validation_label = $label->slice(":,0:@{[ $edge - 1 ]}");
    my $train_data = $data->slice(":,$edge:");
    my $train_label = $label->slice(":,$edge:");
    # build iterators around the sets
    return(mx->io->NDArrayIter(
        batch_size => $batch_size,
        data => $train_data,
        label => $train_label,
    ), mx->io->NDArrayIter(
        batch_size => $batch_size,
        data => $validation_data,
        label => $validation_label,
    ));
}

## the network model
sub nn_fc {
    my $data = mx->sym->Variable('data');
    my $ln = mx->sym->exp(mx->sym->FullyConnected(
        data => mx->sym->log($data),
        num_hidden => 1,
    ));
    my $wide = mx->sym->Concat($data, $ln);
    my $fc = mx->sym->FullyConnected(
	$wide,
	num_hidden => 1
    );
    return mx->sym->MAERegressionOutput(data => $fc, name => 'softmax');
}

sub learn_function {
    my(%args) = @_;
    my $func = $args{func};
    my $batch_size = $args{batch_size}//128;
    my($train_iter, $eval_iter) = samples($batch_size, $func);
    my $sym = nn_fc();

    ## call as ./calculator.pl 1 to just print model and exit
    if($ARGV[0]) {
        my @dsz = @{$train_iter->data->[0][1]->shape};
        my @lsz = @{$train_iter->label->[0][1]->shape};
        my $shape = {
            data          => [ $batch_size, splice @dsz,  1 ],
            softmax_label => [ $batch_size, splice @lsz, 1 ],
        };
        print mx->viz->plot_network($sym, shape => $shape)->graph->as_png;
        exit;
    }

    my $model = mx->mod->Module(
        symbol => $sym,
        context => mx->cpu(),
    );
    $model->fit($train_iter,
        eval_data => $eval_iter,
        optimizer => 'adam',
        optimizer_params => {
            learning_rate => $args{lr}//0.01,
            rescale_grad => 1/$batch_size,
            lr_scheduler  => AI::MXNet::FactorScheduler->new(
        	step => 100,
        	factor => 0.99
            )
        },
        eval_metric => 'mse',
        num_epoch => $args{epoch}//25,
    );

    # refit the model for calling on 1 sample at a time
    my $iter = mx->io->NDArrayIter(
        batch_size => 1,
        data => PDL->pdl([[ 0, 0 ]]),
        label => PDL->pdl([[ 0 ]]),
    );
    $model->reshape(
        data_shapes => $iter->provide_data,
        label_shapes => $iter->provide_label,
    );

    # wrap a helper around making predictions
    my ($arg_params) = $model->get_params;
    for my $k (sort keys %$arg_params)
    {
	print "$k -> ". $arg_params->{$k}->aspdl."\n";
    }
    return sub {
        my($n, $m) = @_;
        return $model->predict(mx->io->NDArrayIter(
            batch_size => 1,
            data => PDL->new([[ $n, $m ]]),
        ))->aspdl->list;
    };
}

my $add = learn_function(func => sub {
    my($n, $m) = @_;
    return $n + $m;
});
my $sub = learn_function(func => sub {
    my($n, $m) = @_;
    return $n - $m;
}, batch_size => 50, epoch => 40);
my $mul = learn_function(func => sub {
    my($n, $m) = @_;
    return $n * $m;
}, batch_size => 50, epoch => 40);
my $div = learn_function(func => sub {
    my($n, $m) = @_;
    return $n / $m;
}, batch_size => 10, epoch => 80);


print "12345 + 54321 ≈ ", $add->(12345, 54321), "\n";
print "188 - 88 ≈ ", $sub->(188, 88), "\n";
print "250 * 2 ≈ ", $mul->(250, 2), "\n";
print "250 / 2 ≈ ", $div->(250, 2), "\n";

