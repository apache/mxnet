# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

package AI::MXNet::Visualization;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use JSON::PP;

=encoding UTF-8

=head1 NAME

    AI::MXNet::Vizualization - Vizualization support for Perl interface to MXNet machine learning library

=head1 SYNOPSIS

    use strict;
    use warnings;
    use AI::MXNet qw(mx);

    ### model
    my $data = mx->symbol->Variable('data');
    my $conv1= mx->symbol->Convolution(data => $data, name => 'conv1', num_filter => 32, kernel => [3,3], stride => [2,2]);
    my $bn1  = mx->symbol->BatchNorm(data => $conv1, name => "bn1");
    my $act1 = mx->symbol->Activation(data => $bn1, name => 'relu1', act_type => "relu");
    my $mp1  = mx->symbol->Pooling(data => $act1, name => 'mp1', kernel => [2,2], stride =>[2,2], pool_type=>'max');

    my $conv2= mx->symbol->Convolution(data => $mp1, name => 'conv2', num_filter => 32, kernel=>[3,3], stride=>[2,2]);
    my $bn2  = mx->symbol->BatchNorm(data => $conv2, name=>"bn2");
    my $act2 = mx->symbol->Activation(data => $bn2, name=>'relu2', act_type=>"relu");
    my $mp2  = mx->symbol->Pooling(data => $act2, name => 'mp2', kernel=>[2,2], stride=>[2,2], pool_type=>'max');


    my $fl   = mx->symbol->Flatten(data => $mp2, name=>"flatten");
    my $fc1  = mx->symbol->FullyConnected(data => $fl,  name=>"fc1", num_hidden=>30);
    my $act3 = mx->symbol->Activation(data => $fc1, name=>'relu3', act_type=>"relu");
    my $fc2  = mx->symbol->FullyConnected(data => $act3, name=>'fc2', num_hidden=>10);
    my $softmax = mx->symbol->SoftmaxOutput(data => $fc2, name => 'softmax');

    ## creates the image file working directory
    mx->viz->plot_network($softmax, save_format => 'png')->render("network.png");

=head1 DESCRIPTION

     Vizualization support for Perl interface to MXNet machine learning library

=head1 Class methods

=head2 print_summary

    convert symbol for detail information

    Parameters
    ----------
    symbol: AI::MXNet::Symbol
        symbol to be visualized
    shape: hashref
        hashref of shapes, str->shape (arrayref[int]), given input shapes
    line_length: int
        total length of printed lines
    positions: arrayref[float]
        relative or absolute positions of log elements in each line
    Returns
    ------
        nothing
=cut

method print_summary(
    AI::MXNet::Symbol        $symbol,
    Maybe[HashRef[Shape]]    $shape=,
    Int                      $line_length=120,
    ArrayRef[Num]            $positions=[.44, .64, .74, 1]
)
{
    my $show_shape;
    my %shape_dict;
    if(defined $shape)
    {
        $show_shape = 1;
        my $interals = $symbol->get_internals;
        my (undef, $out_shapes, undef) = $interals->infer_shape(%{ $shape });
        Carp::confess("Input shape is incomplete")
            unless defined $out_shapes;
        @shape_dict{ @{ $interals->list_outputs } } = @{ $out_shapes };
    }
    my $conf = decode_json($symbol->tojson);
    my $nodes = $conf->{nodes};
    my %heads = map { $_ => 1 } @{ $conf->{heads}[0] };
    if($positions->[-1] <= 1)
    {
        $positions = [map { int($line_length * $_) } @{ $positions }];
    }
    # header names for the different log elements
    my $to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Previous Layer'];
    my $print_row = sub { my ($fields, $positions) = @_;
        my $line = '';
        enumerate(sub {
            my ($i, $field) = @_;
            $line .= $field//'';
            $line = substr($line, 0, $positions->[$i]);
            $line .= ' ' x ($positions->[$i] - length($line));

        }, $fields);
        print $line,"\n";
    };
    print('_' x $line_length,"\n");
    $print_row->($to_display, $positions);
    print('=' x $line_length,"\n");
    my $print_layer_summary = sub { my ($node, $out_shape) = @_;
        my $op = $node->{op};
        my $pre_node = [];
        my $pre_filter = 0;
        if($op ne 'null')
        {
            my $inputs = $node->{inputs};
            for my $item (@{ $inputs })
            {
                my $input_node = $nodes->[$item->[0]];
                my $input_name = $input_node->{name};
                if($input_node->{op} ne 'null' or exists $heads{ $item->[0] })
                {
                    push @{ $pre_node }, $input_name;
                    if($show_shape)
                    {
                        my $key = $input_name;
                        $key .= '_output' if $input_node->{op} ne 'null';
                        if(exists $shape_dict{ $key })
                        {
                            $pre_filter = $pre_filter + int($shape_dict{$key}[1]//0);
                        }
                    }
                }
            }
        }
        my $cur_param = 0;
        if($op eq 'Convolution')
        {
            my $num_filter = $node->{attrs}{num_filter};
            $cur_param = $pre_filter * $num_filter;
            while($node->{attrs}{kernel} =~ /(\d+)/g)
            {
                $cur_param *= $1;
            }
            $cur_param += $num_filter;
        }
        elsif($op eq 'FullyConnected')
        {
            $cur_param = $pre_filter * ($node->{attrs}{num_hidden} + 1);
        }
        elsif($op eq 'BatchNorm')
        {
            my $key = "$node->{name}_output";
            if($show_shape)
            {
                my $num_filter = $shape_dict{$key}[1];
                $cur_param = $num_filter * 2;
            }
        }
        my $first_connection;
        if(not $pre_node)
        {
            $first_connection = '';
        }
        else
        {
            $first_connection = $pre_node->[0];
        }
        my $fields = [
            $node->{name} . '(' . $op . ')',
            join('x', @{ $out_shape }),
            $cur_param,
            $first_connection
        ];
        $print_row->($fields, $positions);
        if(@{ $pre_node } > 1)
        {
            for my $i (1..@{ $pre_node }-1)
            {
                $fields = ['', '', '', $pre_node->[$i]];
                $print_row->($fields, $positions);
            }
        }
        return $cur_param;
    };
    my $total_params = 0;
    enumerate(sub {
        my ($i, $node) = @_;
        my $out_shape = [];
        my $op = $node->{op};
        return if($op eq 'null' and $i > 0);
        if($op ne 'null' or exists $heads{$i})
        {
            if($show_shape)
            {
                my $key = $node->{name};
                $key .= '_output' if $op ne 'null';
                if(exists $shape_dict{ $key })
                {
                    my $end = @{ $shape_dict{ $key } };
                    @{ $out_shape } = @{ $shape_dict{ $key } }[1..$end-1];
                }
            }
        }
        $total_params += $print_layer_summary->($nodes->[$i], $out_shape);
        if($i == @{ $nodes } - 1)
        {
            print('=' x $line_length, "\n");
        }
        else
        {
            print('_' x $line_length, "\n");
        }
    }, $nodes);
    print("Total params: $total_params\n");
    print('_' x $line_length, "\n");
}

=head2 plot_network

    convert symbol to dot object for visualization

    Parameters
    ----------
    title: str
        title of the dot graph
    symbol: AI::MXNet::Symbol
        symbol to be visualized
    shape: HashRef[Shape]
        If supplied, the visualization will include the shape
        of each tensor on the edges between nodes.
    node_attrs: HashRef of node's attributes
        for example:
            {shape => "oval",fixedsize => "false"}
            means to plot the network in "oval"
    hide_weights: Bool
        if True (default) then inputs with names like `*_weight`
        or `*_bias` will be hidden

    Returns
    ------
    dot: Diagraph
        dot object of symbol
=cut


method plot_network(
    AI::MXNet::Symbol       $symbol,
    Str                    :$title='plot',
    Str                    :$save_format='ps',
    Maybe[HashRef[Shape]]  :$shape=,
    HashRef[Str]           :$node_attrs={},
    Bool                   :$hide_weights=1
)
{
    eval { require GraphViz; };
    Carp::confess("plot_network requires GraphViz module") if $@;
    my $draw_shape;
    my %shape_dict;
    if(defined $shape)
    {
        $draw_shape = 1;
        my $interals = $symbol->get_internals;
        my (undef, $out_shapes, undef) = $interals->infer_shape(%{ $shape });
        Carp::confess("Input shape is incomplete")
            unless defined $out_shapes;
        @shape_dict{ @{ $interals->list_outputs } } = @{ $out_shapes };
    }
    my $conf = decode_json($symbol->tojson);
    my $nodes = $conf->{nodes};
    my %node_attr = (
        qw/ shape box fixedsize true
            width 1.3 height 0.8034 style filled/,
        %{ $node_attrs }
    );
    my $dot = AI::MXNet::Visualization::PythonGraphviz->new(
        graph  => GraphViz->new(name => $title),
        format => $save_format
    );
    # color map
    my @cm = (
        "#8dd3c7", "#fb8072", "#ffffb3", "#bebada", "#80b1d3",
        "#fdb462", "#b3de69", "#fccde5"
    );
    # make nodes
    my %hidden_nodes;
    for my $node (@{ $nodes })
    {
        my $op   = $node->{op};
        my $name = $node->{name};
        # input data
        my %attr = %node_attr;
        my $label = $name;
        if($op eq 'null')
        {
            if($name =~ /(?:_weight|_bias|_beta|_gamma|_moving_var|_moving_mean)$/)
            {
                if($hide_weights)
                {
                    $hidden_nodes{$name} = 1;
                }
                # else we don't render a node, but
                # don't add it to the hidden_nodes set
                # so it gets rendered as an empty oval
                next;
            }
            $attr{shape} = 'ellipse'; # inputs get their own shape
            $label = $name;
            $attr{fillcolor} = $cm[0];
        }
        elsif($op eq 'Convolution')
        {
            my @k = $node->{attrs}{kernel} =~ /(\d+)/g;
            my @stride = ($node->{attrs}{stride}//'') =~ /(\d+)/g;
            $stride[0] //= 1;
            $label = "Convolution\n".join('x',@k).'/'.join('x',@stride).", $node->{attrs}{num_filter}";
            $attr{fillcolor} = $cm[1];
        }
        elsif($op eq 'FullyConnected')
        {
            $label = "FullyConnected\n$node->{attrs}{num_hidden}";
            $attr{fillcolor} = $cm[1];
        }
        elsif($op eq 'BatchNorm')
        {
            $attr{fillcolor} = $cm[3];
        }
        elsif($op eq 'Activation' or $op eq 'LeakyReLU')
        {
            $label = "$op\n$node->{attrs}{act_type}";
            $attr{fillcolor} = $cm[2];
        }
        elsif($op eq 'Pooling')
        {
            my @k = $node->{attrs}{kernel} =~ /(\d+)/g;
            my @stride = ($node->{attrs}{stride}//'') =~ /(\d+)/g;
            $stride[0] //= 1;
            $label = "Pooling\n$node->{attrs}{pool_type}, ".join('x',@k).'/'.join('x',@stride);
            $attr{fillcolor} = $cm[4];
        }
        elsif($op eq 'Concat' or $op eq 'Flatten' or $op eq 'Reshape')
        {
            $attr{fillcolor} = $cm[5];
        }
        elsif($op eq 'Softmax')
        {
            $attr{fillcolor} = $cm[6];
        }
        else
        {
            $attr{fillcolor} = $cm[7];
            if($op eq 'Custom')
            {
                $label = $node->{attrs}{op_type};
            }
        }
        $dot->graph->add_node($name, label => $label, %attr);
    };

    # add edges
    for my $node (@{ $nodes })
    {
        my $op   = $node->{op};
        my $name = $node->{name};
        if($op eq 'null')
        {
            next;
        }
        else
        {
            my $inputs = $node->{inputs};
            for my $item (@{ $inputs })
            {
                my $input_node = $nodes->[$item->[0]];
                my $input_name = $input_node->{name};
                if(not exists $hidden_nodes{ $input_name })
                {
                    my %attr = qw/dir back arrowtail normal/;
                    # add shapes
                    if($draw_shape)
                    {
                        my $key = $input_name;
                        $key   .= '_output' if $input_node->{op} ne 'null';
                        if($input_node->{op} ne 'null' and exists $input_node->{attrs})
                        {
                            if(ref $input_node->{attrs} eq 'HASH' and exists $input_node->{attrs}{num_outputs})
                            {
                                $key .= ($input_node->{attrs}{num_outputs} - 1);
                            }
                        }
                        my $end = @{ $shape_dict{$key} };
                        $attr{label} = join('x', @{ $shape_dict{$key} }[1..$end-1]);
                    }
                    $dot->graph->add_edge($name => $input_name, %attr);
                }
            }
        }
    }
    return $dot;
}

package AI::MXNet::Visualization::PythonGraphviz;
use Mouse;
use AI::MXNet::Types;
has 'format' => (
    is => 'ro',
    isa => enum([qw/debug canon text ps hpgl pcl mif
                    pic gd gd2 gif jpeg png wbmp cmapx
                    imap vdx vrml vtx mp fig svg svgz
                    plain/]
    )
);
has 'graph' => (is => 'ro', isa => 'GraphViz');

method render($output=)
{
    my $method = 'as_' . $self->format;
    return $self->graph->$method($output);
}

1;
