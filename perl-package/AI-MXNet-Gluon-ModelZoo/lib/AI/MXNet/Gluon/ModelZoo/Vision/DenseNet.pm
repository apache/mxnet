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

package AI::MXNet::Gluon::ModelZoo::Vision::DenseNet;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

func _make_dense_block($num_layers, $bn_size, $growth_rate, $dropout, $stage_index)
{
    my $out = nn->HybridSequential(prefix=>"stage${stage_index}_");
    $out->name_scope(sub {
        for(1..$num_layers)
        {
            $out->add(_make_dense_layer($growth_rate, $bn_size, $dropout));
        }
    });
    return $out;
}

func _make_dense_layer($growth_rate, $bn_size, $dropout)
{
    my $new_features = nn->HybridSequential(prefix=>'');
    $new_features->add(nn->BatchNorm());
    $new_features->add(nn->Activation('relu'));
    $new_features->add(nn->Conv2D($bn_size * $growth_rate, kernel_size=>1, use_bias=>0));
    $new_features->add(nn->BatchNorm());
    $new_features->add(nn->Activation('relu'));
    $new_features->add(nn->Conv2D($growth_rate, kernel_size=>3, padding=>1, use_bias=>0));
    if($dropout)
    {
        $new_features->add(nn->Dropout($dropout));
    }

    my $out = nn->HybridConcurrent(axis=>1, prefix=>'');
    $out->add(nn->Identity());
    $out->add($new_features);

    return $out;
}

func _make_transition($num_output_features)
{
    my $out = nn->HybridSequential(prefix=>'');
    $out->add(nn->BatchNorm());
    $out->add(nn->Activation('relu'));
    $out->add(nn->Conv2D($num_output_features, kernel_size=>1, use_bias=>0));
    $out->add(nn->AvgPool2D(pool_size=>2, strides=>2));
    return $out;
}

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::DenseNet - Densenet-BC model from the "Densely Connected Convolutional Networks"
=cut

=head1 DESCRIPTION

    Densenet-BC model from the "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> paper.

    Parameters
    ----------
    num_init_features : Int
        Number of filters to learn in the first convolution layer.
    growth_rate : Int
        Number of filters to add each layer (`k` in the paper).
    block_config : array ref of Int
        List of integers for numbers of layers in each pooling block.
    bn_size : Int, default 4
        Multiplicative factor for number of bottle neck layers.
        (i.e. bn_size * k features in the bottleneck layer)
    dropout : float, default 0
        Rate of dropout after each dense layer.
    classes : int, default 1000
        Number of classification classes.
=cut
has [qw/num_init_features
        growth_rate/] => (is => 'ro', isa => 'Int', required => 1);
has 'block_config'    => (is => 'ro', isa => 'ArrayRef[Int]', required => 1);
has 'bn_size'         => (is => 'ro', isa => 'Int', default => 4);
has 'dropout'         => (is => 'ro', isa => 'Num', default => 0);
has 'classes'         => (is => 'ro', isa => 'Int', default => 1000);
method python_constructor_arguments(){ [qw/num_init_features growth_rate block_config bn_size dropout classes/] }

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->features(nn->HybridSequential(prefix=>''));
        $self->features->add(
            nn->Conv2D(
                $self->num_init_features, kernel_size=>7,
                strides=>2, padding=>3, use_bias=>0
            )
        );
        $self->features->add(nn->BatchNorm());
        $self->features->add(nn->Activation('relu'));
        $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2, padding=>1));
        # Add dense blocks
        my $num_features = $self->num_init_features;
        for(enumerate($self->block_config))
        {
            my ($i, $num_layers) = @$_;
            $self->features->add(_make_dense_block($num_layers, $self->bn_size, $self->growth_rate, $self->dropout, $i+1));
            $num_features += $num_layers * $self->growth_rate;
            if($i != @{ $self->block_config } - 1)
            {
                $self->features->add(_make_transition(int($num_features/2)));
                $num_features = int($num_features/2);
            }
        }
        $self->features->add(nn->BatchNorm());
        $self->features->add(nn->Activation('relu'));
        $self->features->add(nn->AvgPool2D(pool_size=>7));
        $self->features->add(nn->Flatten());

        $self->output(nn->Dense($self->classes));
    });
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    $x = $self->features->($x);
    $x = $self->output->($x);
    return $x;
}

package AI::MXNet::Gluon::ModelZoo::Vision;

my %densenet_spec = (
    121 => [64, 32, [6, 12, 24, 16]],
    161 => [96, 48, [6, 12, 36, 24]],
    169 => [64, 32, [6, 12, 32, 32]],
    201 => [64, 32, [6, 12, 48, 32]]
);

=head2 get_densenet

    Densenet-BC model from the
    "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> paper.

    Parameters
    ----------
    $num_layers : Int
        Number of layers for the variant of densenet. Options are 121, 161, 169, 201.
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method get_densenet(
    Int $num_layers, Bool :$pretrained=0, :$ctx=AI::MXNet::Context->cpu(),
    :$root='~/.mxnet/models',
    Int :$bn_size=4,
    Num :$dropout=0,
    Int :$classes=1000
)
{
    my ($num_init_features, $growth_rate, $block_config) = @{ $densenet_spec{$num_layers} };
    my $net = AI::MXNet::Gluon::ModelZoo::Vision::DenseNet->new(
        $num_init_features, $growth_rate, $block_config,
        $bn_size, $dropout, $classes
    );
    if($pretrained)
    {
        $net->load_parameters(
            AI::MXNet::Gluon::ModelZoo::ModelStore->get_model_file(
                "densenet$num_layers",
                root=>$root
            ),
            ctx=>$ctx
        );
    }
    return $net;
}

=head2 densenet121

    Densenet-BC 121-layer model from the
    "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method densenet121(%kwargs)
{
    return __PACKAGE__->get_densenet(121, %kwargs)
}

=head2 densenet161

    Densenet-BC 161-layer model from the
    "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method densenet161(%kwargs)
{
    return __PACKAGE__->get_densenet(161, %kwargs)
}

=head2 densenet169

    Densenet-BC 169-layer model from the
    "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method densenet169(%kwargs)
{
    return __PACKAGE__->get_densenet(169, %kwargs)
}

=head2 densenet201

    Densenet-BC 201-layer model from the
    "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method densenet201(%kwargs)
{
    return __PACKAGE__->get_densenet(201, %kwargs)
}

1;
