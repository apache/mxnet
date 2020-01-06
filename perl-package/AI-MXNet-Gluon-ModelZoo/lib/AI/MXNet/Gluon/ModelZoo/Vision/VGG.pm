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

use strict;
use warnings;
use AI::MXNet::Function::Parameters;
package AI::MXNet::Gluon::ModelZoo::Vision::VGG;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
use AI::MXNet::Base;

=head1 NAME 

    AI::MXNet::Gluon::ModelZoo::Vision::VGG - VGG model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
=cut

=head1 DESCRIPTION

    VGG model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    layers : array ref of Int
        Numbers of layers in each feature block.
    filters : array ref of Int
        Numbers of filters in each feature block. List length should match the layers.
    classes : Int, default 1000
        Number of classification classes.
    batch_norm : Bool, default 0
        Use batch normalization.
=cut
method python_constructor_arguments() { [qw/layers filters classes batch_norm/] }
has ['layers',
     'filters']   => (is => 'ro', isa => 'ArrayRef[Int]', required => 1);
has  'classes'    => (is => 'ro', isa => 'Int', default => 1000);
has  'batch_norm' => (is => 'ro', isa => 'Bool', default => 0);

sub BUILD
{
    my $self = shift;
    assert(@{ $self->layers } == @{ $self->filters });
    $self->name_scope(sub {
        $self->features($self->_make_features());
        $self->features->add(nn->Dense(4096, activation=>'relu',
                                       weight_initializer=>'normal',
                                       bias_initializer=>'zeros'));
        $self->features->add(nn->Dropout(rate=>0.5));
        $self->features->add(nn->Dense(4096, activation=>'relu',
                                       weight_initializer=>'normal',
                                       bias_initializer=>'zeros'));
        $self->features->add(nn->Dropout(rate=>0.5));
        $self->output(nn->Dense($self->classes,
                                   weight_initializer=>'normal',
                                   bias_initializer=>'zeros'));
    });
}

method _make_features()
{
    my $featurizer = nn->HybridSequential(prefix=>'');
    for(enumerate($self->layers))
    {
        my ($i, $num) = @$_;
        for(0..$num-1)
        {
            $featurizer->add(
                nn->Conv2D(
                    $self->filters->[$i], kernel_size => 3, padding => 1,
                    weight_initializer => mx->init->Xavier(
                        rnd_type    => 'gaussian',
                        factor_type => 'out',
                        magnitude   => 2
                    ),
                    bias_initializer=>'zeros'
                )
            );
            if($self->batch_norm)
            {
                $featurizer->add(nn->BatchNorm());
            }
            $featurizer->add(nn->Activation('relu'));
        }
        $featurizer->add(nn->MaxPool2D(strides=>2));
    }
    return $featurizer;
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    $x = $self->features->($x);
    $x = $self->output->($x);
    return $x;
}

package AI::MXNet::Gluon::ModelZoo::Vision;

# Specification
my %vgg_spec = (
    11 => [[1, 1, 2, 2, 2], [64, 128, 256, 512, 512]],
    13 => [[2, 2, 2, 2, 2], [64, 128, 256, 512, 512]],
    16 => [[2, 2, 3, 3, 3], [64, 128, 256, 512, 512]],
    19 => [[2, 2, 4, 4, 4], [64, 128, 256, 512, 512]]
);

=head2 get_vgg

    VGG model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    $num_layers : Int
        Number of layers for the variant of densenet. Options are 11, 13, 16, 19.
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method get_vgg(
    Int $num_layers, Bool :$pretrained=0, AI::MXNet::Context :$ctx=AI::MXNet::Context->cpu(),
    Str :$root='~/.mxnet/models', Int :$classes=1000, Bool :$batch_norm=0
)
{
    my ($layers, $filters) = @{ $vgg_spec{$num_layers} };
    my $net = AI::MXNet::Gluon::ModelZoo::Vision::VGG->new($layers, $filters, $classes, $batch_norm);
    if($pretrained)
    {
        $net->load_parameters(
            AI::MXNet::Gluon::ModelZoo::ModelStore->get_model_file(
                "vgg$num_layers".($batch_norm ? '_bn' : ''),
                root=>$root
            ),
            ctx=>$ctx
        );
    }
    return $net;
}

=head2 vgg11

    VGG-11 model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method vgg11(%kwargs)
{
    return __PACKAGE__->get_vgg(11, %kwargs);
}

=head2 vgg13

    VGG-13 model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method vgg13(%kwargs)
{
    return __PACKAGE__->get_vgg(13, %kwargs);
}

=head2 vgg16

    VGG-16 model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method vgg16(%kwargs)
{
    return __PACKAGE__->get_vgg(16, %kwargs);
}

=head2 vgg19

    VGG-19 model from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method vgg19(%kwargs)
{
    return __PACKAGE__->get_vgg(19, %kwargs);
}

=head2 vgg11_bn

    VGG-11 model with batch normalization from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method vgg11_bn(%kwargs)
{
    $kwargs{batch_norm} = 1;
    return __PACKAGE__->get_vgg(11, %kwargs);
}

=head2 vgg13_bn

    VGG-13 model with batch normalization from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method vgg13_bn(%kwargs)
{
    $kwargs{batch_norm} = 1;
    return __PACKAGE__->get_vgg(13, %kwargs);
}

=head2 vgg16_bn

    VGG-16 model with batch normalization from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method vgg16_bn(%kwargs)
{
    $kwargs{batch_norm} = 1;
    return __PACKAGE__->get_vgg(16, %kwargs);
}

=head2 vgg19_bn

    VGG-19 model with batch normalization from the "Very Deep Convolutional Networks for Large-Scale Image Recognition"
    <https://arxiv.org/abs/1409.1556> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default AI::MXNet::Context->cpu
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method vgg19_bn(%kwargs)
{
    $kwargs{batch_norm} = 1;
    return __PACKAGE__->get_vgg(19, %kwargs);
}

1;