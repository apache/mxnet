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

package AI::MXNet::Gluon::ModelZoo::Vision::SqueezeNet;
use strict;
use warnings;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Types;
extends 'AI::MXNet::Gluon::HybridBlock';

func _make_fire($squeeze_channels, $expand1x1_channels, $expand3x3_channels)
{
    my $out = nn->HybridSequential(prefix=>'');
    $out->add(_make_fire_conv($squeeze_channels, 1));

    my $paths = nn->HybridConcurrent(axis=>1, prefix=>'');
    $paths->add(_make_fire_conv($expand1x1_channels, 1));
    $paths->add(_make_fire_conv($expand3x3_channels, 3, 1));
    $out->add($paths);

    return $out;
}

func _make_fire_conv($channels, $kernel_size, $padding=0)
{
    my $out = nn->HybridSequential(prefix=>'');
    $out->add(nn->Conv2D($channels, $kernel_size, padding=>$padding));
    $out->add(nn->Activation('relu'));
    return $out;
}

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::SqueezeNet - SqueezeNet model from the "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size"
=cut

=head1 DESCRIPTION

    SqueezeNet model from the "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360> paper.
    SqueezeNet 1.1 model from the official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    version : Str
        Version of squeezenet. Options are '1.0', '1.1'.
    classes : Int, default 1000
        Number of classification classes.
=cut

has 'version' => (is => 'ro', isa => enum([qw[1.0 1.1]]), required => 1);
has 'classes' => (is => 'ro', isa => 'Int', default => 1000);
method python_constructor_arguments() { [qw/version classes/] }

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->features(nn->HybridSequential(prefix=>''));
        if($self->version eq '1.0')
        {
            $self->features->add(nn->Conv2D(96, kernel_size=>7, strides=>2));
            $self->features->add(nn->Activation('relu'));
            $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2, ceil_mode=>1));
            $self->features->add(_make_fire(16, 64, 64));
            $self->features->add(_make_fire(16, 64, 64));
            $self->features->add(_make_fire(32, 128, 128));
            $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2, ceil_mode=>1));
            $self->features->add(_make_fire(32, 128, 128));
            $self->features->add(_make_fire(48, 192, 192));
            $self->features->add(_make_fire(48, 192, 192));
            $self->features->add(_make_fire(64, 256, 256));
            $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2, ceil_mode=>1));
            $self->features->add(_make_fire(64, 256, 256));
        }
        else
        {
            $self->features->add(nn->Conv2D(64, kernel_size=>3, strides=>2));
            $self->features->add(nn->Activation('relu'));
            $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2, ceil_mode=>1));
            $self->features->add(_make_fire(16, 64, 64));
            $self->features->add(_make_fire(16, 64, 64));
            $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2, ceil_mode=>1));
            $self->features->add(_make_fire(32, 128, 128));
            $self->features->add(_make_fire(32, 128, 128));
            $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2, ceil_mode=>1));
            $self->features->add(_make_fire(48, 192, 192));
            $self->features->add(_make_fire(48, 192, 192));
            $self->features->add(_make_fire(64, 256, 256));
            $self->features->add(_make_fire(64, 256, 256));
        }
        $self->features->add(nn->Dropout(0.5));

        $self->output(nn->HybridSequential(prefix=>''));
        $self->output->add(nn->Conv2D($self->classes, kernel_size=>1));
        $self->output->add(nn->Activation('relu'));
        $self->output->add(nn->AvgPool2D(13));
        $self->output->add(nn->Flatten());
    });
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    $x = $self->features->($x);
    $x = $self->output->($x);
    return $x;
}


package AI::MXNet::Gluon::ModelZoo::Vision;

=head2 get_squeezenet

    SqueezeNet model from the "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360> paper.
    SqueezeNet 1.1 model from the official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    $version : Str
        Version of squeezenet. Options are '1.0', '1.1'.
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method get_squeezenet(
    Str $version, Bool :$pretrained=0, AI::MXNet::Context :$ctx=AI::MXNet::Context->cpu(),
    Str :$root='~/.mxnet/models', Int :$classes=1000
)
{
    my $net = AI::MXNet::Gluon::ModelZoo::Vision::SqueezeNet->new($version, $classes);
    if($pretrained)
    {
        $net->load_parameters(
            AI::MXNet::Gluon::ModelZoo::ModelStore->get_model_file(
                "squeezenet$version",
                root=>$root
            ),
            ctx=>$ctx
        );
    }
    return $net;
}

=head2 squeezenet1_0

    SqueezeNet 1.0 model from the "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters
    and <0.5MB model size" <https://arxiv.org/abs/1602.07360> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method squeezenet1_0(%kwargs)
{
    return __PACKAGE__->get_squeezenet('1.0', %kwargs);
}

=head2 squeezenet1_1

    SqueezeNet 1.1 model from the official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method squeezenet1_1(%kwargs)
{
    return __PACKAGE__->get_squeezenet('1.1', %kwargs);
}

1;
