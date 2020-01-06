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
package AI::MXNet::Gluon::ModelZoo::Vision::MobileNet::RELU6;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    return $F->clip($x, a_min => 0, a_max => 6, name=>"relu6");
}

package AI::MXNet::Gluon::ModelZoo::Vision::MobileNet::LinearBottleneck;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
has [qw/in_channels channels t stride/] => (is => 'ro', isa => 'Int', required => 1);
method python_constructor_arguments(){ [qw/in_channels channels t stride/] }

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::MobileNet::LinearBottleneck - LinearBottleneck used in MobileNetV2 model
=cut

=head1 DESCRIPTION

    LinearBottleneck used in MobileNetV2 model from the
    "Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381> paper.

    Parameters
    ----------
    in_channels : Int
        Number of input channels.
    channels : Int
        Number of output channels.
    t : Int
        Layer expansion ratio.
    stride : Int
        stride
=cut

func _add_conv(
    $out, $channels, :$kernel=1, :$stride=1, :$pad=0,
    :$num_group=1, :$active=1, :$relu6=0
)
{
    $out->add(nn->Conv2D($channels, $kernel, $stride, $pad, groups=>$num_group, use_bias=>0));
    $out->add(nn->BatchNorm(scale=>1));
    if($active)
    {
        $out->add($relu6 ? AI::MXNet::Gluon::ModelZoo::Vision::MobileNet::RELU6->new : nn->Activation('relu'));
    }
}

sub BUILD
{
    my $self = shift;
    $self->use_shortcut($self->stride == 1 and $self->in_channels == $self->channels);
    $self->name_scope(sub {
        $self->out(nn->HybridSequential());
        _add_conv($self->out, $self->in_channels * $self->t, relu6=>1);
        _add_conv(
            $self->out, $self->in_channels * $self->t, kernel=>3, stride=>$self->stride,
            pad=>1, num_group=>$self->in_channels * $self->t, relu6=>1
        );
        _add_conv($self->out, $self->channels, active=>0, relu6=>1);
    });
}

method hybrid_forward($F, $x)
{
    my $out = $self->out->($x);
    if($self->use_shortcut)
    {
        $out = $F->elemwise_add($out, $x);
    }
    return $out;
}

package AI::MXNet::Gluon::ModelZoo::Vision::MobileNet;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'multiplier' => (is => 'ro', isa => 'Num', default => 1);
has 'classes'    => (is => 'ro', isa => 'Int', default => 1000);
method python_constructor_arguments(){ [qw/multiplier classes/] }

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::MobileNet - MobileNet model from the
        "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
=cut

=head1 DESCRIPTION

    MobileNet model from the
    "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861> paper.

    Parameters
    ----------
    multiplier : Num, default 1.0
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : Int, default 1000
        Number of classes for the output layer.
=cut

func _add_conv(
    $out, :$channels=1, :$kernel=1, :$stride=1, :$pad=0,
    :$num_group=1, :$active=1, :$relu6=0
)
{
    $out->add(nn->Conv2D($channels, $kernel, $stride, $pad, groups=>$num_group, use_bias=>0));
    $out->add(nn->BatchNorm(scale=>1));
    if($active)
    {
        $out->add($relu6 ? AI::MXNet::Gluon::ModelZoo::Vision::MobileNet::RELU6->new : nn->Activation('relu'));
    }
}


func _add_conv_dw($out, :$dw_channels=, :$channels=, :$stride=, :$relu6=0)
{
    _add_conv($out, channels=>$dw_channels, kernel=>3, stride=>$stride,
              pad=>1, num_group=>$dw_channels, relu6=>$relu6);
    _add_conv($out, channels=>$channels, relu6=>$relu6);
}

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->features(nn->HybridSequential(prefix=>''));
        $self->features->name_scope(sub {
            _add_conv($self->features, channels=>int(32 * $self->multiplier), kernel=>3, pad=>1, stride=>2);
            my $dw_channels = [map { int($_ * $self->multiplier) } (32, 64, (128)x2, (256)x2, (512)x6, 1024)];
            my $channels = [map { int($_ * $self->multiplier) } (64, (128)x2, (256)x2, (512)x6, (1024)x2)];
            my $strides = [(1, 2)x3, (1)x5, 2, 1];
            for(zip($dw_channels, $channels, $strides))
            {
                my ($dwc, $c, $s) = @$_;
                _add_conv_dw($self->features, dw_channels=>$dwc, channels=>$c, stride=>$s);
            }
            $self->features->add(nn->GlobalAvgPool2D());
            $self->features->add(nn->Flatten());
        });
        $self->output(nn->Dense($self->classes));
    });
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    $x = $self->features->($x);
    $x = $self->output->($x);
    return $x;
}

package AI::MXNet::Gluon::ModelZoo::Vision::MobileNetV2;
use AI::MXNet::Gluon::Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::Gluon::HybridBlock';
has 'multiplier' => (is => 'ro', isa => 'Num', default => 1);
has 'classes'    => (is => 'ro', isa => 'Int', default => 1000);
method python_constructor_arguments(){ [qw/multiplier classes/] }

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::MobileNetV2 - MobileNet model from the
        "Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation"
=cut

=head1 DESCRIPTION

    MobileNetV2 model from the
    "Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381> paper.

    Parameters
    ----------
    multiplier : Num, default 1.0
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    classes : Int, default 1000
        Number of classes for the output layer.
=cut

func _add_conv(
    $out, $channels, :$kernel=1, :$stride=1, :$pad=0,
    :$num_group=1, :$active=1, :$relu6=0
)
{
    $out->add(nn->Conv2D($channels, $kernel, $stride, $pad, groups=>$num_group, use_bias=>0));
    $out->add(nn->BatchNorm(scale=>1));
    if($active)
    {
        $out->add($relu6 ? AI::MXNet::Gluon::ModelZoo::Vision::MobileNet::RELU6->new : nn->Activation('relu'));
    }
}

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->features(nn->HybridSequential(prefix=>'features_'));
        $self->features->name_scope(sub {
            _add_conv(
                $self->features, int(32 * $self->multiplier), kernel=>3,
                stride=>2, pad=>1, relu6=>1
            );

            my $in_channels_group = [map { int($_ * $self->multiplier) } (32, 16, (24)x2, (32)x3, (64)x4, (96)x3, (160)x3)];
            my $channels_group = [map { int($_ * $self->multiplier) } (16, (24)x2, (32)x3, (64)x4, (96)x3, (160)x3, 320)];
            my $ts = [1, (6)x16];
            my $strides = [(1, 2)x2, 1, 1, 2, (1)x6, 2, (1)x3];

            for(zip($in_channels_group, $channels_group, $ts, $strides))
            {
                my ($in_c, $c, $t, $s) = @$_;
                $self->features->add(
                    AI::MXNet::Gluon::ModelZoo::Vision::MobileNet::LinearBottleneck->new(
                        in_channels=>$in_c, channels=>$c,
                        t=>$t, stride=>$s
                    )
                );
            }

            my $last_channels = $self->multiplier > 1 ? int(1280 * $self->multiplier) : 1280;
            _add_conv($self->features, $last_channels, relu6=>1);
            $self->features->add(nn->GlobalAvgPool2D());
        });

        $self->output(nn->HybridSequential(prefix=>'output_'));
        $self->output->name_scope(sub {
            $self->output->add(
                nn->Conv2D($self->classes, 1, use_bias=>0, prefix=>'pred_'),
                nn->Flatten()
            );
        });
    });
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    $x = $self->features->($x);
    $x = $self->output->($x);
    return $x;
}

package AI::MXNet::Gluon::ModelZoo::Vision;

=head2 get_mobilenet

    MobileNet model from the
    "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861> paper.

    Parameters
    ----------
    $multiplier : Num
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method get_mobilenet(
    Num $multiplier, Bool :$pretrained=0, AI::MXNet::Context :$ctx=AI::MXNet::Context->cpu(),
    Str :$root='~/.mxnet/models'
)
{
    my $net = AI::MXNet::Gluon::ModelZoo::Vision::MobileNet->new($multiplier);
    if($pretrained)
    {
        my $version_suffix = sprintf("%.2f", $multiplier);
        if($version_suffix eq '1.00' or $version_suffix eq '0.50')
        {
            $version_suffix =~ s/.$//;
        }
        $net->load_parameters(
            AI::MXNet::Gluon::ModelZoo::ModelStore->get_model_file(
                "mobilenet$version_suffix",
                root=>$root
            ),
            ctx=>$ctx
        );
    }
    return $net;
}

=head2 get_mobilenet_v2

    MobileNetV2 model from the
    "Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381> paper.

    Parameters
    ----------
    $multiplier : Num
        The width multiplier for controling the model size. Only multipliers that are no
        less than 0.25 are supported. The actual number of channels is equal to the original
        channel size multiplied by this multiplier.
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method get_mobilenet_v2(
    Num $multiplier, Bool :$pretrained=0, AI::MXNet::Context :$ctx=AI::MXNet::Context->cpu(),
    Str :$root='~/.mxnet/models'
)
{
    my $net = AI::MXNet::Gluon::ModelZoo::Vision::MobileNetV2->new($multiplier);
    if($pretrained)
    {
        my $version_suffix = sprintf("%.2f", $multiplier);
        if($version_suffix eq '1.00' or $version_suffix eq '0.50')
        {
            $version_suffix =~ s/.$//;
        }
        $net->load_parameters(
            AI::MXNet::Gluon::ModelZoo::ModelStore->get_model_file(
                "mobilenetv2_$version_suffix",
                root=>$root
            ),
            ctx=>$ctx
        );
    }
    return $net;
}

=head2 mobilenet1_0

    MobileNet model from the
    "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861> paper, with width multiplier 1.0.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
=cut

method mobilenet1_0(%kwargs)
{
    return __PACKAGE__->get_mobilenet(1.0, %kwargs);
}

=head2 mobilenet_v2_1_0

    MobileNetV2 model from the
    "Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
=cut

method mobilenet_v2_1_0(%kwargs)
{
    return __PACKAGE__->get_mobilenet_v2(1.0, %kwargs);
}

=head2 mobilenet0_75

    MobileNet model from the
    "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861> paper, with width multiplier 0.75.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
=cut

method mobilenet0_75(%kwargs)
{
    return __PACKAGE__->get_mobilenet(0.75, %kwargs);
}

=head2 mobilenet_v2_0_75

    MobileNetV2 model from the
    "Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
=cut

method mobilenet_v2_0_75(%kwargs)
{
    return __PACKAGE__->get_mobilenet_v2(0.75, %kwargs);
}

=head2 mobilenet0_5

    MobileNet model from the
    "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861> paper, with width multiplier 0.5.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
=cut

method mobilenet0_5(%kwargs)
{
    return __PACKAGE__->get_mobilenet(0.5, %kwargs);
}

=head2 mobilenet_v2_0_5

    MobileNetV2 model from the
    "Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
=cut

method mobilenet_v2_0_5(%kwargs)
{
    return __PACKAGE__->get_mobilenet_v2(0.5, %kwargs);
}

=head2 mobilenet0_25

    MobileNet model from the
    "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
    <https://arxiv.org/abs/1704.04861> paper, with width multiplier 0.25.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
=cut

method mobilenet0_25(%kwargs)
{
    return __PACKAGE__->get_mobilenet(0.25, %kwargs);
}

=head2 mobilenet_v2_0_25

    MobileNetV2 model from the
    "Inverted Residuals and Linear Bottlenecks:
      Mobile Networks for Classification, Detection and Segmentation"
    <https://arxiv.org/abs/1801.04381> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
=cut

method mobilenet_v2_0_25(%kwargs)
{
    return __PACKAGE__->get_mobilenet_v2(0.25, %kwargs);
}

1;
