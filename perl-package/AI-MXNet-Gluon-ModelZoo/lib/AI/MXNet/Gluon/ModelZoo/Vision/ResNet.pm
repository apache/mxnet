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

package AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BasicBlockV1;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME 

    AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BasicBlockV1 - BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
=cut

=head1 DESCRIPTION

    BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 18, 34 layers.

    Parameters
    ----------
    channels : Int
        Number of output channels.
    stride : Int
        Stride size.
    downsample : Bool, default 0
        Whether to downsample the input.
    in_channels : Int, default 0
        Number of input channels. Default is 0, to infer from the graph.
=cut

has ['channels',
      'stride']   => (is => 'ro', isa => 'Int', required => 1);
has 'downsample' => (is => 'rw', default => 0);
has 'in_channels' => (is => 'ro', isa => 'Int', default => 0);
method python_constructor_arguments() { [qw/channels stride downsample/] }
func _conv3x3($channels, $stride, $in_channels)
{
    return nn->Conv2D(
        $channels, kernel_size=>3, strides=>$stride, padding=>1,
        use_bias=>0, in_channels=>$in_channels
    );
}

sub BUILD
{
    my $self = shift;
    $self->body(nn->HybridSequential(prefix=>''));
    $self->body->add(_conv3x3($self->channels, $self->stride, $self->in_channels));
    $self->body->add(nn->BatchNorm());
    $self->body->add(nn->Activation('relu'));
    $self->body->add(_conv3x3($self->channels, 1, $self->channels));
    $self->body->add(nn->BatchNorm());
    if($self->downsample)
    {
        $self->downsample(nn->HybridSequential(prefix=>''));
        $self->downsample->add(
            nn->Conv2D($self->channels, kernel_size=>1, strides=>$self->stride,
                       use_bias=>0, in_channels=>$self->in_channels)
        );
        $self->downsample->add(nn->BatchNorm());
    }
    else
    {
        $self->downsample(undef);
    }
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    my $residual = $x;
    $x = $self->body->($x);
    if(defined $self->downsample)
    {
        $residual = $self->downsample->($residual);
    }
    $x = $F->Activation($residual+$x, act_type=>'relu');
    return $x;
}

package AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BottleneckV1;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BottleneckV1 - Bottleneck V1 from "Deep Residual Learning for Image Recognition"
=cut

=head1 DESCRIPTION

    Bottleneck V1 from "Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385> paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
=cut

has ['channels',
      'stride']   => (is => 'ro', isa => 'Int', required => 1);
has 'downsample'  => (is => 'rw', default => 0);
has 'in_channels' => (is => 'ro', isa => 'Int', default => 0);
method python_constructor_arguments() { [qw/channels stride downsample/] }
func _conv3x3($channels, $stride, $in_channels)
{
    return nn->Conv2D(
        $channels, kernel_size=>3, strides=>$stride, padding=>1,
        use_bias=>0, in_channels=>$in_channels
    );
}

sub BUILD
{
    my $self = shift;
    $self->body(nn->HybridSequential(prefix=>''));
    $self->body->add(nn->Conv2D(int($self->channels/4), kernel_size=>1, strides=>$self->stride));
    $self->body->add(nn->BatchNorm());
    $self->body->add(nn->Activation('relu'));
    $self->body->add(_conv3x3(int($self->channels/4), 1, int($self->channels/4)));
    $self->body->add(nn->BatchNorm());
    $self->body->add(nn->Activation('relu'));
    $self->body->add(nn->Conv2D($self->channels, kernel_size=>1, strides=>1));
    $self->body->add(nn->BatchNorm());
    if($self->downsample)
    {
        $self->downsample(nn->HybridSequential(prefix=>''));
        $self->downsample->add(
            nn->Conv2D($self->channels, kernel_size=>1, strides=>$self->stride,
                       use_bias=>0, in_channels=>$self->in_channels)
        );
        $self->downsample->add(nn->BatchNorm());
    }
    else
    {
        $self->downsample(undef);
    }
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    my $residual = $x;
    $x = $self->body->($x);
    if(defined $self->downsample)
    {
        $residual = $self->downsample->($residual);
    }
    $x = $F->Activation($residual+$x, act_type=>'relu');
    return $x;
}

package AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BasicBlockV2;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME 

    AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BasicBlockV2 - BasicBlock V2 from "Identity Mappings in Deep Residual Networks"
=cut

=head1 DESCRIPTION

    Bottleneck V2 from "Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027> paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : Int
        Number of output channels.
    stride : Int
        Stride size.
    downsample : Bool, default 0
        Whether to downsample the input.
    in_channels : Int, default 0
        Number of input channels. Default is 0, to infer from the graph.
=cut

has ['channels',
      'stride']   => (is => 'ro', isa => 'Int', required => 1);
has 'downsample' => (is => 'rw', default => 0);
has 'in_channels' => (is => 'ro', isa => 'Int', default => 0);
method python_constructor_arguments() { [qw/channels stride downsample/] }
func _conv3x3($channels, $stride, $in_channels)
{
    return nn->Conv2D(
        $channels, kernel_size=>3, strides=>$stride, padding=>1,
        use_bias=>0, in_channels=>$in_channels
    );
}

sub BUILD
{
    my $self = shift;
    $self->bn1(nn->BatchNorm());
    $self->conv1(_conv3x3($self->channels, $self->stride, $self->in_channels));
    $self->bn2(nn->BatchNorm());
    $self->conv2(_conv3x3($self->channels, 1, $self->channels));
    if($self->downsample)
    {
        $self->downsample(
            nn->Conv2D($self->channels, kernel_size=>1, strides=>$self->stride,
                       use_bias=>0, in_channels=>$self->in_channels)
        );
    }
    else
    {
        $self->downsample(undef);
    }
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    my $residual = $x;
    $x = $self->bn1->($x);
    $x = $F->Activation($x, act_type=>'relu');
    if(defined $self->downsample)
    {
        $residual = $self->downsample->($x);
    }
    $x = $self->conv1->($x);

    $x = $self->bn2->($x);
    $x = $F->Activation($x, act_type=>'relu');
    $x = $self->conv2->($x);

    return $x + $residual;
}


package AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BottleneckV2;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BottleneckV2 - Bottleneck V2 from "Identity Mappings in Deep Residual Networks"
=cut

=head1 DESCRIPTION

    Bottleneck V2 from "Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027> paper.
    This is used for ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
=cut

has ['channels',
      'stride']   => (is => 'ro', isa => 'Int', required => 1);
has 'downsample' => (is => 'rw', default => 0);
has 'in_channels' => (is => 'ro', isa => 'Int', default => 0);
method python_constructor_arguments() { [qw/channels stride downsample/] }
func _conv3x3($channels, $stride, $in_channels)
{
    return nn->Conv2D(
        $channels, kernel_size=>3, strides=>$stride, padding=>1,
        use_bias=>0, in_channels=>$in_channels
    );
}

sub BUILD
{
    my $self = shift;
    $self->bn1(nn->BatchNorm());
    $self->conv1(nn->Conv2D(int($self->channels/4), kernel_size=>1, strides=>1, use_bias=>0));
    $self->bn2(nn->BatchNorm());
    $self->conv2(_conv3x3(int($self->channels/4), $self->stride, int($self->channels/4)));
    $self->bn3(nn->BatchNorm());
    $self->conv3(nn->Conv2D($self->channels, kernel_size=>1, strides=>1, use_bias=>0));
    if($self->downsample)
    {
        $self->downsample(
            nn->Conv2D($self->channels, kernel_size=>1, strides=>$self->stride,
                       use_bias=>0, in_channels=>$self->in_channels)
        );
    }
    else
    {
        $self->downsample(undef);
    }
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    my $residual = $x;
    $x = $self->bn1->($x);
    $x = $F->Activation($x, act_type=>'relu');
    if(defined $self->downsample)
    {
        $residual = $self->downsample->($x);
    }
    $x = $self->conv1->($x);

    $x = $self->bn2->($x);
    $x = $F->Activation($x, act_type=>'relu');
    $x = $self->conv2->($x);

    $x = $self->bn3->($x);
    $x = $F->Activation($x, act_type=>'relu');
    $x = $self->conv3->($x);

    return $x + $residual;
}


# Nets
package AI::MXNet::Gluon::ModelZoo::Vision::ResNet::V1;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
use AI::MXNet::Base;

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::ResNet::V1 - ResNet V1 model from "Deep Residual Learning for Image Recognition"
=cut

=head1 DESCRIPTION

    ResNet V1 model from from "Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385> paper.

    Parameters
    ----------
    block : AI::MXNet::Gluon::HybridBlock
        Class for the residual block. Options are AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BasicBlockV1,
        AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BottleneckV1.
    layers : array ref of Int
        Numbers of layers in each block
    channels : array ref of Int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default 0
        Enable thumbnail.
=cut

has 'block'     => (is => 'ro', isa => 'Str', required => 1);
has ['layers',
    'channels'] => (is => 'ro', isa => 'ArrayRef[Int]', required => 1);
has 'classes'   => (is => 'ro', isa => 'Int', default => 1000);
has 'thumbnail' => (is => 'ro', isa => 'Bool', default => 0);
method python_constructor_arguments() { [qw/block layers channels classes thumbnail/] }
func _conv3x3($channels, $stride, $in_channels)
{
    return nn->Conv2D(
        $channels, kernel_size=>3, strides=>$stride, padding=>1,
        use_bias=>0, in_channels=>$in_channels
    );
}

sub BUILD
{
    my $self = shift;
    assert(@{ $self->layers } == (@{ $self->channels } - 1));
    $self->name_scope(sub {
        $self->features(nn->HybridSequential(prefix=>''));
        if($self->thumbnail)
        {
            $self->features->add(_conv3x3($self->channels->[0], 1, 0));
        }
        else
        {
            $self->features->add(nn->Conv2D($self->channels->[0], 7, 2, 3, use_bias=>0));
            $self->features->add(nn->BatchNorm());
            $self->features->add(nn->Activation('relu'));
            $self->features->add(nn->MaxPool2D(3, 2, 1));
        }
        for(enumerate($self->layers))
        {
            my ($i, $num_layer) = @$_;
            my $stride = $i == 0 ? 1 : 2;
            $self->features->add(
                $self->_make_layer(
                    $self->block, $num_layer, $self->channels->[$i+1],
                    $stride, $i+1, in_channels=>$self->channels->[$i]
                )
            );
        }
        $self->features->add(nn->GlobalAvgPool2D());
        $self->output(nn->Dense($self->classes, in_units=>$self->channels->[-1]));
    });
}

method _make_layer($block, $layers, $channels, $stride, $stage_index, :$in_channels=0)
{
    my $layer = nn->HybridSequential(prefix=>"stage${stage_index}_");
    $layer->name_scope(sub {
        $layer->add(
            $block->new(
                $channels, $stride, $channels != $in_channels, in_channels=>$in_channels,
                prefix=>''
            )
        );
        for(1..$layers-1)
        {
            $layer->add($block->new($channels, 1, 0, in_channels=>$channels, prefix=>''));
        }
    });
    return $layer;
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    $x = $self->features->($x);
    $x = $self->output->($x);
    return $x;
}


package AI::MXNet::Gluon::ModelZoo::Vision::ResNet::V2;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';
use AI::MXNet::Base;

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::ResNet::V2 - ResNet V2 model from "Identity Mappings in Deep Residual Networks"
=cut

=head1 DESCRIPTION

    ResNet V2 model from "Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027> paper.

    Parameters
    ----------
    block : AI::MXNet::Gluon::HybridBlock
        Class for the residual block. Options are AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BasicBlockV2,
        AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BottleneckV2.
    layers : array ref of Int
        Numbers of layers in each block
    channels : array ref of Int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default 0
        Enable thumbnail.
=cut

has 'block'     => (is => 'ro', isa => 'Str', required => 1);
has ['layers',
    'channels'] => (is => 'ro', isa => 'ArrayRef[Int]', required => 1);
has 'classes'   => (is => 'ro', isa => 'Int', default => 1000);
has 'thumbnail' => (is => 'ro', isa => 'Bool', default => 0);
method python_constructor_arguments() { [qw/block layers channels classes thumbnail/] }
func _conv3x3($channels, $stride, $in_channels)
{
    return nn->Conv2D(
        $channels, kernel_size=>3, strides=>$stride, padding=>1,
        use_bias=>0, in_channels=>$in_channels
    );
}

sub BUILD
{
    my $self = shift;
    assert(@{ $self->layers } == (@{ $self->channels } - 1));
    $self->name_scope(sub {
        $self->features(nn->HybridSequential(prefix=>''));
        $self->features->add(nn->BatchNorm(scale=>0, center=>0));
        if($self->thumbnail)
        {
            $self->features->add(_conv3x3($self->channels->[0], 1, 0));
        }
        else
        {
            $self->features->add(nn->Conv2D($self->channels->[0], 7, 2, 3, use_bias=>0));
            $self->features->add(nn->BatchNorm());
            $self->features->add(nn->Activation('relu'));
            $self->features->add(nn->MaxPool2D(3, 2, 1));
        }
        my $in_channels = $self->channels->[0];
        for(enumerate($self->layers))
        {
            my ($i, $num_layer) = @$_;
            my $stride = $i == 0 ? 1 : 2;
            $self->features->add(
                $self->_make_layer(
                    $self->block, $num_layer, $self->channels->[$i+1],
                    $stride, $i+1, in_channels=>$in_channels
                )
            );
            $in_channels = $self->channels->[$i+1];
        }
        $self->features->add(nn->BatchNorm());
        $self->features->add(nn->Activation('relu'));
        $self->features->add(nn->GlobalAvgPool2D());
        $self->features->add(nn->Flatten());
        $self->output(nn->Dense($self->classes, in_units=>$in_channels));
    });
}

method _make_layer($block, $layers, $channels, $stride, $stage_index, :$in_channels=0)
{
    my $layer = nn->HybridSequential(prefix=>"stage${stage_index}_");
    $layer->name_scope(sub {
        $layer->add(
            $block->new(
                $channels, $stride, $channels != $in_channels, in_channels=>$in_channels,
                prefix=>''
            )
        );
        for(1..$layers-1)
        {
            $layer->add($block->new($channels, 1, 0, in_channels=>$channels, prefix=>''));
        }
    });
    return $layer;
}

method hybrid_forward(GluonClass $F, GluonInput $x)
{
    $x = $self->features->($x);
    $x = $self->output->($x);
    return $x;
}

package AI::MXNet::Gluon::ModelZoo::Vision;

# Specification
my %resnet_spec = (
    18  => ['basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]],
    34  => ['basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]],
    50  => ['bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]],
    101 => ['bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]],
    152 => ['bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048]]
);

my @resnet_net_versions = qw(AI::MXNet::Gluon::ModelZoo::Vision::ResNet::V1 AI::MXNet::Gluon::ModelZoo::Vision::ResNet::V2);
my @resnet_block_versions = (
    {
        basic_block => 'AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BasicBlockV1',
        bottle_neck => 'AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BottleneckV1'
    },
    {
        basic_block => 'AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BasicBlockV2',
        bottle_neck => 'AI::MXNet::Gluon::ModelZoo::Vision::ResNet::BottleneckV2'
    },
);

=head2 get_resnet

    ResNet V1 model from "Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385> paper.
    ResNet V2 model from "Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027> paper.

    Parameters
    ----------
    $version : Int
        Version of ResNet. Options are 1, 2.
    $num_layers : Int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

# Constructor
method get_resnet(
    Int $version, Int $num_layers, Bool :$pretrained=0,
    AI::MXNet::Context :$ctx=AI::MXNet::Context->cpu(),
    Str :$root='~/.mxnet/models',
    Maybe[Int]  :$classes=,
    Maybe[Bool] :$thumbnail=
)
{
    my ($block_type, $layers, $channels) = @{ $resnet_spec{$num_layers} };
    my $resnet_class = $resnet_net_versions[$version-1];
    confess("invalid resnet $version [$version], can be 1,2") unless $resnet_class;
    my $block_class = $resnet_block_versions[$version-1]{$block_type};
    my $net = $resnet_class->new(
        $block_class, $layers, $channels,
        (defined($classes) ? (classes => $classes) : ()),
        (defined($thumbnail) ? (thumbnail => $thumbnail) : ())
    );
    if($pretrained)
    {
        $net->load_parameters(
            AI::MXNet::Gluon::ModelZoo::ModelStore->get_model_file(
                "resnet${num_layers}_v$version",
                root=>$root
            ),
            ctx=>$ctx
        );
    }
    return $net;
}

=head2 resnet18_v1

    ResNet-18 V1 model from "Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet18_v1(%kwargs)
{
    return __PACKAGE__->get_resnet(1, 18, %kwargs);
}

=head2 resnet34_v1

    ResNet-34 V1 model from "Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet34_v1(%kwargs)
{
    return __PACKAGE__->get_resnet(1, 34, %kwargs);
}

=head2 resnet50_v1

    ResNet-50 V1 model from "Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet50_v1(%kwargs)
{
    return __PACKAGE__->get_resnet(1, 50, %kwargs);
}

=head2 resnet101_v1

    ResNet-101 V1 model from "Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet101_v1(%kwargs)
{
    return __PACKAGE__->get_resnet(1, 101, %kwargs);
}

=head2 resnet152_v1

    ResNet-152 V1 model from "Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet152_v1(%kwargs)
{
    return __PACKAGE__->get_resnet(1, 152, %kwargs);
}

=head2 resnet18_v2

    ResNet-18 V2 model from "Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet18_v2(%kwargs)
{
    return __PACKAGE__->get_resnet(2, 18, %kwargs);
}

=head2 resnet34_v2

    ResNet-34 V2 model from "Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet34_v2(%kwargs)
{
    return __PACKAGE__->get_resnet(2, 34, %kwargs);
}

=head2 resnet50_v2

    ResNet-50 V2 model from "Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet50_v2(%kwargs)
{
    return __PACKAGE__->get_resnet(2, 50, %kwargs);
}

=head2 resnet101_v2

    ResNet-101 V2 model from "Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet101_v2(%kwargs)
{
    return __PACKAGE__->get_resnet(2, 101, %kwargs);
}

=head2 resnet152_v2

    ResNet-152 V2 model from "Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method resnet152_v2(%kwargs)
{
    return __PACKAGE__->get_resnet(2, 152, %kwargs);
}

1;
