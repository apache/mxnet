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

package AI::MXNet::Gluon::ModelZoo::Vision::Inception::V3;
use strict;
use warnings;
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;
use AI::MXNet::Gluon::Mouse;
extends 'AI::MXNet::Gluon::HybridBlock';

func _make_basic_conv(%kwargs)
{
    my $out = nn->HybridSequential(prefix=>'');
    $out->add(nn->Conv2D(use_bias=>0, %kwargs));
    $out->add(nn->BatchNorm(epsilon=>0.001));
    $out->add(nn->Activation('relu'));
    return $out;
}

func _make_branch($use_pool, @conv_settings)
{
    my $out = nn->HybridSequential(prefix=>'');
    if($use_pool eq 'avg')
    {
        $out->add(nn->AvgPool2D(pool_size=>3, strides=>1, padding=>1));
    }
    elsif($use_pool eq 'max')
    {
        $out->add(nn->MaxPool2D(pool_size=>3, strides=>2));
    }
    my @setting_names = ('channels', 'kernel_size', 'strides', 'padding');
    for my $setting (@conv_settings)
    {
        my %kwargs;
        for(enumerate($setting))
        {
            my ($i, $value) = @$_;
            if(defined $value)
            {
                $kwargs{ $setting_names[$i] } = $value;
            }
        }
        $out->add(_make_basic_conv(%kwargs));
    }
    return $out;
}

func _make_A($pool_features, $prefix)
{
    my $out = nn->HybridConcurrent(axis=>1, prefix=>$prefix);
    $out->name_scope(sub {
        $out->add(_make_branch('', [64, 1, undef, undef]));
        $out->add(_make_branch(
            '',
            [48, 1, undef, undef],
            [64, 5, undef, 2]
        ));
        $out->add(_make_branch(
            '',
            [64, 1, undef, undef],
            [96, 3, undef, 1],
            [96, 3, undef, 1]
        ));
        $out->add(_make_branch('avg', [$pool_features, 1, undef, undef]));
    });
    return $out;
}

func _make_B($prefix)
{
    my $out = nn->HybridConcurrent(axis=>1, prefix=>$prefix);
    $out->name_scope(sub {
        $out->add(_make_branch('', [384, 3, 2, undef]));
        $out->add(_make_branch(
            '',
            [64, 1, undef, undef],
            [96, 3, undef, 1],
            [96, 3, 2, undef]
        ));
        $out->add(_make_branch('max'));
    });
    return $out;
}

func _make_C($channels_7x7, $prefix)
{
    my $out = nn->HybridConcurrent(axis=>1, prefix=>$prefix);
    $out->name_scope(sub {
        $out->add(_make_branch('', [192, 1, undef, undef]));
        $out->add(_make_branch(
            '',
            [$channels_7x7, 1, undef, undef],
            [$channels_7x7, [1, 7], undef, [0, 3]],
            [192, [7, 1], undef, [3, 0]]
        ));
        $out->add(_make_branch(
            '',
            [$channels_7x7, 1, undef, undef],
            [$channels_7x7, [7, 1], undef, [3, 0]],
            [$channels_7x7, [1, 7], undef, [0, 3]],
            [$channels_7x7, [7, 1], undef, [3, 0]],
            [192, [1, 7], undef, [0, 3]]
        ));
        $out->add(_make_branch(
            'avg',
            [192, 1, undef, undef]
        ));
    });
    return $out;
}

func _make_D($prefix)
{
    my $out = nn->HybridConcurrent(axis=>1, prefix=>$prefix);
    $out->name_scope(sub {
        $out->add(_make_branch(
            '',
            [192, 1, undef, undef],
            [320, 3, 2, undef]
        ));
        $out->add(_make_branch(
            '',
            [192, 1, undef, undef],
            [192, [1, 7], undef, [0, 3]],
            [192, [7, 1], undef, [3, 0]],
            [192, 3, 2, undef]
        ));
        $out->add(_make_branch('max'));
    });
    return $out;
}

func _make_E($prefix)
{
    my $out = nn->HybridConcurrent(axis=>1, prefix=>$prefix);
    $out->name_scope(sub {
        $out->add(_make_branch('', [320, 1, undef, undef]));

        my $branch_3x3 = nn->HybridSequential(prefix=>'');
        $out->add($branch_3x3);
        $branch_3x3->add(_make_branch(
            '',
            [384, 1, undef, undef]
        ));
        my $branch_3x3_split = nn->HybridConcurrent(axis=>1, prefix=>'');
        $branch_3x3_split->add(_make_branch('', [384, [1, 3], undef, [0, 1]]));
        $branch_3x3_split->add(_make_branch('', [384, [3, 1], undef, [1, 0]]));
        $branch_3x3->add($branch_3x3_split);

        my $branch_3x3dbl = nn->HybridSequential(prefix=>'');
        $out->add($branch_3x3dbl);
        $branch_3x3dbl->add(_make_branch(
            '',
            [448, 1, undef, undef],
            [384, 3, undef, 1]
        ));
        my $branch_3x3dbl_split = nn->HybridConcurrent(axis=>1, prefix=>'');
        $branch_3x3dbl->add($branch_3x3dbl_split);
        $branch_3x3dbl_split->add(_make_branch('', [384, [1, 3], undef, [0, 1]]));
        $branch_3x3dbl_split->add(_make_branch('', [384, [3, 1], undef, [1, 0]]));

        $out->add(_make_branch('avg', [192, 1, undef, undef]));
    });
    return $out;
}

func make_aux($classes)
{
    my $out = nn->HybridSequential(prefix=>'');
    $out->add(nn->AvgPool2D(pool_size=>5, strides=>3));
    $out->add(_make_basic_conv(channels=>128, kernel_size=>1));
    $out->add(_make_basic_conv(channels=>768, kernel_size=>5));
    $out->add(nn->Flatten());
    $out->add(nn->Dense($classes));
    return $out;
}

=head1 NAME

    AI::MXNet::Gluon::ModelZoo::Vision::Inception::V3 - Inception v3 model.
=cut

=head1 DESCRIPTION

    Inception v3 model from
    "Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567> paper.

    Parameters
    ----------
    classes : Int, default 1000
        Number of classification classes.
=cut

has 'classes' => (is => 'ro', isa => 'Int', default => 1000);
method python_constructor_arguments(){ ['classes'] }

sub BUILD
{
    my $self = shift;
    $self->name_scope(sub {
        $self->features(nn->HybridSequential(prefix=>''));
        $self->features->add(_make_basic_conv(channels=>32, kernel_size=>3, strides=>2));
        $self->features->add(_make_basic_conv(channels=>32, kernel_size=>3));
        $self->features->add(_make_basic_conv(channels=>64, kernel_size=>3, padding=>1));
        $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2));
        $self->features->add(_make_basic_conv(channels=>80, kernel_size=>1));
        $self->features->add(_make_basic_conv(channels=>192, kernel_size=>3));
        $self->features->add(nn->MaxPool2D(pool_size=>3, strides=>2));
        $self->features->add(_make_A(32, 'A1_'));
        $self->features->add(_make_A(64, 'A2_'));
        $self->features->add(_make_A(64, 'A3_'));
        $self->features->add(_make_B('B_'));
        $self->features->add(_make_C(128, 'C1_'));
        $self->features->add(_make_C(160, 'C2_'));
        $self->features->add(_make_C(160, 'C3_'));
        $self->features->add(_make_C(192, 'C4_'));
        $self->features->add(_make_D('D_'));
        $self->features->add(_make_E('E1_'));
        $self->features->add(_make_E('E2_'));
        $self->features->add(nn->AvgPool2D(pool_size=>8));
        $self->features->add(nn->Dropout(0.5));

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

=head2 inception_v3

    Inception v3 model from
    "Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567> paper.

    Parameters
    ----------
    :$pretrained : Bool, default 0
        Whether to load the pretrained weights for model.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.
=cut

method inception_v3(
    Bool :$pretrained=0, AI::MXNet::Context :$ctx=AI::MXNet::Context->cpu(),
    Str :$root='~/.mxnet/models', Int :$classes=1000
)
{
    my $net = AI::MXNet::Gluon::ModelZoo::Vision::Inception::V3->new($classes);
    if($pretrained)
    {
        $net->load_parameters(
            AI::MXNet::Gluon::ModelZoo::ModelStore->get_model_file(
                "inceptionv3",
                root=>$root
            ),
            ctx=>$ctx
        );
    }
    return $net;
}

1;