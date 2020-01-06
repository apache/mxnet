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

package Bottleneck {
    # Pre-activation residual block
    # Identity Mapping in Deep Residual Networks
    # ref https://arxiv.org/abs/1603.05027
    use AI::MXNet::Gluon::Mouse;
    extends 'AI::MXNet::Gluon::Block';
    has ['inplanes',
           'planes'] => (is => 'rw', required => 1);
    has 'stride'     => (is => 'rw', default => 1);
    has 'downsample' => (is => 'rw');
    has 'norm_layer' => (is => 'rw', default => 'AI::MXNet::Gluon::NN::InstanceNorm');
    method python_constructor_arguments(){ [qw/inplanes planes stride downsample norm_layer/] }
    sub BUILD
    {
        my $self = shift;
        $self->expansion(4);
        if(defined $self->downsample)
        {
            $self->residual_layer(
                nn->Conv2D(
                    in_channels=>$self->inplanes,
                    channels=>$self->planes * $self->expansion,
                    kernel_size=>1, strides=>[$self->stride, $self->stride]
                )
            );
        }
        $self->conv_block(nn->Sequential());
        $self->conv_block->name_scope(sub {
            $self->conv_block->add(
                $self->norm_layer->new(in_channels=>$self->inplanes)
            );
            $self->conv_block->add(nn->Activation('relu'));
            $self->conv_block->add(
                nn->Conv2D(in_channels=>$self->inplanes, 
                    channels=>$self->planes,
                    kernel_size=>1
                )
            );
            $self->conv_block->add($self->norm_layer->new(in_channels=>$self->planes));
            $self->conv_block->add(nn->Activation('relu'));
            $self->conv_block->add(
                ConvLayer->new(
                    $self->planes, $self->planes, kernel_size=>3,
                    stride=>$self->stride
                )
            );
            $self->conv_block->add($self->norm_layer->new(in_channels=>$self->planes));
            $self->conv_block->add(nn->Activation('relu'));
            $self->conv_block->add(
                nn->Conv2D(
                    in_channels=>$self->planes,
                    channels=>$self->planes * $self->expansion,
                    kernel_size=>1
                )
            );
        });
    }

    method forward($x)
    {
        my $residual;
        if(defined $self->downsample)
        {
            $residual = $self->residual_layer->($x);
        }
        else
        {
            $residual = $x;
        }
        return $residual + $self->conv_block->($x);
    }
}

package UpBottleneck {
    # Up-sample residual block (from MSG-Net paper)
    # Enables passing identity all the way through the generator
    # ref https://arxiv.org/abs/1703.06953
    use AI::MXNet::Gluon::Mouse;
    extends 'AI::MXNet::Gluon::Block';
    has ['inplanes',
           'planes'] => (is => 'rw', required => 1);
    has 'stride'     => (is => 'rw', default => 2);
    has 'norm_layer' => (is => 'rw', default => 'AI::MXNet::Gluon::NN::InstanceNorm');
    method python_constructor_arguments(){ [qw/inplanes planes stride norm_layer/] }
    sub BUILD
    {
        my $self = shift;
        $self->expansion(4);
        $self->residual_layer(
            UpsampleConvLayer->new(
                $self->inplanes,
                $self->planes * $self->expansion,
                kernel_size=>1, stride=>1,
                upsample=>$self->stride
            )
        );
        $self->conv_block(nn->Sequential());
        $self->conv_block->name_scope(sub {
            $self->conv_block->add($self->norm_layer->new(in_channels=>$self->inplanes));
            $self->conv_block->add(nn->Activation('relu'));
            $self->conv_block->add(
                nn->Conv2D(
                    in_channels=>$self->inplanes,
                    channels=>$self->planes,
                    kernel_size=>1
                )
            );
            $self->conv_block->add($self->norm_layer->new(in_channels=>$self->planes));
            $self->conv_block->add(nn->Activation('relu'));
            $self->conv_block->add(
                UpsampleConvLayer->new(
                    $self->planes, $self->planes,
                    kernel_size=>3, stride=>1,
                    upsample=>$self->stride
                )
            );
            $self->conv_block->add($self->norm_layer->new(in_channels=>$self->planes));
            $self->conv_block->add(nn->Activation('relu'));
            $self->conv_block->add(
                nn->Conv2D(
                    in_channels=>$self->planes,
                    channels=>$self->planes * $self->expansion,
                    kernel_size=>1
                )
            );
        });
    }

    method forward($x)
    {
        return  $self->residual_layer->($x) + $self->conv_block->($x);
    }
}

package ConvLayer {
    use AI::MXNet::Gluon::Mouse;
    use POSIX qw(floor);
    extends 'AI::MXNet::Gluon::Block';
    has [qw/in_channels out_channels kernel_size stride/] => (is => 'rw');
    method python_constructor_arguments(){ [qw/in_channels out_channels kernel_size stride/] }
    sub BUILD
    {
        my $self = shift;
        $self->pad(nn->ReflectionPad2D(floor($self->kernel_size/2)));
        $self->conv2d(
            nn->Conv2D(
                in_channels=>$self->in_channels,
                channels=>$self->out_channels,
                kernel_size=>$self->kernel_size,
                strides=>[$self->stride, $self->stride],
                padding=>0
            )
        );
    }

    method forward($x)
    {
        $x = $self->pad->($x);
        my $out = $self->conv2d->($x);
        return $out;
    }
}


package UpsampleConvLayer {
    # UpsampleConvLayer
    # Upsamples the input and then does a convolution. This method gives better results
    # compared to ConvTranspose2d.
    # ref: http://distill.pub/2016/deconv-checkerboard/
    use AI::MXNet::Gluon::Mouse;
    use POSIX qw(floor);
    extends 'AI::MXNet::Gluon::Block';
    has [qw/in_channels out_channels kernel_size stride upsample/] => (is => 'rw');
    method python_constructor_arguments(){ [qw/in_channels out_channels kernel_size stride upsample/] }
    sub BUILD
    {
        my $self = shift;
        $self->conv2d(
            nn->Conv2D(
                in_channels=>$self->in_channels,
                channels=>$self->out_channels,
                kernel_size=>$self->kernel_size,
                strides=>[$self->stride, $self->stride],
                padding=>floor($self->kernel_size/2)
            )
        );
    }

    method forward($x)
    {
        if($self->upsample)
        {
            $x = nd->UpSampling($x, scale=>$self->upsample, sample_type=>'nearest');
        }
        my $out = $self->conv2d->($x);
        return $out;
    }
}

package GramMatrix {
    use AI::MXNet::Gluon::Mouse;
    extends 'AI::MXNet::Gluon::Block';
    method forward($x)
    {
        my ($b, $ch, $h, $w) = @{ $x->shape };
        my $features = $x->reshape([$b, $ch, $w * $h]);
        my $gram = nd->batch_dot($features, $features, transpose_b=>1) / ($ch * $h * $w);
        return $gram;
    }
};

package Inspiration {
    # Inspiration Layer (from MSG-Net paper)
    # tuning the featuremap with target Gram Matrix
    # ref https://arxiv.org/abs/1703.06953
    use AI::MXNet::Gluon::Mouse;
    extends 'AI::MXNet::Gluon::Block';
    has 'C' => (is => 'rw', required => 1);
    has 'B' => (is => 'rw', default => 1);
    method python_constructor_arguments(){ [qw/C B/] }
    sub BUILD
    {
        my $self = shift;
        $self->weight($self->params->get('weight', shape=>[1,$self->C,$self->C],
                                      init=>mx->initializer->Uniform(),
                                      allow_deferred_init=>1));
        $self->gram(nd->random->uniform(shape=>[$self->B, $self->C, $self->C]));
    }

    method set_target($target)
    {
        $self->gram($target);
    }

    method forward($x)
    {
        $self->P(nd->batch_dot($self->weight->data->broadcast_to($self->gram->shape), $self->gram));
        return nd->batch_dot(
                nd->SwapAxis($self->P,1,2)->broadcast_to([$x->shape->[0], $self->C, $self->C]),
                $x->reshape([0, 0, $x->shape->[2]*$x->shape->[3]])
        )->reshape($x->shape);
    }
}

package Net {
    use AI::MXNet::Gluon::Mouse;
    extends 'AI::MXNet::Gluon::Block';
    has 'input_nc'    => (is => 'rw', default => 3);
    has 'output_nc'   => (is => 'rw', default => 3);
    has 'ngf'         => (is => 'rw', default => 64);
    has 'norm_layer'  => (is => 'rw', default => 'AI::MXNet::Gluon::NN::InstanceNorm');
    has 'n_blocks'    => (is => 'rw', default => 6);
    has 'gpu_ids'     => (is => 'rw', default => sub { [] });
    method python_constructor_arguments(){ [qw/input_nc output_nc ngf norm_layer n_blocks gpu_ids/] }
    sub BUILD
    {
        my $self = shift;
        $self->gram(GramMatrix->new);

        my $block = 'Bottleneck';
        my $upblock = 'UpBottleneck';
        my $expansion = 4;

        $self->name_scope(sub {
            $self->model1(nn->Sequential());
            $self->ins(Inspiration->new($self->ngf*$expansion));
            $self->model(nn->Sequential());

            $self->model1->add(ConvLayer->new($self->input_nc, 64, kernel_size=>7, stride=>1));
            $self->model1->add($self->norm_layer->new(in_channels=>64));
            $self->model1->add(nn->Activation('relu'));
            $self->model1->add($block->new(64, 32, 2, 1, $self->norm_layer));
            $self->model1->add($block->new(32*$expansion, $self->ngf, 2, 1, $self->norm_layer));

            $self->model->add($self->model1);
            $self->model->add($self->ins);

            for(1..$self->n_blocks)
            {
                $self->model->add($block->new($self->ngf*$expansion, $self->ngf, 1, undef, $self->norm_layer));
            }

            $self->model->add($upblock->new($self->ngf*$expansion, 32, 2, $self->norm_layer));
            $self->model->add($upblock->new(32*$expansion, 16, 2, $self->norm_layer));
            $self->model->add($self->norm_layer->new(in_channels=>16*$expansion));
            $self->model->add(nn->Activation('relu'));
            $self->model->add(ConvLayer->new(16*$expansion, $self->output_nc, kernel_size=>7, stride=>1));
        });
    }

    method set_target($x)
    {
        my $F = $self->model1->($x);
        my $G = $self->gram->($F);
        $self->ins->set_target($G);
    }

    method forward($input)
    {
        return $self->model->($input);
    }
}

1;
