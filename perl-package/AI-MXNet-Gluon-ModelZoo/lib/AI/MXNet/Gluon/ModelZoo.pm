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

package AI::MXNet::Gluon::ModelZoo;
use strict;
use warnings;
use AI::MXNet qw(mx);
use AI::MXNet::Gluon qw(gluon);
use AI::MXNet::Gluon::NN qw(nn);
use AI::MXNet::Gluon::Contrib;
use AI::MXNet::Gluon::ModelZoo::Vision;
use Exporter;
use base qw(Exporter);
@AI::MXNet::Gluon::ModelZoo::EXPORT_OK = qw(get_model);
our $VERSION = '1.32';

=head1 NAME

    AI::MXNet::Gluon::ModelZoo - A collection of pretrained MXNet Gluon models
=cut

=head1 SYNOPSIS

    ## run forward prediction on random data
    use AI::MXNet::Gluon::ModelZoo qw(get_model);
    my $alexnet = get_model('alexnet', pretrained => 1);
    my $out = $alexnet->(mx->nd->random->uniform(shape=>[1, 3, 224, 224]));
    print $out->aspdl;
=cut

=head1 DESCRIPTION

    This module houses a collection of pretrained models (the parameters are hosted on public mxnet servers).
    https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html
    See examples/image_classification.pl for the example of real time image classification
    using a pretrained model from the ModelZoo
=cut

our %models = qw/
    resnet18_v1 resnet18_v1
    resnet34_v1 resnet34_v1
    resnet50_v1 resnet50_v1
    resnet101_v1 resnet101_v1
    resnet152_v1 resnet152_v1
    resnet18_v2 resnet18_v2
    resnet34_v2 resnet34_v2
    resnet50_v2 resnet50_v2
    resnet101_v2 resnet101_v2
    resnet152_v2 resnet152_v2
    vgg11 vgg11
    vgg13 vgg13
    vgg16 vgg16
    vgg19 vgg19
    vgg11_bn vgg11_bn
    vgg13_bn vgg13_bn
    vgg16_bn vgg16_bn
    vgg19_bn vgg19_bn
    alexnet alexnet
    densenet121 densenet121
    densenet161 densenet161
    densenet169 densenet169
    densenet201 densenet201
    squeezenet1.0 squeezenet1_0
    squeezenet1.1 squeezenet1_1
    inceptionv3 inception_v3
    mobilenet1.0 mobilenet1_0
    mobilenet0.75 mobilenet0_75
    mobilenet0.5 mobilenet0_5
    mobilenet0.25 mobilenet0_25
    mobilenetv2_1.0 mobilenet_v2_1_0
    mobilenetv2_0.75 mobilenet_v2_0_75
    mobilenetv2_0.5 mobilenet_v2_0_5
    mobilenetv2_0.25 mobilenet_v2_0_25
/;


=head2 get_model

    Returns a pre-defined model by name

    Parameters
    ----------
    $name : Str
        Name of the model.
    :$pretrained : Bool
        Whether to load the pretrained weights for model.
    :$classes : Int
        Number of classes for the output layer.
    :$ctx : AI::MXNet::Context, default CPU
        The context in which to load the pretrained weights.
    :$root : Str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
=cut

sub get_model
{
    if(exists $models{lc $_[1]})
    {
        shift;
    }
    my ($name, %kwargs) = @_;
    $name = lc $name;
    Carp::confess(
        "Model $name is not present in the zoo\nValid models are:\n".
        join(', ', sort keys %models)."\n"
    ) unless exists $models{$name};
    my $sub = $models{$name};
    AI::MXNet::Gluon::ModelZoo::Vision->$sub(%kwargs);
}

sub vision { 'AI::MXNet::Gluon::ModelZoo::Vision' }

1;
