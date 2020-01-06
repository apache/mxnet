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

func tensor_load_rgbimage($filename, $size=)
{
    my $img = mx->image->imread($filename);
    if($size)
    {
        $img = mx->image->resize_short($img, $size);
    }
    return $img->transpose([2,0,1])->expand_dims(axis=>0)->astype('float32');
}

func tensor_save_rgbimage($img, $filename)
{
    $img = nd->clip($img, a_min => 0, a_max => 255)->transpose([1,2,0])->aspdl;
    $img->slice('X', 'X', '-1:0')->byte->wpic($filename);
}

func tensor_save_bgrimage($tensor, $filename)
{
    $tensor = $tensor->at(0);
    my ($b, $g, $r) = @{ nd->split($tensor, num_outputs=>3, axis=>0) };
    $tensor = nd->concat($r, $g, $b, dim=>0);
    tensor_save_rgbimage($tensor, $filename);
}


func preprocess_batch($batch)
{
    $batch = nd->swapaxes($batch, 0, 1);
    my ($r, $g, $b) = @{ nd->split($batch, num_outputs=>3, axis=>0) };
    $batch = nd->concat($b, $g, $r, dim=>0);
    $batch = nd->swapaxes($batch, 0, 1);
    return $batch;
}

func evaluate(%args)
{
    my $ctx = mx->cpu;
    # images
    my $content_image = tensor_load_rgbimage($args{content_image}, $args{content_size});
    my $style_image = tensor_load_rgbimage($args{style_image}, $args{style_size});
    $style_image = preprocess_batch($style_image);
    # model
    my $style_model = Net->new(ngf=>$args{ngf});
    $style_model->load_parameters($args{model}, ctx=>$ctx);

    # forward
    $style_model->set_target($style_image);
    my $output = $style_model->($content_image);
    tensor_save_bgrimage($output->[0], $args{output_image});
}

1;
