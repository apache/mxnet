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
use Test::More tests => 6;
use JSON::PP;
use AI::MXNet 'mx';

sub check_metric
{
    my ($metric, @args) = @_;
    $metric = mx->metric->create($metric, @args);
    my $str_metric = encode_json($metric->get_config());
    my $metric2 = mx->metric->create($str_metric);
    is_deeply($metric->get_config(), $metric2->get_config());
}


sub test_metrics
{
    check_metric('acc', axis=>0);
    check_metric('f1');
    check_metric('perplexity', -1);
    check_metric('pearsonr');
    check_metric('confidence', 2, [0.5, 0.9]);
    my $composite = mx->metric->create(['acc', 'f1']);
    check_metric($composite);
}

test_metrics();
