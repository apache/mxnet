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

package AI::MXNet::Types;
use strict;
use warnings;
use Mouse::Util::TypeConstraints;
use Exporter;
use base qw(Exporter);
@AI::MXNet::Types::EXPORT = qw(find_type_constraint enum);

class_type 'PDL';
class_type 'PDL::Matrix';
class_type 'AI::MXNet::NDArray';
class_type 'AI::MXNet::Symbol';
class_type 'AI::MXNet::NDArray::Slice';
class_type 'AI::MXNet::Executor';
class_type 'AI::MXNet::DataDesc';
class_type 'AI::MXNet::Callback';
class_type 'AI::MXNet::EvalMetric';
class_type 'AI::MXNet::DataParallelExecutorGroup';
class_type 'AI::MXNet::Optimizer';
class_type 'AI::MXNet::Initializer';
class_type 'AI::MXNet::KVStore';
class_type 'AI::MXNet::InitDesc';
class_type 'AI::MXNet::IRHeader';
class_type 'AI::MXNet::Updater';
class_type 'AI::MXNet::KVStore';
class_type 'AI::MXNet::Gluon::Block';
class_type 'AI::MXNet::Gluon::Data::Set';
class_type 'AI::MXNet::Gluon::RNN::HybridRecurrentCell';
class_type 'AI::MXNet::Symbol::NameManager';
subtype "AcceptableInput" => as "Num|PDL|PDL::Matrix|AI::MXNet::NDArray|AI::MXNet::NDArray::Slice|ArrayRef";
subtype "Index"           => as "Int";
subtype "DimSize"         => as "Int" => where { $_ >= 0 };
subtype "Dropout"         => as "Num" => where { $_ >= 0 and $_ <= 1 };
subtype "Shape"           => as "ArrayRef[DimSize]";
subtype "CudaKernelShape" => as "Shape" => where { @$_ == 3 };
subtype "WholeDim"        => as "Str" => where { $_ eq 'X' };
subtype "Slice"           => as "ArrayRef[Index]|WholeDim|Index" => where { ref $_ ? @$_ > 0 : 1 };
subtype "Dtype"           => as enum([qw[float32 float64 float16 uint8 int32]]);
subtype "ProfilerMode"    => as enum([qw[symbolic all]]);
subtype "GluonClass"      => as enum([qw[AI::MXNet::NDArray AI::MXNet::Symbol]]);
subtype "GluonInput"      => as "AI::MXNet::NDArray|AI::MXNet::Symbol|ArrayRef[AI::MXNet::NDArray|AI::MXNet::Symbol]";
subtype "ProfilerState"   => as enum([qw[stop run]]);
subtype "GradReq"         => as enum([qw[add write null]]);
subtype "KVStoreStr"      => as enum([qw[local device dist dist_sync dist_async]]);
subtype "PoolType"        => as enum([qw[max avg sum]]);
subtype "NameShape"       => as "ArrayRef" => where {
    find_type_constraint("Str")->check($_->[0])
        and
    find_type_constraint("Shape")->check($_->[1])
};
subtype "Callback"        => as "CodeRef|ArrayRef[Coderef]|AI::MXNet::Callback|ArrayRef[AI::MXNet::Callback]";
subtype "EvalMetric"      => as "AI::MXNet::EvalMetric|Str|CodeRef";
subtype "Metric"          => as "Maybe[EvalMetric]";
subtype "Optimizer"       => as "AI::MXNet::Optimizer|Str";
subtype "Initializer"     => as "AI::MXNet::Initializer|Str";
subtype "Updater"         => as "AI::MXNet::Updater|CodeRef";
subtype "KVStore"         => as "AI::MXNet::KVStore|KVStoreStr";
subtype "Activation"      => as "AI::MXNet::Symbol|Str|CodeRef";
subtype "SymbolOrArrayOfSymbols" => as "AI::MXNet::Symbol|ArrayRef[AI::MXNet::Symbol]";
subtype "NameShapeOrDataDesc" => as "NameShape|AI::MXNet::DataDesc";
subtype "AdvancedSlice"   => as "ArrayRef[ArrayRef|PDL|PDL::Matrix|AI::MXNet::NDArray]";

1;
