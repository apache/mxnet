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
class_type 'AI::MXNet::InitDesc';
class_type 'AI::MXNet::IRHeader';
subtype "AcceptableInput" => as "Num|PDL|PDL::Matrix|AI::MXNet::NDArray|AI::MXNet::NDArray::Slice|ArrayRef";
subtype "Index"           => as "Int";
subtype "DimSize"         => as "Int" => where { $_ >= 0 };
subtype "Shape"           => as "ArrayRef[DimSize]";
subtype "WholeDim"        => as "Str" => where { $_ eq 'X' };
subtype "Slice"           => as "ArrayRef[Index]|WholeDim|Index" => where { ref $_ ? @$_ > 0 : 1 };
subtype "Dtype"           => as enum([qw[float32 float64 float16 uint8 int32]]);
subtype "Metric"          => as "Maybe[CodeRef|Str]";
subtype "ProfilerMode"    => as enum([qw[symbolic all]]);
subtype "ProfilerState"   => as enum([qw[stop run]]);
subtype "GradReq"         => as enum([qw[add write null]]);
subtype "NameShape"       => as "ArrayRef" => where {
    find_type_constraint("Str")->check($_->[0])
        and
    find_type_constraint("Shape")->check($_->[1])
};
subtype "Callback"        => as "CodeRef|ArrayRef[Coderef]|AI::MXNet::Callback|ArrayRef[AI::MXNet::Callback]";
subtype "EvalMetric"      => as "AI::MXNet::EvalMetric|Str|CodeRef";
subtype "Optimizer"       => as "AI::MXNet::Optimizer|Str";
subtype "Activation"      => as "AI::MXNet::Symbol|Str";
subtype "SymbolOrArrayOfSymbols" => as "AI::MXNet::Symbol|ArrayRef[AI::MXNet::Symbol]";
subtype "NameShapeOrDataDesc" => as "NameShape|AI::MXNet::DataDesc";
