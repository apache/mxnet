package AI::MXNet::Function::Parameters;
use strict;
use warnings;
use Function::Parameters ();
use AI::MXNet::Types ();
sub import {
    Function::Parameters->import(
        {
            func => {
                defaults => 'function_strict',
                runtime  => 1,
                reify_type => sub {
                    Mouse::Util::TypeConstraints::find_or_create_isa_type_constraint($_[0])
                }
            },
            method => {
                defaults => 'method_strict',
                runtime  => 1,
                reify_type => sub {
                    Mouse::Util::TypeConstraints::find_or_create_isa_type_constraint($_[0])
                }
            },
        }
    );
}

{
    no warnings 'redefine';
    *Function::Parameters::_croak = sub {
        local($Carp::CarpLevel) = 1;
        Carp::confess ("@_");
    };
}

1;