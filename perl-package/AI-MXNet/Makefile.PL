use strict;
use warnings;

use 5.014000;

use ExtUtils::MakeMaker 6.30;



my %WriteMakefileArgs = (
  "ABSTRACT" => "Perl interface to MXNet machine learning library",
  "AUTHOR" => "Sergey Kolychev <sergeykolychev.github\@gmail.com>",
  "BUILD_REQUIRES" => {},
  "CONFIGURE_REQUIRES" => {
    "ExtUtils::MakeMaker" => "6.30"
  },
  "DISTNAME" => "AI-MXNet",
  "EXE_FILES" => [],
  "LICENSE" => "apache_2_0",
  "NAME" => "AI::MXNet",
  "PREREQ_PM" => {
    "AI::MXNetCAPI" => "0.95",
    "AI::NNVMCAPI" => "0.95",
    "Function::Parameters" => "1.0705",
    "Mouse" => "2.1.0",
    "PDL" => "2.007",
    "GraphViz" => "2.14"
  },
  "TEST_REQUIRES" => {},
  "VERSION" => "0.9506",
  "test" => {
    "TESTS" => "t/*.t"
  }
);


my %FallbackPrereqs = (
  "AI::MXNetCAPI" => "0.95",
  "AI::NNVMCAPI" => "0.95",
  "Function::Parameters" => "1.0705",
  "Mouse" => "2.1.0",
  "PDL" => "2.007",
  "GraphViz" => "2.14"
);


unless ( eval { ExtUtils::MakeMaker->VERSION(6.63_03) } ) {
  delete $WriteMakefileArgs{TEST_REQUIRES};
  delete $WriteMakefileArgs{BUILD_REQUIRES};
  $WriteMakefileArgs{PREREQ_PM} = \%FallbackPrereqs;
}

delete $WriteMakefileArgs{CONFIGURE_REQUIRES}
  unless eval { ExtUtils::MakeMaker->VERSION(6.52) };

WriteMakefile(%WriteMakefileArgs);



