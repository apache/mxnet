use ExtUtils::MakeMaker;
use DynaLoader;
`swig -noproxy -c++ -perl mxnet.i`;
unlink "MXNetCAPI.pm";

WriteMakefile(
    NAME            => 'AI::MXNetCAPI',
    LICENSE         => 'apache_2_0',
    AUTHOR          => 'Sergey Kolychev <sergeykolychev.github@gmail.com>',
    VERSION_FROM    => 'lib/AI/MXNetCAPI.pm',
    ABSTRACT_FROM   => 'lib/AI/MXNetCAPI.pm',
    LIBS           => ['-L../../lib -lmxnet'],
    INC            => '-I../../include/mxnet',
    OBJECT         => 'mxnet_wrap.o',
    PREREQ_PM      => {
        # prereqs
        # build/test prereqs
        'Test::More'   => 0,
    },
    PL_FILES        => {},
);
