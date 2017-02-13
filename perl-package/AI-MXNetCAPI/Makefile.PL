use 5.018002;
use ExtUtils::MakeMaker;
use DynaLoader;
locate_mxnet();
WriteMakefile(
    NAME            => 'AI::MXNetCAPI',
    LICENSE         => 'perl',
    AUTHOR          => 'Sergey Kolychev <mxnet4perl@gmail.com>',
    VERSION_FROM    => 'lib/AI/MXNetCAPI.pm',
    ABSTRACT_FROM   => 'lib/AI/MXNetCAPI.pm',
    LIBS           => ['-lmxnet'],
    INC            => '-I/home/developer/mxnet/include/mxnet',
    OBJECT         => 'mxnet_wrap.o',
    PREREQ_PM      => {
        # prereqs
        # build/test prereqs
        'Test::More'   => 0,
    },
    PL_FILES        => {},
);


sub locate_mxnet {
    my @paths = DynaLoader::dl_findfile("-lmxnet");
    die <<'REASON' unless @paths;
Could not load the mxnet library. Please see the README file on how
to install it. Be sure to also install the C headers (c_api.h).
REASON
}