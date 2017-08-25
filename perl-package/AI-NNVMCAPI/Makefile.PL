use ExtUtils::MakeMaker;
`swig -noproxy -c++ -perl nnvm.i`;
unlink "NNVMCAPI.pm";
my @tmp = split(/ /, $ExtUtils::MakeMaker::Config{lddlflags});
my @lddlflags;
while(my $flag = shift(@tmp))
{
   if($flag eq '-arch')
   {
      my $arch = shift(@tmp);
      if($arch eq 'i386')
      {
         next;
      }
      else
      {
         push @lddlflags, ($flag, $arch);
      }
   }
   else
   {
      push @lddlflags, $flag;
   }
}

WriteMakefile(
    NAME            => 'AI::NNVMCAPI',
    LICENSE         => 'apache_2_0',
    AUTHOR          => 'Sergey Kolychev <sergeykolychev.github@gmail.com>',
    VERSION_FROM    => 'lib/AI/NNVMCAPI.pm',
    ABSTRACT_FROM   => 'lib/AI/NNVMCAPI.pm',
    LIBS           => ['-L../../lib -lmxnet'],
    INC            => '-I../../nnvm/include/nnvm',
    OBJECT         => 'nnvm_wrap.o',
    LDDLFLAGS      => join(' ', @lddlflags),
    PREREQ_PM      => {
        # prereqs
        # build/test prereqs
        'Test::More'   => 0,
    },
    PL_FILES        => {},
);
