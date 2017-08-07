package AI::MXNet::Image;
use strict;
use warnings;
use Scalar::Util qw(blessed);
use AI::MXNet::Base;
use AI::MXNet::Function::Parameters;

=head1 NAME

    AI::MXNet:Image - Read individual image files and perform augmentations.
=cut

=head2 imdecode

    Decode an image from string. Requires OpenCV to work.

    Parameters
    ----------
    $buf : str, array ref, pdl, ndarray
        Binary image data.
    :$flag : int
        0 for grayscale. 1 for colored.
    :$to_rgb : int
        0 for BGR format (OpenCV default). 1 for RGB format (MXNet default).
    :$out : NDArray
        Output buffer. Do not specify for automatic allocation.
=cut

method imdecode(Str|PDL $buf, Int :$flag=1, Int :$to_rgb=1, Maybe[AI::MXNet::NDArray] :$out=)
{
    if(not ref $buf)
    {
        my $pdl_type = PDL::Type->new(DTYPE_MX_TO_PDL->{'uint8'});
        my $len; { use bytes; $len = length $buf; }
        my $pdl = PDL->new_from_specification($pdl_type, $len);
        ${$pdl->get_dataref} = $buf;
        $pdl->upd_data;
        $buf = $pdl;
    }
    if(not (blessed $buf and $buf->isa('AI::MXNet::NDArray')))
    {
        $buf = AI::MXNet::NDArray->array($buf, dtype=>'uint8');
    }
    return AI::MXNet::NDArray->_cvimdecode($buf, { flag => $flag, to_rgb => $to_rgb, ($out ? (out => $out) : ()) });
}

=head2 scale_down

Scale down crop size if it's bigger than the image size.

    Parameters:
    -----------
    Shape $src_size
    Shape $size

    Returns:
    --------
    ($w, $h)
=cut

method scale_down(Shape $src_size, Shape $size)
{
    my ($w, $h) = @{ $size };
    my ($sw, $sh) = @{ $src_size };
    if($sh < $h)
    {
        ($w, $h) = (($w*$sh)/$h, $sh);
    }
    if($sw < $w)
    {
        ($w, $h) = ($sw, ($h*$sw)/$w);
    }
    return (int($w), int($h));
}

=head2 resize_short

    Resize shorter edge to the size.

    Parameters:
    -----------
    AI::MXNet::NDArray $src
    Int                $size
    Int                $interp=2

    Returns:
    --------
    AI::MXNet::NDArray $resized_image
=cut

method resize_short(AI::MXNet::NDArray $src, Int $size, Int $interp=2)
{
    my ($new_h, $new_w);
    my ($h, $w) = @{ $src->shape };
    if($h > $w)
    {
        ($new_h, $new_w) = ($size*$h/$w, $size);
    }
    else
    {
        ($new_h, $new_w) = ($size, $size*$w/$h);
    }
    return AI::MXNet::NDArray->_cvimresize($src, $new_w, $new_h, { interp=>$interp });
}

=head2 fixed_crop

    Crop src at fixed location, and (optionally) resize it to the size.

    Parameters:
    -----------
    AI::MXNet::NDArray $src
    Int                $x0
    Int                $y0
    Int                $w
    Int                $h
    Maybe[Shape]       $size=
    Int                $interp=2

    Returns:
    --------
    AI::MXNet::NDArray $cropped_image
=cut

method fixed_crop(AI::MXNet::NDArray $src, Int $x0, Int $y0, Int $w, Int $h, Maybe[Shape] $size=, Int $interp=2)
{
    my $out = AI::MXNet::NDArray->crop($src, { begin=>[$y0, $x0, 0], end=>[$y0+$h, $x0+$w, $src->shape->[2]] });
    if(defined $size and join(',', $w, $h) ne join(',', @{ $size }))
    {
        $out = AI::MXNet::NDArray->_cvimresize($out, @{ $size }, { interp=>$interp });
    }
    return $out;
}

=head2 random_crop

    Randomly crop src with size. Upsample result if src is smaller than the size.

    Parameters:
    -----------
    AI::MXNet::NDArray $src
    Shape              $size=
    Int                $interp=2

    Returns:
    --------
    ($cropped_image, [$x0, $y0, $new_w, $new_h])
=cut

method random_crop(AI::MXNet::NDArray $src, Shape $size, Int $interp=2)
{
    my ($h, $w) = @{ $src->shape };
    my ($new_w, $new_h) = __PACKAGE__->scale_down([$w, $h], $size);

    my $x0 = int(rand($w - $new_w + 1));
    my $y0 = int(rand($h - $new_h + 1));

    my $out = __PACKAGE__->fixed_crop($src, $x0, $y0, $new_w, $new_h, $size, $interp);
    return ($out, [$x0, $y0, $new_w, $new_h]);
}

=head2 center_crop

    Randomly crop src with size around the center. Upsample result if src is smaller than the size.

    Parameters:
    -----------
    AI::MXNet::NDArray $src
    Shape              $size=
    Int                $interp=2

    Returns:
    --------
    ($cropped_image, [$x0, $y0, $new_w, $new_h])
=cut

method center_crop(AI::MXNet::NDArray $src, Shape $size, Int $interp=2)
{
    my ($h, $w) = @{ $src->shape };
    my ($new_w, $new_h) = __PACKAGE__->scale_down([$w, $h], $size);

    my $x0 = int(($w - $new_w)/2);
    my $y0 = int(($h - $new_h)/2);

    my $out = __PACKAGE__->fixed_crop($src, $x0, $y0, $new_w, $new_h, $size, $interp);
    return ($out, [$x0, $y0, $new_w, $new_h]);
}

=head2 color_normalize

    Normalize src with mean and std.

    Parameters:
    -----------
    AI::MXNet::NDArray $src
    Num|AI::MXNet::NDArray $mean
    Maybe[Num|AI::MXNet::NDArray] $std=
    Int $interp=2

    Returns:
    --------
    AI::MXNet::NDArray $normalized_image
=cut

method color_normalize(AI::MXNet::NDArray $src, Num|AI::MXNet::NDArray $mean, Maybe[Num|AI::MXNet::NDArray] $std=)
{
    $src -= $mean;
    if(defined $std)
    {
        $src /= $std;
    }
    return $src;
}

=head2 random_size_crop

    Randomly crop src with size. Randomize area and aspect ratio.

    Parameters:
    -----------
    AI::MXNet::NDArray $src
    Shape              $size
    Num                $min_area
    ArrayRef[Int]      [$from, $to] # $ratio
    Maybe[Int]         $interp=2

    Returns:
    --------
    ($cropped_image, [$x0, $y0, $new_w, $new_h])
=cut

method random_size_crop(AI::MXNet::NDArray $src, Shape $size, Num $min_area, ArrayRef[Num] $ratio, Maybe[Int] $interp=2)
{
    my ($h, $w) = @{ $src->shape };
    my ($from, $to) = @{ $ratio };
    my $new_ratio = $from + ($to-$from) * rand;
    my $max_area;
    if($new_ratio * $h > $w)
    {
        $max_area = $w*int($w/$new_ratio);
    }
    else
    {
        $max_area = $h*int($h*$new_ratio);
    }

    $min_area *= $h*$w;
    if($max_area < $min_area)
    {
        return __PACKAGE__->random_crop($src, $size, $interp);
    }
    my $new_area = $min_area + ($max_area-$min_area) * rand;
    my $new_w = int(sqrt($new_area*$new_ratio));
    my $new_h = $new_w;

    assert($new_w <= $w and $new_h <= $h);
    my $x0 = int(rand($w - $new_w + 1));
    my $y0 = int(rand($h - $new_h + 1));

    my $out = __PACKAGE__->fixed_crop($src, $x0, $y0, $new_w, $new_h, $size, $interp);
    return ($out, [$x0, $y0, $new_w, $new_h]);
}

=head2 ResizeAug

    Makes "resize shorter edge to size augumenter" closure.

    Parameters:
    -----------
    Shape              $size
    Int                $interp=2

    Returns:
    --------
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns [__PACKAGE__->resize_short($src, $size, $interp)]
=cut

method ResizeAug(Shape $size, Int $interp=2)
{
    my $aug = sub {
        my $src = shift;
        return [__PACKAGE__->resize_short($src, $size, $interp)];
    };
    return $aug;
}

=head2 RandomCropAug

    Makes "random crop augumenter" closure.

    Parameters:
    -----------
    Shape              $size
    Int                $interp=2

    Returns:
    --------
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns [(__PACKAGE__->random_crop($src, $size, $interp))[0]]
=cut

method RandomCropAug(Shape $size, Int $interp=2)
{
    my $aug = sub {
        my $src = shift;
        return [(__PACKAGE__->random_crop($src, $size, $interp))[0]];
    };
    return $aug;
}

=head2 RandomSizedCropAug

    Makes "random crop augumenter" closure.

    Parameters:
    -----------
    Shape              $size
    Num                $min_area
    ArrayRef[Num]      $ratio
    Int                $interp=2

    Returns:
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns [(__PACKAGE__->random_size_crop($src, $size, $min_area, $ratio, $interp))[0]]
=cut

method RandomSizedCropAug(Shape $size, Num $min_area, ArrayRef[Num] $ratio, Int $interp=2)
{
    my $aug = sub {
        my $src = shift;
        return [(__PACKAGE__->random_size_crop($src, $size, $min_area, $ratio, $interp))[0]];
    };
    return $aug;
}

=head2 CenterCropAug

    Makes "center crop augumenter" closure.

    Parameters:
    -----------
    Shape              $size
    Int                $interp=2

    Returns:
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns [(__PACKAGE__->center_crop($src, $size, $interp))[0]]
=cut

method CenterCropAug(Shape $size, Int $interp=2)
{
    my $aug = sub {
        my $src = shift;
        return [(__PACKAGE__->center_crop($src, $size, $interp))[0]];
    };
    return $aug;
}

=head2 RandomOrderAug

    Makes "Apply list of augmenters in random order" closure.

    Parameters:
    -----------
    ArrayRef[CodeRef]  $ts

    Returns:
    --------
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns ArrayRef[AI::MXNet::NDArray]
=cut

method RandomOrderAug(ArrayRef[CodeRef] $ts)
{
    my $aug = sub {
        my $src = shift;
        my @ts = List::Util::shuffle(@{ $ts });
        my @tmp;
        for my $t (@ts)
        {
            push @tmp, &{$t}($src);
        }
        return \@tmp;
    };
    return $aug;
}

=head2 RandomOrderAug

    Makes "Apply random brightness, contrast and saturation jitter in random order" closure

    Parameters:
    -----------
    Num $brightness
    Num $contrast
    Num $saturation

    Returns:
    --------
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns ArrayRef[AI::MXNet::NDArray]
=cut

method ColorJitterAug(Num $brightness, Num $contrast, Num $saturation)
{
    my @ts;
    my $coef = AI::MXNet::NDArray->array([[[0.299, 0.587, 0.114]]]);
    if($brightness > 0)
    {
        my $baug = sub { my $src = shift;
            my $alpha = 1 + -$brightness + 2 * $brightness * rand;
            $src *= $alpha;
            return [$src];
        };
        push @ts, $baug;
    }

    if($contrast > 0)
    {
        my $caug = sub { my $src = shift;
            my $alpha = 1 + -$contrast + 2 * $contrast * rand;
            my $gray  = $src*$coef;
            $gray = (3.0*(1.0-$alpha)/$gray->size)*$gray->sum;
            $src *= $alpha;
            $src += $gray;
            return [$src];
        };
        push @ts, $caug;
    }

    if($saturation > 0)
    {
        my $saug = sub { my $src = shift;
            my $alpha = 1 + -$saturation + 2 * $saturation * rand;
            my $gray  = $src*$coef;
            $gray = AI::MXNet::NDArray->sum($gray, { axis=>2, keepdims =>1 });
            $gray *= (1.0-$alpha);
            $src *= $alpha;
            $src += $gray;
            return [$src];
        };
        push @ts, $saug;
    }

    return __PACKAGE__->RandomOrderAug(\@ts);
}

=head2 LightingAug

    Makes "Add PCA based noise" closure.

    Parameters:
    -----------
    Num $alphastd
    PDL $eigval
    PDL $eigvec

    Returns:
    --------
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns ArrayRef[AI::MXNet::NDArray]
=cut

method LightingAug(Num $alphastd, PDL $eigval, PDL $eigvec)
{
    my $aug = sub { my $src = shift;
        my $alpha = AI::MXNet::NDArray->zeros([3]);
        AI::MXNet::Random->normal(0, $alphastd, { out => $alpha });
        my $rgb = ($eigvec*$alpha->aspdl) x $eigval;
        $src += AI::MXNet::NDArray->array($rgb);
        return [$src]
    };
    return $aug
}

=head2 ColorNormalizeAug

    Makes "Mean and std normalization" closure.

    Parameters:
    -----------
    PDL $mean
    PDL $std

    Returns:
    --------
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns [__PACKAGE__->color_normalize($src, $mean, $std)]
=cut

method ColorNormalizeAug(PDL $mean, PDL $std)
{
    $mean = AI::MXNet::NDArray->array($mean);
    $std = AI::MXNet::NDArray->array($std);
    my $aug = sub { my $src = shift;
        return [__PACKAGE__->color_normalize($src, $mean, $std)]
    };
    return $aug;
}

=head2 HorizontalFlipAug

    Makes "Random horizontal flipping" closure.

    Parameters:
    -----------
    Num $p < 1

    Returns:
    --------
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns [$p > rand ? AI::MXNet::NDArray->flip($src, axis=1>) : $src]
=cut

method HorizontalFlipAug(Num $p)
{
    my $aug = sub { my $src = shift;
        return [$p > rand() ? AI::MXNet::NDArray->flip($src, { axis=>1 }) : $src]
    };
    return $aug;
}

=head2 CastAug

    Makes "Cast to float32" closure.

    Returns:
    --------
    CodeRef that accepts AI::MXNet::NDArray $src as input
    and returns [$src->astype('float32')]
=cut

method CastAug()
{
    my $aug = sub { my $src = shift;
        return [$src->astype('float32')]
    };
    return $aug;
}

=head2 CreateAugmenter

    Create augumenter list

    Parameters:
    -----------
    Shape          :$data_shape,
    Bool           :$resize=0,
    Bool           :$rand_crop=0,
    Bool           :$rand_resize=0,
    Bool           :$rand_mirror=0,
    Maybe[Num|PDL] :$mean=,
    Maybe[Num|PDL] :$std=,
    Num            :$brightness=0,
    Num            :$contrast=0,
    Num            :$saturation=0,
    Num            :$pca_noise=0,
    Int            :$inter_method=2
=cut

method CreateAugmenter(
Shape          :$data_shape,
Bool           :$resize=0,
Bool           :$rand_crop=0,
Bool           :$rand_resize=0,
Bool           :$rand_mirror=0,
Maybe[Num|PDL] :$mean=,
Maybe[Num|PDL] :$std=,
Num            :$brightness=0,
Num            :$contrast=0,
Num            :$saturation=0,
Num            :$pca_noise=0,
Int            :$inter_method=2
)
{
    my @auglist;
    if($resize > 0)
    {
        push @auglist, __PACKAGE__->ResizeAug($resize, $inter_method);
    }

    my $crop_size = [$data_shape->[2], $data_shape->[1]];
    if($rand_resize)
    {
        assert($rand_crop);
        push @auglist, __PACKAGE__->RandomSizedCropAug($crop_size, 0.3, [3.0/4.0, 4.0/3.0], $inter_method);
    }
    elsif($rand_crop)
    {
        push @auglist, __PACKAGE__->RandomCropAug($crop_size, $inter_method);
    }
    else
    {
        push @auglist, __PACKAGE__->CenterCropAug($crop_size, $inter_method);
    }

    if($rand_mirror)
    {
        push @auglist, __PACKAGE__->HorizontalFlipAug(0.5);
    }

    push @auglist, __PACKAGE__->CastAug;

    if($brightness or $contrast or $saturation)
    {
        push @auglist, __PACKAGE__->ColorJitterAug($brightness, $contrast, $saturation);
    }
    if($pca_noise > 0)
    {
        my $eigval = AI::MXNet::NDArray->array([55.46, 4.794, 1.148])->aspdl;
        my $eigvec = AI::MXNet::NDArray->array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])->aspdl;
        push @auglist, __PACKAGE__->LightingAug($pca_noise, $eigval, $eigvec);
    }

    if($mean)
    {
        $mean = AI::MXNet::NDArray->array([123.68, 116.28, 103.53])->aspdl;
    }
    if($std)
    {
        $std = AI::MXNet::NDArray->array([58.395, 57.12, 57.375])->aspdl;
    }
    if(defined $mean)
    {
        assert(defined $std);
        push @auglist, __PACKAGE__->ColorNormalizeAug($mean, $std);
    }

    return \@auglist;
}

method ImageIter(@args) { AI::MXNet::ImageIter->new(@args) }

package AI::MXNet::ImageIter;
use Mouse;
use AI::MXNet::Base;
extends 'AI::MXNet::DataIter';

=head1 NAME

    AI::MXNet::ImageIter - Image data iterator.
=cut

=head1 DESCRIPTION


    Image data iterator with a large number of augumentation choices.
    Supports reading from both .rec files and raw image files with image list.

    To load from .rec files, please specify path_imgrec. Also specify path_imgidx
    to use data partition (for distributed training) or shuffling.

    To load from raw image files, specify path_imglist and path_root.

    Parameters
    ----------
    batch_size : Int
        Number of examples per batch
    data_shape : Shape
        Data shape in (channels, height, width).
        For now, only RGB image with 3 channels is supported.
    label_width : Int
        dimension of label
    path_imgrec : str
        path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec
    path_imglist : str
        path to image list (.lst)
        Created with tools/im2rec.py or with custom script.
        Format: index\t[one or more label separated by \t]\trelative_path_from_root
    imglist: array ref
        a list of image with the label(s)
        each item is a list [imagelabel: float or array ref of float, imgpath]
    path_root : str
        Root folder of image files
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration.
    Can be slow for HDD.
    part_index : int
        Partition index
    num_parts : int
        Total number of partitions.
    data_name='data' Str
    label_name='softmax_label' Str
    kwargs : hash ref with any additional arguments for augmenters
=cut

has 'batch_size'  => (is => 'ro', isa => 'Int',   required => 1);
has 'data_shape'  => (is => 'ro', isa => 'Shape', required => 1);
has 'label_width' => (is => 'ro', isa => 'Int',   default  => 1);
has 'data_name'   => (is => 'ro', isa => 'Str',   default  => 'data');
has 'label_name'  => (is => 'ro', isa => 'Str',   default  => 'softmax_label');
has [qw/path_imgrec
        path_imglist
        path_root
        path_imgidx
    /]            => (is => 'ro', isa => 'Str');
has 'shuffle'     => (is => 'ro', isa => 'Bool', default => 0);
has 'part_index'  => (is => 'ro', isa => 'Int', default => 0);
has 'num_parts'   => (is => 'ro', isa => 'Int', default => 0);
has 'aug_list'    => (is => 'rw', isa => 'ArrayRef[CodeRef]');
has 'imglist'     => (is => 'rw', isa => 'ArrayRef|HashRef');
has 'kwargs'      => (is => 'ro', isa => 'HashRef');
has [qw/imgidx
        imgrec
        seq
        cur
        provide_data
        provide_label
           /]     => (is => 'rw', init_arg => undef);

sub BUILD
{
    my $self = shift;
    assert($self->path_imgrec or $self->path_imglist or ref $self->imglist eq 'ARRAY');
    if($self->path_imgrec)
    {
        print("loading recordio...\n");
        if($self->path_imgidx)
        {
            $self->imgrec(
                AI::MXNet::IndexedRecordIO->new(
                    idx_path => $self->path_imgidx,
                    uri => $self->path_imgrec,
                    flag => 'r'
                )
            );
            $self->imgidx([@{ $self->imgrec->keys }]);
        }
        else
        {
            $self->imgrec(AI::MXNet::RecordIO->new(uri => $self->path_imgrec, flag => 'r'));
        }
    }
    my %imglist;
    my @imgkeys;
    if($self->path_imglist)
    {
        print("loading image list...\n");
        open(my $f, $self->path_imglist) or confess("can't open ${\ $self->path_imglist } : $!");
        while(my $line = <$f>)
        {
            chomp($line);
            my @line = split(/\t/, $line);
            my $label = AI::MXNet::NDArray->array([@line[1..@line-1]]);
            my $key   = $line[0];
            $imglist{$key} = [$label, $line[-1]];
            push @imgkeys, $key;
        }
        $self->imglist(\%imglist);
    }
    elsif(ref $self->imglist eq 'ARRAY')
    {
        print("loading image list...\n");
        my %result;
        my $index = 1;
        for my $img (@{ $self->imglist })
        {
            my $key = $index++;
            my $label;
            if(not ref $img->[0])
            {
                $label = AI::MXNet::NDArray->array([$img->[0]]);
            }
            else
            {
                $label = AI::MXNet::NDArray->array($img->[0]);
                $result{$key} = [$label, $img->[1]];
                push @imgkeys, $key;
            }
        }
        $self->imglist(\%result);
    }
    assert(@{ $self->data_shape } == 3 and $self->data_shape->[0] == 3);
    $self->provide_data([
        AI::MXNet::DataDesc->new(
            name  => $self->data_name,
            shape => [$self->batch_size, @{ $self->data_shape }]
        )
    ]);
    if($self->label_width > 1)
    {
        $self->provide_label([
            AI::MXNet::DataDesc->new(
                name  => $self->label_name,
                shape => [$self->batch_size, $self->label_width]
            )
        ]);
    }
    else
    {
        $self->provide_label([
            AI::MXNet::DataDesc->new(
                name  => $self->label_name,
                shape => [$self->batch_size]
            )
        ]);
    }
    if(not defined $self->imgrec)
    {
        $self->seq(\@imgkeys);
    }
    elsif($self->shuffle or $self->num_parts > 1)
    {
        assert(defined $self->imgidx);
        $self->seq($self->imgidx);
    }
    if($self->num_parts > 1)
    {
        assert($self->part_index < $self->num_parts);
        my $N = @{ $self->seq };
        my $C = $N/$self->num_parts;
        $self->seq([@{ $self->seq }[$self->part_index*$C..($self->part_index+1)*$C-1]]);
    }
    if(defined $self->aug_list or defined $self->kwargs)
    {
        $self->aug_list(AI::MXNet::Image->CreateAugmenter(data_shape => $self->data_shape, %{ $self->kwargs//{} }));
    }
    $self->cur(0);
    $self->reset();
}

method reset()
{
    if($self->shuffle)
    {
        @{ $self->seq } = List::Util::shuffle(@{ $self->seq });
    }
    if(defined $self->imgrec)
    {
        $self->imgrec->reset;
    }
    $self->cur(0);
}

method next_sample()
{
    if(defined $self->seq)
    {
        return undef if($self->cur >= @{ $self->seq });
        my $idx = $self->seq->[$self->cur];
        $self->cur($self->cur + 1);
        if(defined $self->imgrec)
        {
            my $s = $self->imgrec->read_idx($idx);
            my ($header, $img) = AI::MXNet::RecordIO->unpack($s);
            if(not defined $self->imglist)
            {
                return ($header->label, $img);
            }
            else
            {
                return ($self->imglist->{$idx}[0], $img);
            }
        }
        else
        {
            my ($label, $fname) = $self->imglist->{$idx};
            if(not defined $self->imgrec)
            {
                open(F, $self->path_root . "/$fname") or confess("can't open $fname $!");
                my $img;
                { local $/ = undef; $img = <F> };
                close(F);
                return ($label, $img);
            }
        }
    }
    else
    {
        my $s = $self->imgrec->read;
        return undef if(not defined $s);
        my ($header, $img) = AI::MXNet::RecordIO->unpack($s);
        return ($header->label, $img)
    }
}

method next()
{
    my $batch_size = $self->batch_size;
    my ($c, $h, $w) = @{ $self->data_shape };
    my $batch_data  = AI::MXNet::NDArray->empty([$batch_size, $c, $h, $w]);
    my $batch_label = AI::MXNet::NDArray->empty(@{$self->provide_label->[0]}[1]);
    my $i = 0;
    while ($i < $batch_size)
    {
        my ($label, $s) = $self->next_sample;
        last if not defined $label;
        my $data = [AI::MXNet::Image->imdecode($s)];
        if(@{ $data->[0]->shape } == 0)
        {
            AI::MXNet::Logging->debug('Invalid image, skipping.');
            next;
        }
        for my $aug (@{ $self->aug_list })
        {
            $data = [map { @{ $aug->($_) } } @$data];
        }
        for my $d (@$data)
        {
            assert(($i < $batch_size), 'Batch size must be multiples of augmenter output length');
            $batch_data->at($i)  .= AI::MXNet::NDArray->transpose($d, { axes=>[2, 0, 1] });
            $batch_label->at($i) .= $label;
            $i++;
        }
    }
    return undef if not $i;
    return AI::MXNet::DataBatch->new(data=>[$batch_data], label=>[$batch_label], pad => $batch_size-$i);
}

1;
