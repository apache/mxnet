# Problem Descriptor

**benchdnn** supports two notions of a problem description: some drivers utilize
so called problem descriptors, others utilize dimensions (integer numbers)
separated by `x` delimiter. The following table describes common problem
descriptor entry names across different drivers:

| Entry      | Description
| :---       | :---
| g          | Groups (for grouped (de-)convolutions).
| mb         | Minibatch (number of images processed at once).
| ic, oc     | Input and output channels (aka feature maps).
| id, ih, iw | Input depth, height and width.
| od, oh, ow | Output depth, height and width.
| kd, kh, kw | Kernel (filter, weights) depth, height and width.
| sd, sh, sw | Stride over depth, height and width.
| dd, dh, dw | Dilation by depth, height and width.
| pd, ph, pw | Front, top and left padding.
| n          | Descriptor name.
| _          | Underscore. Optional delimiter between entries for readability.

Notes:
* Every value from the table above accepts only an integer input value. Each
  driver's documentation provides guidance on the remaining acceptable entries
  and describes supported entries in the driver's descriptors.
* Underscore delimiter can be inserted to separate entries for ease of reading
  the descriptor. It should follow the value of an entry; for example:
  `g20_mb1_ic20ih10_oc20oh10_kh3ph1`.
* Some entries use the default value across all drivers supporting the
  descriptor (unless stated otherwise in the driver documentation):
    * The default `g` value is `1`.
    * The default `mb` value is `2`.
    * The default `sd`, `sh` and `sw` value is `1`.
    * The default `dd`, `dh` and `dw` value is `0`.
* Descriptor name should be provided in a `...nNAME` way, where NAME is a
  string literal without spaces. For better readability, NAME may be surrounded
  by quotes, which should be escaped if passed from the command-line interface.
* In case any depth value `xd` is provided, the problem will be considered as a
  3D spatial problem even if a value makes the problem look like a 2D problem.
  The same applies to any height value `xh` and width value `xw`. If the `xw`
  value is not provided, there are two possible scenarios:
    * Driver supports 0D spatial (or no spatial) problems and will continue
      execution considering only minibatch and channel values.
    * Driver does not support 0D spatial problems, because it is invalid for an
      operation, and will report an error.
* The descriptor supports the only implicit rule - it promotes values from
  higher dimensions to lower ones if the latter were not specified by the user
  forming a "cubic" (for 3D spatial case) or "square" (for 2D spatial case)
  problems. E.g. for convolution if one specifies only `ic`, `oc`, `id`, `kd`,
  the driver will use default values for `g`, `mb`, `sd`, `dd`. Then it will
  deduce `od` and `pd` values based on the input, and then propagate `xd` values
  to `xh` values and then to `xw` (exact place where the implicit rule applies).
