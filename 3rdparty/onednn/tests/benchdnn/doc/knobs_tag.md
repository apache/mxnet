# Format tags

**Benchdnn** supports three kinds of memory format tags:
- Meta-tags (benchdnn abstraction).
    - Examples: `abx`, `aBx16b`
- Library tags (refer to `dnnl::memory::format_tag` enum).
    - Examples: `nchw`, `acdb`, `nChw8c`
- Other valid tags not presented in `dnnl::memory::format_tag` enum (controlled by
  `--allow-enum-tags-only` option).
    - Examples: `abcdefghij`, `Ab3a`

If an unsupported tag is specified, an error will be reported. The list of library
supported tags can be found in dnnl.hpp header file. Meta-tags are xD-spatial
tags which adapt to the number of dimensions specified by a problem descriptor
(for descriptor-based drivers) or dimensions. Below are examples of plain and blocked
meta-tags:

| Plain tags   | Description
| :---         | :---
| abx          | Includes `a`, `ab`, `abc`, `abcd`, `abcde`, `abcdef` tags and their former names for activations and weights.
| axb          | Includes `a`, `ab`, `acb`, `acdb`, `acdeb` tags and their former names for activations.
| xba          | Includes `a`, `ba`, `cba`, `cdba`, `cdeba` tags and their former names for weights.

| Blocked tags | Description
| :---         | :---
| aBx4b        | Includes `aBc4b`, `aBcd4b`, `aBcde4b` tags and their former names for activations.
| aBx8b        | Includes `aBc8b`, `aBcd8b`, `aBcde8b` tags and their former names for activations.
| aBx16b       | Includes `aBc16b`, `aBcd16b`, `aBcde16b` tags and their former names for activations.
| ABx16a16b    | Includes `ABc16a16b`, `ABcd16a16b`, `ABcde16a16b` tags and their former names for activations.
| aBx32b       | Includes `aBc32b`, `aBcd32b`, `aBcde32b` tags and their former names for activations.
| ABx32a32b    | Includes `ABc32a32b`, `ABcd32a32b`, `ABcde32a32b` tags and their former names for activations.

The following special tags are supported:

| Special tags | Description
| :---         | :---
| any          | Refer to ``Special tags`` below.
| undef        | Refer to ``Special tags`` below.

## Special tags

### Any

Special tag `any` corresponds to the `dnnl_format_tag_any` format tag and
instructs the driver to follow the library programming model. Once passed to the
driver, lets the library decide which physical layout will be used for a certain
memory descriptor of the given problem. Only supported by those drivers whose
primitives programming model supports the `dnnl_format_tag_any` format tag
for memory descriptors.

### Undef

Special tag `undef` corresponds to the `dnnl_format_tag_undef` format tag and
instructs the driver to pass the null pointer for a specified memory descriptor
when creating a primitive descriptor, allowing the library to construct a memory
descriptor based on other input arguments. Only supported by those drivers whose
primitives programming model supports passing null pointers (C API) or omit
certain memory descriptors (C++ API). When passing `undef` as a tag, all other
input settings for that memory descriptor will not be considered, such as data
type.
