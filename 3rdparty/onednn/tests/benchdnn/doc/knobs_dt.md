# Data Types

**Benchdnn** supports the same data types as the library does (memory::data_type
enum). If an unsupported data type is specified, an error will be reported.
The following data types are supported:

| Data type | Description
| :---      | :---
| f32       | standard float
| s32       | standard int or int32_t
| s8        | standard char or int8_t
| u8        | standard unsigned char or uint8_t
| f16       | 2-byte float (1 sign bit, 5 exp bits, 10 mantissa bits)
| bf16      | 2-byte float (1 sign bit, 8 exp bits, 7 mantissa bits)

