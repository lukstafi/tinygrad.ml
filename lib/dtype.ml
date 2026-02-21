(** Data type system for tinygrad_ml, ported from tinygrad/dtype.py.
    DType represents scalar and vector data types used throughout the IR and backends. *)

(** Address space for pointer types *)
type addr_space = Global | Local | Reg

(** Scalar base type — the fundamental types that tinygrad operates on *)
type scalar =
  | Bool
  | Int8 | Int16 | Int32 | Int64
  | Uint8 | Uint16 | Uint32 | Uint64
  | Float16 | BFloat16 | Float32 | Float64

(** A dtype is either a scalar, a vector of scalars, a pointer to a dtype, or void *)
type t =
  | Scalar of scalar
  | Vec of scalar * int       (** vec(scalar, count) where count > 1 *)
  | Ptr of t * addr_space * int  (** ptr(base, addrspace, size) where size=-1 means unknown *)
  | Void

(** Image dtype for OpenCL image types — we include for completeness but won't use *)
type image_dtype = { base: t; shape: int * int }

(** Get the scalar base of any dtype *)
let scalar_of = function
  | Scalar s | Vec (s, _) -> s
  | Ptr _ -> failwith "ptr has no scalar"
  | Void -> failwith "void has no scalar"

(** Get the vector count *)
let count = function
  | Scalar _ -> 1
  | Vec (_, c) -> c
  | Ptr _ -> 1
  | Void -> 0

(** Bit size of a scalar type *)
let scalar_bitsize = function
  | Bool -> 1
  | Int8 | Uint8 -> 8
  | Int16 | Uint16 | Float16 | BFloat16 -> 16
  | Int32 | Uint32 | Float32 -> 32
  | Int64 | Uint64 | Float64 -> 64

(** Byte size of a scalar type *)
let scalar_itemsize s = (scalar_bitsize s + 7) / 8

(** Bit size of a dtype *)
let bitsize = function
  | Scalar s -> scalar_bitsize s
  | Vec (s, c) -> scalar_bitsize s * c
  | Ptr _ -> 64
  | Void -> 0

(** Byte size of a dtype *)
let itemsize dt = (bitsize dt + 7) / 8

(** Create a vector dtype *)
let vec s n =
  if n = 1 then Scalar s
  else Vec (s, n)

(** Create a pointer dtype *)
let ptr ?(size= -1) ?(addrspace=Global) base = Ptr (base, addrspace, size)

(** Get the base type of a dtype (self for non-ptr) *)
let base = function
  | Ptr (b, _, _) -> b
  | dt -> dt

(** Check if dtype is a pointer *)
let is_ptr = function Ptr _ -> true | _ -> false

(** Check if scalar is a float type *)
let is_float = function
  | Float16 | BFloat16 | Float32 | Float64 -> true
  | _ -> false

(** Check if scalar is an unsigned integer type *)
let is_unsigned = function
  | Uint8 | Uint16 | Uint32 | Uint64 | Bool -> true
  | _ -> false

(** Check if scalar is an integer type *)
let is_int s = not (is_float s) && s <> Bool

(** Priority for upcasting — higher means it wins *)
let priority = function
  | Bool -> 0
  | Int8 -> 1 | Uint8 -> 2
  | Int16 -> 3 | Uint16 -> 4
  | Float16 -> 5 | BFloat16 -> 6
  | Int32 -> 7 | Uint32 -> 8 | Float32 -> 9
  | Int64 -> 10 | Uint64 -> 11 | Float64 -> 12

(** Least upper bound in the type promotion lattice *)
let least_upper_scalar a b =
  if priority a >= priority b then a else b

(** struct format character for a scalar *)
let fmt = function
  | Bool -> "?"
  | Int8 -> "b" | Uint8 -> "B"
  | Int16 -> "h" | Uint16 -> "H"
  | Int32 -> "i" | Uint32 -> "I"
  | Int64 -> "q" | Uint64 -> "Q"
  | Float16 -> "e" | BFloat16 -> "e"  (* bf16 doesn't have a standard format *)
  | Float32 -> "f" | Float64 -> "d"

(** Name string for a scalar *)
let scalar_name = function
  | Bool -> "bool"
  | Int8 -> "char" | Int16 -> "short" | Int32 -> "int" | Int64 -> "long"
  | Uint8 -> "unsigned char" | Uint16 -> "unsigned short" | Uint32 -> "unsigned int" | Uint64 -> "unsigned long"
  | Float16 -> "half" | BFloat16 -> "__bf16"
  | Float32 -> "float" | Float64 -> "double"

(** Human-readable name for a dtype *)
let name = function
  | Scalar s -> scalar_name s
  | Vec (s, c) -> Printf.sprintf "%s%d" (scalar_name s) c
  | Ptr (b, _, _) ->
    let rec name_inner = function
      | Scalar s -> scalar_name s
      | Vec (s, c) -> Printf.sprintf "%s%d" (scalar_name s) c
      | Ptr (b, _, _) -> name_inner b ^ "*"
      | Void -> "void"
    in name_inner b ^ "*"
  | Void -> "void"

(** Pretty print for repr *)
let to_string dt =
  let sn = function
    | Bool -> "bool" | Int8 -> "int8" | Int16 -> "int16" | Int32 -> "int32" | Int64 -> "int64"
    | Uint8 -> "uint8" | Uint16 -> "uint16" | Uint32 -> "uint32" | Uint64 -> "uint64"
    | Float16 -> "float16" | BFloat16 -> "bfloat16" | Float32 -> "float" | Float64 -> "double"
  in
  match dt with
  | Scalar s -> "dtypes." ^ sn s
  | Vec (s, c) -> Printf.sprintf "dtypes.%s.vec(%d)" (sn s) c
  | Ptr _ -> "dtypes.ptr"
  | Void -> "dtypes.void"

(** Minimum value for a dtype (used for identity_element of MAX) *)
let min_val = function
  | Bool -> 0.0
  | Int8 -> -128.0 | Int16 -> -32768.0 | Int32 -> Float.of_int Int32.(to_int min_int) | Int64 -> Int.to_float Int.min_int
  | Uint8 | Uint16 | Uint32 | Uint64 -> 0.0
  | Float16 | BFloat16 | Float32 | Float64 -> Float.neg_infinity

(** Convenience constructors *)
let bool = Scalar Bool
let int8 = Scalar Int8
let int16 = Scalar Int16
let int32 = Scalar Int32
let int64 = Scalar Int64
let uint8 = Scalar Uint8
let uint16 = Scalar Uint16
let uint32 = Scalar Uint32
let uint64 = Scalar Uint64
let float16 = Scalar Float16
let bfloat16 = Scalar BFloat16
let float32 = Scalar Float32
let float64 = Scalar Float64
let void = Void
let index = int32  (* matches tinygrad's dtypes.index which is int32 *)

(** Floats set *)
let floats = [Float16; BFloat16; Float32; Float64]

(** All integer types *)
let ints = [Int8; Int16; Int32; Int64; Uint8; Uint16; Uint32; Uint64]
