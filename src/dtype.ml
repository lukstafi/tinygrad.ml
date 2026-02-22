type scalar = F32 | I32 | Bool

type t =
  | Scalar of scalar
  | Ptr of scalar

let f32 = Scalar F32
let f32_ptr = Ptr F32

let size_in_bytes = function
  | Scalar F32 -> 4
  | Scalar I32 -> 4
  | Scalar Bool -> 1
  | Ptr _ -> Sys.word_size / 8

let to_c_type = function
  | Scalar F32 -> "float"
  | Scalar I32 -> "int"
  | Scalar Bool -> "bool"
  | Ptr F32 -> "float*"
  | Ptr I32 -> "int*"
  | Ptr Bool -> "bool*"
