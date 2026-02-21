type scalar = F32

type t =
  | Scalar of scalar
  | Ptr of scalar

let f32 = Scalar F32
let f32_ptr = Ptr F32

let size_in_bytes = function
  | Scalar F32 -> 4
  | Ptr _ -> Sys.word_size / 8

let to_c_type = function
  | Scalar F32 -> "float"
  | Ptr F32 -> "float*"
