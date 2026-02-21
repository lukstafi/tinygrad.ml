(** Renderer base types and ProgramSpec.
    Ported from tinygrad/renderer/__init__.py *)

(** A compiled program ready for execution *)
type program_spec = {
  name: string;
  src: string;
  device: string;
  global_size: int list;     (** grid dimensions *)
  local_size: int list option;  (** block dimensions, None for CPU *)
  globals: int list;         (** param indices for buffer pointers *)
  outs: int list;            (** which params are output buffers *)
  ins: int list;             (** which params are input buffers *)
  vars: (string * int * int) list;  (** (name, min, max) for runtime variables *)
}

let make_program_spec ~name ~src ~device
    ?(global_size=[1;1;1]) ?local_size
    ?(globals=[]) ?(outs=[]) ?(ins=[]) ?(vars=[]) () =
  { name; src; device; global_size; local_size; globals; outs; ins; vars }

(** Renderer interface — each backend implements this *)
module type S = sig
  val device : string
  val has_local : bool
  val has_shared : bool
  val global_max : int * int * int
  val local_max : (int * int * int) option
  val shared_max : int

  (** Render a list of linearized UOps into source code + program spec *)
  val render : Uop.t list -> program_spec
end
