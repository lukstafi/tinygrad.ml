type device =
  | Cpu_c
  | Cuda
  | Metal

let device_to_string = function
  | Cpu_c -> "cpu"
  | Cuda -> "cuda"
  | Metal -> "metal"

let device_of_string s =
  match String.lowercase_ascii s with
  | "cpu" | "cpu-c" | "c" -> Ok Cpu_c
  | "cuda" -> Ok Cuda
  | "metal" | "msl" -> Ok Metal
  | _ -> Error ("unknown device: " ^ s)

let default_device () =
  match Sys.getenv_opt "TG_DEVICE" with
  | None -> Cpu_c
  | Some s ->
      (match device_of_string s with
      | Ok d -> d
      | Error _ -> Cpu_c)

let available_devices () =
  [
    (Cpu_c, Cpu_c_backend.available ());
    (Cuda, Cuda_backend.available ());
    (Metal, Metal_backend.available ());
  ]

let run_binop ~(device : device) ~(op : Uop.binop) ~(a : Buffer.t) ~(b : Buffer.t) =
  match device with
  | Cpu_c -> Cpu_c_backend.run_binop ~op ~a ~b
  | Cuda -> Cuda_backend.run_binop ~op ~a ~b
  | Metal -> Metal_backend.run_binop ~op ~a ~b
