module type S = sig
  val device_name : string
  val available : unit -> (unit, string) result
  val run_binop : op:Uop.binop -> a:Buffer.t -> b:Buffer.t -> (Buffer.t, string) result
end
