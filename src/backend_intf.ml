module type S = sig
  val device_name : string
  val available : unit -> (unit, string) result
  val run_expr : expr:Uop.expr -> inputs:Buffer.t list -> shape:int array -> (Buffer.t, string) result
  val run_reduce :
    op:Uop.reduce_op -> expr:Uop.expr -> inputs:Buffer.t list -> shape:int array -> (float, string) result
  val run_binop : op:Uop.binop -> a:Buffer.t -> b:Buffer.t -> (Buffer.t, string) result
end
