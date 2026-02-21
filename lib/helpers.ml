(** Utility functions shared across tinygrad_ml, ported from tinygrad/helpers.py *)

let prod l = List.fold_left ( * ) 1 l
let ceildiv a b = (a + b - 1) / b
let round_up a b = ceildiv a b * b

let all_int l = List.for_all (fun x -> x >= 0) l

let dedup l =
  let seen = Hashtbl.create 16 in
  List.filter (fun x ->
    if Hashtbl.mem seen x then false
    else (Hashtbl.add seen x (); true)
  ) l

let partition f l =
  let rec go yes no = function
    | [] -> (List.rev yes, List.rev no)
    | x :: xs -> if f x then go (x :: yes) no xs else go yes (x :: no) xs
  in go [] [] l

(** Flatten a list of lists *)
let flatten = List.concat

(** Convert a string to a valid C function name *)
let to_function_name s =
  String.map (fun c -> if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c = '_' then c else '_') s

(** Strip outermost parens from a string *)
let strip_parens s =
  let n = String.length s in
  if n >= 2 && s.[0] = '(' && s.[n-1] = ')' then String.sub s 1 (n - 2)
  else s

(** Argsort: return indices that would sort the list *)
let argsort l =
  let indexed = List.mapi (fun i x -> (i, x)) l in
  let sorted = List.sort (fun (_, a) (_, b) -> compare a b) indexed in
  List.map fst sorted
