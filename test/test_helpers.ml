let check_close ~msg a b =
  let is_finite x =
    match classify_float x with
    | FP_normal | FP_subnormal | FP_zero -> true
    | FP_infinite | FP_nan -> false
  in
  if not (is_finite a && is_finite b) then begin
    if a <> b then failwith (Printf.sprintf "%s: expected %.8f, got %.8f" msg a b)
  end else begin
    let eps = max 1e-6 (1e-6 *. Float.abs a) in
    if Float.abs (a -. b) > eps then
      failwith (Printf.sprintf "%s: expected %.8f, got %.8f" msg a b)
  end

let check_array ~msg expected got =
  if Array.length expected <> Array.length got then
    failwith
      (Printf.sprintf "%s: length mismatch expected=%d got=%d"
         msg (Array.length expected) (Array.length got));
  Array.iteri
    (fun i e -> check_close ~msg:(Printf.sprintf "%s[%d]" msg i) e got.(i))
    expected
